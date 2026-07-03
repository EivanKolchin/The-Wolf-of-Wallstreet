"""StrategyAgent — runs the validated managed-beta book live, alongside the NN agent.

The managed-beta book is a DAILY, multi-asset, TARGET-WEIGHT strategy (hold each asset at a weight
set by trend × vol-target × macro de-risk), which is a different shape from the NN agent's discrete
SL/TP trades. So this agent owns its own thin reconciliation loop: each rebalance it computes the
current target weights (``managed_book.target_weights`` — the exact backtest logic) and issues the
DELTA orders to move holdings toward them, skipping deltas below a threshold to bound turnover.

SAFETY (honours the project's never-execute-trades rule):
  * Paper by default. The default order router is an in-memory PaperBook (simulates fills + cost,
    tracks weights/turnover) — it never touches a broker.
  * Live routing is intentionally NOT implemented: with ``paper=False`` the agent refuses to run
    unless ``live_router`` is explicitly supplied by the operator. We never auto-wire real money.
  * Reuses the existing RiskManager gate when provided (drawdown halt, exposure caps).

Dependencies are small injected interfaces (bar provider, portfolio, optional risk gate), so the
core ``rebalance_once`` is unit-testable with fakes and free of redis/db/broker coupling.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Dict, List, Optional

import pandas as pd

try:
    import structlog
    logger = structlog.get_logger("strategy_agent")
except Exception:  # pragma: no cover
    import logging
    logger = logging.getLogger("strategy_agent")

from backend.strategies.managed_beta import ManagedBetaParams
from backend.strategies.managed_book import target_weights
from backend.strategies.combined_book import combined_target_weights
from backend.strategies.ts_momentum import TSMomentumParams
from backend.risk.news_overlay import assess_news_risk, NewsOverlayConfig, NewsRiskAction


@dataclass
class DailyBarProvider:
    """Live daily-bar source for the managed-beta universe (yfinance covers ETFs AND crypto:
    SPY/QQQ/TLT/GLD/TQQQ + BTC-USD/ETH-USD + ^VIX). In-memory TTL cache so a daily rebalance
    re-pulls at most a couple of times a day. Returns a [timestamp, OHLCV] frame or None."""
    start: str = "2015-01-01"
    ttl_seconds: float = 43_200.0                          # 12h
    _cache: Dict[str, tuple] = field(default_factory=dict)  # symbol -> (fetched_at, DataFrame)

    def get_daily(self, symbol: str):
        import time
        now = time.time()
        hit = self._cache.get(symbol)
        if hit is not None and now - hit[0] < self.ttl_seconds:
            return hit[1]
        try:
            import yfinance as yf
            raw = yf.download(symbol, start=self.start, interval="1d", auto_adjust=True,
                              progress=False, threads=False)
            if raw is None or raw.empty:
                return hit[1] if hit else None
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            df = pd.DataFrame({
                "timestamp": pd.to_datetime(raw.index),
                "open": raw["Open"].to_numpy(float), "high": raw["High"].to_numpy(float),
                "low": raw["Low"].to_numpy(float), "close": raw["Close"].to_numpy(float),
                "volume": raw["Volume"].to_numpy(float),
            }).dropna(subset=["close"]).reset_index(drop=True)
            self._cache[symbol] = (now, df)
            return df
        except Exception as e:
            logger.warning("daily_bar_fetch_failed", symbol=symbol, error=str(e)[:100])
            return hit[1] if hit else None


@dataclass
class PaperBook:
    """In-memory target-weight paper book for shadow running: holds signed weights, MARKS them
    to market so equity reflects real P&L (not just cost drag), charges round-trip cost on
    rebalances, and keeps a net-worth history for the dashboard equity curve. No real money."""
    equity: float = 100_000.0
    cost_bps: float = 5.0
    positions: Dict[str, float] = field(default_factory=dict)   # symbol -> signed weight
    turnover: float = 0.0
    orders: List[dict] = field(default_factory=list)
    initial_equity: float = 0.0
    last_prices: Dict[str, float] = field(default_factory=dict)  # symbol -> last mark price
    net_worth_history: List[dict] = field(default_factory=list)  # [{ts, value}] for the curve
    realized_cost: float = 0.0
    max_history: int = 5000

    def __post_init__(self):
        if self.initial_equity <= 0.0:
            self.initial_equity = self.equity
        if not self.net_worth_history:
            self.net_worth_history.append({"ts": __import__("time").time(),
                                           "value": round(self.equity, 2)})

    async def snapshot(self) -> dict:
        return {"equity": self.equity, "positions": dict(self.positions)}

    async def apply(self, sym: str, target_w: float, delta: float) -> None:
        self.turnover += abs(delta)
        cost = abs(delta) * self.equity * (self.cost_bps / 1e4)          # rebalance cost
        self.equity -= cost
        self.realized_cost += cost
        self.positions[sym] = target_w
        self.orders.append({"symbol": sym, "target_weight": target_w, "delta": delta})

    def mark_to_market(self, prices: Dict[str, float], now_ts: float) -> float:
        """Apply each held position's price return SINCE the last mark to equity, then record a
        net-worth history point. Returns the marked gross return. Positions are weight fractions
        of equity, so the book return = Σ wᵢ·(pᵢ/pᵢ_prev − 1). Symbols without a prior price
        (just added) contribute 0 this step and seed their price for the next."""
        gross_ret = 0.0
        for sym, w in self.positions.items():
            p_now = prices.get(sym)
            p_prev = self.last_prices.get(sym)
            if p_now and p_prev and p_prev > 0 and abs(w) > 1e-9:
                gross_ret += float(w) * (float(p_now) / float(p_prev) - 1.0)
        self.equity *= (1.0 + gross_ret)
        for sym, p in prices.items():
            if p and float(p) > 0:
                self.last_prices[sym] = float(p)
        self.net_worth_history.append({"ts": float(now_ts), "value": round(self.equity, 2)})
        if len(self.net_worth_history) > self.max_history:
            self.net_worth_history = self.net_worth_history[-self.max_history:]
        return gross_ret

    def portfolio_view(self) -> dict:
        """Dashboard payload: total value, cash, per-allocation dollar values, net-worth curve."""
        gross = sum(abs(w) for w in self.positions.values())
        cash = self.equity * (1.0 - gross)                              # uninvested (neg = levered)
        allocations = []
        for sym, w in sorted(self.positions.items(), key=lambda kv: -abs(kv[1])):
            if abs(w) < 1e-9:
                continue
            allocations.append({
                "symbol": sym, "weight": round(float(w), 4),
                "value": round(float(w) * self.equity, 2),
                "side": "long" if w > 0 else "short",
            })
        pnl = self.equity - self.initial_equity
        return {
            "total_value": round(self.equity, 2),
            "initial_value": round(self.initial_equity, 2),
            "cash": round(cash, 2),
            "gross_exposure": round(gross, 4),
            "net_exposure": round(sum(self.positions.values()), 4),
            "total_pnl": round(pnl, 2),
            "total_pnl_pct": round((pnl / self.initial_equity) * 100.0, 4) if self.initial_equity else 0.0,
            "allocations": allocations,
            "history": self.net_worth_history[-1000:],
            "turnover": round(self.turnover, 4),
            "total_cost": round(self.realized_cost, 2),
        }


@dataclass
class StrategyAgent:
    universe: List[str]
    bar_provider: object                                  # async get_daily(symbol) -> DataFrame|None
    portfolio: object = field(default_factory=PaperBook)  # async snapshot(); async apply(sym, w, d)
    risk_manager: object = None                           # optional .approve(decision, state)
    params: ManagedBetaParams = field(default_factory=ManagedBetaParams)
    target_vol: float = 0.15
    max_leverage: float = 2.0
    alloc: str = "equal"
    min_rebalance_delta: float = 0.02                     # skip weight changes smaller than this
    rebalance_seconds: float = 86_400.0                   # daily
    paper: bool = True
    live_router: Optional[Callable[[str, float, float], Awaitable[None]]] = None
    vix_symbol: str = "^VIX"
    # ── optional 2nd sleeve: 4h regime-gated TS-momentum (the validated diversifier) ──
    ts_bar_provider: object = None                        # get_bars(symbol)->4h OHLCV|None (or get_daily)
    ts_universe: List[str] = field(default_factory=list)  # crypto perps for the TS sleeve
    ts_params: TSMomentumParams = field(default_factory=TSMomentumParams)
    w_managed: float = 0.5                                # capital fraction to the managed sleeve
    w_ts: float = 0.5                                     # capital fraction to the TS sleeve
    # ── optional news → risk overlay (directional, position-aware; tighten-only) ──
    news_provider: object = None                          # impacts_for(symbol)->list | callable(symbol)
    news_cfg: NewsOverlayConfig = field(default_factory=NewsOverlayConfig)
    news_verifier: object = None                          # optional LLM verifier: async verdict(...)→NewsRiskAction|None
    # optional cross-asset propagation: EntityGraph whose edge weights are re-validated
    # against the bars each rebalance (dead narratives decay to 0 automatically)
    entity_graph: object = None
    # optional DMN/QuantileTCN overlay gate on the TS sleeve (OOS-validated timing skill;
    # scales perp weights in [floor, 1], never originates — fails open to the baseline)
    overlay_gate: object = None
    _asset_halts: Dict[str, float] = field(default_factory=dict)   # symbol -> halt-until epoch sec
    _stop: asyncio.Event = field(default_factory=asyncio.Event)

    def __post_init__(self):
        if not self.paper and self.live_router is None:
            # Never auto-route real orders. Live requires an explicit operator-provided router.
            raise RuntimeError("StrategyAgent: live mode requires an explicit live_router — "
                               "real order routing is intentionally not auto-wired (never-execute).")

    async def _get_bars(self) -> Dict[str, pd.DataFrame]:
        out: Dict[str, pd.DataFrame] = {}
        for s in self.universe:
            try:
                df = self.bar_provider.get_daily(s)
                if asyncio.iscoroutine(df):
                    df = await df
                if df is not None and len(df) > 5:
                    out[s] = df
            except Exception as e:
                logger.warning("bar_fetch_failed", symbol=s, error=str(e)[:100])
        return out

    async def _maybe_vix(self, bars: Dict[str, pd.DataFrame]):
        try:
            v = self.bar_provider.get_daily(self.vix_symbol)
            if asyncio.iscoroutine(v):
                v = await v
            return v
        except Exception:
            return None

    async def _get_ts_bars(self) -> Dict[str, pd.DataFrame]:
        """Fetch 4h bars for the TS-momentum sleeve (uses get_bars, falling back to get_daily)."""
        out: Dict[str, pd.DataFrame] = {}
        prov = self.ts_bar_provider
        if prov is None or not self.ts_universe:
            return out
        getter = getattr(prov, "get_bars", None) or getattr(prov, "get_daily", None)
        for s in self.ts_universe:
            try:
                df = getter(s)
                if asyncio.iscoroutine(df):
                    df = await df
                if df is not None and len(df) > 5:
                    out[s] = df
            except Exception as e:
                logger.warning("ts_bar_fetch_failed", symbol=s, error=str(e)[:100])
        return out

    async def _news_impacts_raw(self, symbol: str) -> list:
        prov = self.news_provider
        if prov is None:
            return []
        try:
            fn = prov if callable(prov) else getattr(prov, "impacts_for", None)
            if fn is None:
                return []
            res = fn(symbol)
            if asyncio.iscoroutine(res):
                res = await res
            return list(res or [])
        except Exception as e:
            logger.warning("news_impacts_failed", symbol=symbol, error=str(e)[:100])
            return []

    async def _news_impacts(self, symbol: str) -> list:
        """Recent NewsImpact-like objects for a symbol: the symbol's OWN news, plus news of
        related source assets propagated through the entity graph (magnitude/confidence
        scaled by the correlation-validated live edge weight — a BTC crash headline reaches
        MSTR/COIN in proportion to how much they still trade as crypto proxies)."""
        impacts = await self._news_impacts_raw(symbol)
        g = self.entity_graph
        if g is None:
            return impacts
        try:
            from backend.signals.entity_graph import source_symbols
            for src_asset, w in g.related_sources(symbol):
                seen = 0
                for key in source_symbols(src_asset):
                    src_impacts = await self._news_impacts_raw(key)
                    for imp in src_impacts:
                        impacts.append(g.derive(imp, src_asset, symbol, w))
                        seen += 1
                    if src_impacts:
                        break                      # one resolution of the source is enough
                if seen:
                    logger.info("news_propagated_via_edge", src=src_asset, dst=symbol,
                                weight=round(w, 2), n=seen)
        except Exception as e:
            logger.warning("news_propagation_failed", symbol=symbol, error=str(e)[:100])
        return impacts

    async def _apply_news(self, symbol: str, target_w: float, now_sec: float) -> tuple[float, str]:
        """Adjust a target weight by the directional news overlay. Returns (adjusted_weight, note).
        Honours an active per-asset halt; records a new halt when the overlay calls for one. The
        de-risking trade toward the (smaller/zero) target is NOT blocked — selling out of danger is
        the point."""
        # already inside a halt window → force flat
        until = self._asset_halts.get(symbol, 0.0)
        if until > now_sec:
            return 0.0, f"halted_until_{int(until)}"
        if until and until <= now_sec:
            self._asset_halts.pop(symbol, None)             # halt expired
        if self.news_provider is None:
            return target_w, ""
        impacts = await self._news_impacts(symbol)
        if not impacts:
            return target_w, ""
        direction = "long" if target_w > 0 else "short" if target_w < 0 else "hold"
        action = assess_news_risk(symbol, direction, impacts, cfg=self.news_cfg)
        if self.news_verifier is not None:                  # optional LLM cross-check (tighten-only)
            try:
                verdict = self.news_verifier.verdict(symbol, direction, impacts, action)
                if asyncio.iscoroutine(verdict):
                    verdict = await verdict
                if verdict is not None:
                    from backend.risk.news_overlay import _tighten_only
                    action = _tighten_only(action, verdict)
            except Exception as e:
                logger.warning("news_verifier_failed", symbol=symbol, error=str(e)[:100])
        if action.action == "halt_asset" and action.halt_seconds > 0:
            self._asset_halts[symbol] = now_sec + action.halt_seconds
        if action.action != "allow":
            logger.info("news_overlay_applied", symbol=symbol, action=action.action,
                        size_scale=round(action.size_scale, 3), reason=action.reason)
        return target_w * action.size_scale, action.reason if action.action != "allow" else ""

    async def rebalance_once(self) -> dict:
        """One rebalance: compute target weights, reconcile vs current holdings, route the deltas.
        Returns a plan dict {targets, orders, skipped, blocked} — observable + unit-testable."""
        bars = await self._get_bars()
        if len(bars) < 2:
            logger.warning("strategy_rebalance_insufficient_data", n=len(bars))
            return {"targets": {}, "orders": [], "skipped": [], "blocked": "insufficient_data"}

        vix_df = await self._maybe_vix(bars)
        vix = vix_df["close"].to_numpy() if vix_df is not None and "close" in getattr(vix_df, "columns", []) else None

        ts_bars = await self._get_ts_bars()
        if self.entity_graph is not None:
            try:   # re-validate edge weights against this rebalance's data (decays stale edges)
                self.entity_graph.refresh({**bars, **ts_bars})
            except Exception as e:
                logger.warning("entity_graph_refresh_failed", error=str(e)[:100])

        # MARK TO MARKET: revalue the positions we ALREADY hold at the latest prices BEFORE we
        # rebalance, so the paper book's equity reflects real P&L (gains/losses on held names),
        # not just cost drag. Only the paper book supports this; a live broker holds real value.
        prices = self._latest_prices({**bars, **ts_bars})
        if prices and hasattr(self.portfolio, "mark_to_market"):
            try:
                self.portfolio.mark_to_market(prices, __import__("time").time())
            except Exception as e:
                logger.warning("mark_to_market_failed", error=str(e)[:100])
        overlay_notes = {}
        if ts_bars:                                       # two-sleeve combined book
            targets = combined_target_weights(
                bars, ts_bars, w_managed=self.w_managed, w_ts=self.w_ts,
                managed_params=self.params, ts_params=self.ts_params,
                managed_kwargs=dict(target_vol=self.target_vol, max_leverage=self.max_leverage,
                                    alloc=self.alloc, vix=vix))
        else:                                             # managed-beta only (backward-compatible)
            targets = target_weights(bars, self.params, target_vol=self.target_vol,
                                     max_leverage=self.max_leverage, alloc=self.alloc, vix=vix)

        if self.overlay_gate is not None and ts_bars:
            try:   # scale perp weights by the validated DMN gate (fails open on any error)
                overlay_notes = self.overlay_gate.apply(targets, list(ts_bars))
            except Exception as e:
                logger.warning("overlay_gate_failed", error=str(e)[:120])

        snap = await self.portfolio.snapshot()
        cur = snap.get("positions", {})
        equity = float(snap.get("equity", 0.0))
        now_sec = __import__("time").time()
        orders, skipped, news_notes = [], [], {}
        # reconcile every symbol that is targeted OR currently held (so dropped names get flattened)
        for s in sorted(set(targets) | set(cur)):
            tw = float(targets.get(s, 0.0))
            tw, note = await self._apply_news(s, tw, now_sec)   # directional news overlay
            if note:
                news_notes[s] = note
            cw = float(cur.get(s, 0.0))
            delta = tw - cw
            if abs(delta) < self.min_rebalance_delta:
                skipped.append(s)
                continue
            if self.risk_manager is not None and hasattr(self.risk_manager, "is_halted") \
                    and self.risk_manager.is_halted:
                return {"targets": targets, "orders": orders, "skipped": skipped,
                        "news": news_notes, "blocked": "risk_halted"}
            await self._route(s, tw, delta, equity)
            orders.append({"symbol": s, "target_weight": tw, "delta": delta})
        logger.info("strategy_rebalanced", n_orders=len(orders), n_skipped=len(skipped),
                    sleeves=("2" if ts_bars else "1"), n_news=len(news_notes),
                    n_overlay=len(overlay_notes),
                    gross=float(sum(abs(v) for v in targets.values())))
        await self._publish_portfolio()
        return {"targets": targets, "orders": orders, "skipped": skipped,
                "news": news_notes, "overlay": overlay_notes, "blocked": None}

    @staticmethod
    def _latest_prices(all_bars: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """Latest close per symbol from the fetched bars (the mark-to-market prices). Keys match
        the target-weight keys (managed: SPY/BTC-USD…, TS: BTCUSDT…)."""
        out: Dict[str, float] = {}
        for sym, df in all_bars.items():
            try:
                if df is not None and "close" in getattr(df, "columns", []) and len(df):
                    px = float(df["close"].to_numpy()[-1])
                    if px > 0:
                        out[sym] = px
            except Exception:
                continue
        return out

    async def _publish_portfolio(self) -> None:
        """Publish the paper book's dashboard view to Redis (``strategy:portfolio``) so the API
        process (which doesn't share memory with this agent process) can serve it. Best-effort —
        a Redis hiccup never breaks a rebalance, and a live (non-paper) book simply has no view."""
        if not hasattr(self.portfolio, "portfolio_view"):
            return
        try:
            import json as _json
            from backend.memory.redis_client import get_redis
            view = self.portfolio.portfolio_view()
            view["paper"] = bool(self.paper)
            view["updated_at"] = __import__("time").time()
            r = await get_redis()
            await r.set("strategy:portfolio", _json.dumps(view))
        except Exception as e:
            logger.debug("strategy_portfolio_publish_failed", error=str(e)[:100])

    async def _route(self, sym: str, target_w: float, delta: float, equity: float) -> None:
        if self.paper:
            await self.portfolio.apply(sym, target_w, delta)
        else:
            await self.live_router(sym, target_w, delta)   # operator-supplied; never auto-wired

    async def run(self) -> None:
        mode = "PAPER" if self.paper else "LIVE(operator-router)"
        logger.info("strategy_agent_started", universe=self.universe, mode=mode,
                    rebalance_seconds=self.rebalance_seconds)
        while not self._stop.is_set():
            try:
                await self.rebalance_once()
            except Exception as e:
                logger.error("strategy_rebalance_error", error=str(e)[:200])
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self.rebalance_seconds)
            except asyncio.TimeoutError:
                pass

    def stop(self) -> None:
        self._stop.set()
