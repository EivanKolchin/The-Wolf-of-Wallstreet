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
    max_history: int = 9000                                      # ~1yr of hourly marks for the range buttons

    def __post_init__(self):
        if self.initial_equity <= 0.0:
            self.initial_equity = self.equity
        if not self.net_worth_history:
            self.net_worth_history.append({"ts": __import__("time").time(),
                                           "value": round(self.equity, 2)})

    async def snapshot(self) -> dict:
        return {"equity": self.equity, "positions": dict(self.positions)}

    def to_state(self) -> dict:
        """Full serialisable state for cross-restart persistence (Redis). Everything needed to
        resume the book exactly where it left off — so a backend restart no longer resets equity
        to the seed and re-pays the establishment cost."""
        return {
            "equity": float(self.equity),
            "initial_equity": float(self.initial_equity),
            "cost_bps": float(self.cost_bps),
            "positions": {k: float(v) for k, v in self.positions.items()},
            "turnover": float(self.turnover),
            "realized_cost": float(self.realized_cost),
            "last_prices": {k: float(v) for k, v in self.last_prices.items()},
            "net_worth_history": list(self.net_worth_history[-self.max_history:]),
            "peak_equity": float(getattr(self, "peak_equity", self.equity)),
        }

    def load_state(self, s: dict) -> None:
        """Restore from a ``to_state`` dict. Missing keys keep their current values (forward-compat)."""
        if not isinstance(s, dict) or not s:
            return
        self.equity = float(s.get("equity", self.equity))
        self.initial_equity = float(s.get("initial_equity", self.initial_equity)) or self.equity
        self.cost_bps = float(s.get("cost_bps", self.cost_bps))
        self.positions = {str(k): float(v) for k, v in (s.get("positions") or {}).items()}
        self.turnover = float(s.get("turnover", 0.0))
        self.realized_cost = float(s.get("realized_cost", 0.0))
        self.last_prices = {str(k): float(v) for k, v in (s.get("last_prices") or {}).items()}
        hist = s.get("net_worth_history")
        if isinstance(hist, list) and hist:
            self.net_worth_history = hist[-self.max_history:]
        self.peak_equity = float(s.get("peak_equity", self.equity)) or self.equity

    async def apply(self, sym: str, target_w: float, delta: float) -> None:
        self.turnover += abs(delta)
        cost = abs(delta) * self.equity * (self.cost_bps / 1e4)          # rebalance cost
        self.equity -= cost
        self.realized_cost += cost
        self.positions[sym] = target_w
        self.orders.append({"symbol": sym, "target_weight": target_w, "delta": delta})

    def mark_to_market(self, prices: Dict[str, float], now_ts: float,
                       record_history: bool = True) -> float:
        """Apply each held position's price return SINCE the last mark to equity, then (optionally)
        record a net-worth history point. Returns the marked gross return. Positions are weight
        fractions of equity, so the book return = Σ wᵢ·(pᵢ/pᵢ_prev − 1). Symbols without a prior
        price (just added) contribute 0 this step and seed their price for the next.

        ``record_history=False`` updates equity / last_prices / drawdown but skips appending to the
        net-worth curve — used by the fast intra-rebalance mark loop, which revalues every ~20s
        (so the dashboard number moves live) but only persists a curve point every few minutes to
        keep the history from ballooning.

        Side effects for the journal/post-mortem loop: ``last_contributions`` holds each held
        symbol's return contribution this mark; ``current_drawdown`` tracks the peak-to-now DD."""
        gross_ret = 0.0
        repriced = False
        self.last_contributions = {}
        for sym, w in self.positions.items():
            p_now = prices.get(sym)
            p_prev = self.last_prices.get(sym)
            if p_now and p_prev and p_prev > 0 and abs(w) > 1e-9:
                if float(p_now) != float(p_prev):
                    repriced = True
                c = float(w) * (float(p_now) / float(p_prev) - 1.0)
                gross_ret += c
                self.last_contributions[sym] = c
        # A "stale" mark (every held symbol returned its previous price — a failed/cached fetch,
        # a closed market, or a weekend) has gross_ret==0. Flag it so callers can skip recording a
        # flat curve point and skip feeding fabricated 0.0 returns into the online learning loop.
        self.last_mark_repriced = repriced
        self.equity *= (1.0 + gross_ret)
        for sym, p in prices.items():
            if p and float(p) > 0:
                self.last_prices[sym] = float(p)
        self.peak_equity = max(getattr(self, "peak_equity", self.equity), self.equity)
        self.current_drawdown = (self.equity / self.peak_equity - 1.0) if self.peak_equity else 0.0
        # Only persist a curve point when prices actually moved — a stale fetch would otherwise
        # freeze then gap the curve (reads as "always red") and pad the history with flat points.
        # If nothing is held yet (fresh book, positions added post-mark), still record so the curve
        # has a baseline heartbeat.
        if record_history and (repriced or not self.positions):
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
            "history": self.net_worth_history[-9000:],   # full curve; the frontend range buttons window it
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
    rebalance_seconds: float = 86_400.0                   # daily (target-weight change cadence)
    # Fast intra-rebalance mark loop: revalue held positions at LIVE prices this often and
    # republish, so the dashboard net worth MOVES in real time between the daily rebalances
    # (target weights / routing still only change on rebalance_seconds).
    mark_seconds: float = 20.0
    # Persist a net-worth curve point at most this often (equity itself updates every mark_seconds,
    # but a curve point every 20s would balloon the history) — keeps ~max_history*this of curve.
    history_interval_seconds: float = 300.0
    paper: bool = True
    live_router: Optional[Callable[[str, float, float], Awaitable[None]]] = None
    vix_symbol: str = "^VIX"
    # ── optional 2nd sleeve: 4h regime-gated TS-momentum (the validated diversifier) ──
    ts_bar_provider: object = None                        # get_bars(symbol)->4h OHLCV|None (or get_daily)
    ts_universe: List[str] = field(default_factory=list)  # crypto perps for the TS sleeve
    ts_params: TSMomentumParams = field(default_factory=TSMomentumParams)
    w_managed: float = 0.5                                # capital fraction to the managed sleeve
    w_ts: float = 0.5                                     # capital fraction to the TS sleeve
    # book-level vol target: after both sleeves combine, scale the WHOLE book to this annualized
    # vol (from trailing cov of actual holdings), capped by book_max_leverage. 0 = off. This is the
    # risk lever the backtest validated — without it the two internally-targeted sleeves under-risk.
    book_vol_target: float = 0.0
    book_max_leverage: float = 2.0
    # book-equity drawdown de-gear (validated drawdown_degear formula, off the book's OWN equity)
    dd_degear_threshold: float = 0.10
    dd_degear_floor: float = 0.25
    # Kelly-conviction sizing: tilt weights by signed trend strength (bet bigger on strong trends).
    # 0 = off (neutral). conviction_cap bounds the tilt multiplier to [1/cap, cap].
    conviction_gain: float = 0.0
    conviction_cap: float = 2.0
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
    # append-only JSONL journal of every rebalance/order/mark/outcome — the reviewable record
    # the owner periodically hands to a strong model for deep post-mortem. Records, never decides.
    journal: object = None
    # optional cross-sectional anomaly scanner (wide-spread / illiquidity de-gear; tighten-only)
    scanner: object = None
    _asset_halts: Dict[str, float] = field(default_factory=dict)   # symbol -> halt-until epoch sec
    _stop: asyncio.Event = field(default_factory=asyncio.Event)
    _last_hist_ts: float = 0.0                            # last net-worth curve point wall-clock
    _last_rebalance_ts: float = 0.0                       # last full rebalance wall-clock
    _started_at: float = 0.0                              # wall-clock this agent's run() began

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
                if df is not None and len(df) > 6:
                    # Drop the LAST row — it's the in-progress 4h candle. Evaluating the Donchian
                    # breakout on a still-forming bar repaints (a breakout can trigger intrabar and
                    # then close back inside the channel), producing false entries the closed-bar
                    # backtest never saw. Trim it so signals only ever see completed candles.
                    out[s] = df.iloc[:-1].reset_index(drop=True)
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
        # De-gearing a flat/hold target (|w|≈0) is a no-op — skip the overlay so routine news
        # doesn't flood the journal/log with cosmetic 'hold' de-gears (measured news IC ≈ 0; the
        # overlay's value is protecting REAL exposure, not annotating zero-weight positions).
        if abs(target_w) < 1e-9:
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

    async def rebalance_once(self, live_prices: Optional[Dict[str, float]] = None) -> dict:
        """One rebalance: compute target weights, reconcile vs current holdings, route the deltas.
        Returns a plan dict {targets, orders, skipped, blocked} — observable + unit-testable.

        ``live_prices`` (optional; supplied by the production ``run()`` loop) overrides the bar-close
        mark prices for held names, so this mark is CONTINUOUS with the fast intra-rebalance marks.
        Without it, re-marking on an older bar close after a run of live marks saw-tooths the equity
        curve and injects that artifact into the journaled ``book_return`` the overlay learns from.
        Left None by unit tests, which keeps this method pure — no network I/O."""
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
        # Live quotes (when injected) win over bar closes for the mark; bar closes remain the
        # fallback and still seed prices for names being opened this rebalance.
        if live_prices:
            prices.update({k: v for k, v in live_prices.items() if v and v > 0})
        if prices and hasattr(self.portfolio, "mark_to_market"):
            try:
                book_ret = self.portfolio.mark_to_market(prices, __import__("time").time())
                self._after_mark(book_ret)
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

        if self.scanner is not None and ts_bars:
            try:   # anomaly scanner: tighten-only de-gear on wide-spread / illiquid perps
                for s in list(ts_bars):
                    if s in targets and targets[s] != 0.0:
                        scale = float(self.scanner.degear_scale(s))
                        if scale < 1.0:
                            targets[s] *= scale
                            logger.info("anomaly_degear", symbol=s, scale=round(scale, 3))
            except Exception as e:
                logger.warning("anomaly_scanner_failed", error=str(e)[:100])

        # KELLY-CONVICTION SIZING: tilt weights toward the highest-conviction positions (strong,
        # persistent trends in the position's own direction) BEFORE the book vol-target re-normalizes
        # gross. Fractional-Kelly in spirit — conviction ≈ edge proxy, vol-target handles variance —
        # so a strong trend gets a bigger bet and a marginal one gets a smaller bet, without changing
        # total book risk. tanh-saturated + capped so it can never blow up. Off unless gain>0.
        if self.conviction_gain and self.conviction_gain > 0.0:
            try:
                conv = self._conviction_scale(targets, {**bars, **ts_bars})
                if conv:
                    targets = {s: w * conv.get(s, 1.0) for s, w in targets.items()}
                    tilted = {s: round(c, 2) for s, c in conv.items() if abs(c - 1.0) > 0.05}
                    if tilted:
                        logger.info("conviction_tilt_applied", tilts=tilted)
            except Exception as e:
                logger.warning("conviction_tilt_failed", error=str(e)[:120])

        # BOOK-LEVEL VOL TARGET: scale the whole combined book to the intended annualized vol from
        # the trailing realized vol of the actual holdings (capped by book_max_leverage). The two
        # sleeves self-target their own internal vols; this re-risks the COMBINED book to intent.
        if self.book_vol_target and self.book_vol_target > 0.0:
            try:
                rv = self._book_realized_vol(targets, {**bars, **ts_bars})
                if rv and rv > 1e-6:
                    gross = sum(abs(v) for v in targets.values())
                    scale = self.book_vol_target / rv
                    if gross > 0 and gross * scale > self.book_max_leverage:
                        scale = self.book_max_leverage / gross     # respect the leverage cap
                    if abs(scale - 1.0) > 1e-3:
                        targets = {s: w * scale for s, w in targets.items()}
                        logger.info("book_vol_target_applied", realized_vol=round(rv, 4),
                                    target_vol=self.book_vol_target, scale=round(scale, 3),
                                    new_gross=round(sum(abs(v) for v in targets.values()), 3))
            except Exception as e:
                logger.warning("book_vol_target_failed", error=str(e)[:120])

        # BOOK-EQUITY DRAWDOWN DE-GEAR: when the paper book is in a peak-to-trough drawdown beyond
        # the threshold, scale exposure toward the floor (the validated drawdown_degear discipline,
        # driven by the book's OWN equity — the RiskManager's halt tracks the dead NN book instead).
        dd_now = float(getattr(self.portfolio, "current_drawdown", 0.0))   # <= 0
        dd = -dd_now
        if dd > self.dd_degear_threshold:
            dd_scale = max(self.dd_degear_floor,
                           1.0 - (dd - self.dd_degear_threshold) / max(1.0 - self.dd_degear_threshold, 1e-9))
            targets = {s: w * dd_scale for s, w in targets.items()}
            logger.warning("book_drawdown_degear", drawdown=round(dd, 4), scale=round(dd_scale, 3))

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
        # NOTE: publishing is deliberately NOT done here. `rebalance_once` must stay free of
        # Redis I/O so unit tests can drive it with fakes — a `_publish_portfolio()` call here
        # once wrote a TEST book (universe A/B/C, equity wrecked) straight over the OWNER'S live
        # `strategy:paperbook:state`, because get_redis() reaches the real server when it's up.
        # The run() loop publishes after this returns.
        plan = {"targets": targets, "orders": orders, "skipped": skipped,
                "news": news_notes, "overlay": overlay_notes, "blocked": None}
        if self.journal is not None:
            try:
                self.journal.record_rebalance(plan, equity=equity, prices=prices,
                                              paper=bool(self.paper))
            except Exception as e:
                logger.warning("journal_rebalance_failed", error=str(e)[:100])
        return plan

    def _after_mark(self, book_return: float) -> None:
        """Post-mark bookkeeping: journal the mark + per-symbol outcomes, and feed realized
        per-symbol returns to the overlay's online conviction/conformal loop. Aggregation-level
        LEARNING happens only in scripts/postmortem.py — never here (per-loss mutation is the
        online-AWR failure mode this project already measured)."""
        book = self.portfolio
        # Skip stale marks (no held symbol actually re-priced): feeding fabricated 0.0 per-symbol
        # returns into the journal / overlay conviction loop teaches it on flat noise.
        if not getattr(book, "last_mark_repriced", True):
            return
        contribs: Dict[str, float] = dict(getattr(book, "last_contributions", {}) or {})
        if self.journal is not None:
            try:
                self.journal.record_mark(
                    equity=float(getattr(book, "equity", 0.0)),
                    book_return=float(book_return),
                    drawdown=float(getattr(book, "current_drawdown", 0.0)),
                    contributions=contribs,
                    weights=dict(getattr(book, "positions", {}) or {}),
                )
            except Exception as e:
                logger.warning("journal_mark_failed", error=str(e)[:100])
        if self.overlay_gate is not None and hasattr(self.overlay_gate, "record_outcome"):
            for sym, c in contribs.items():
                w = float((getattr(book, "positions", {}) or {}).get(sym, 0.0))
                if abs(w) > 1e-9:
                    try:   # per-symbol return (sign is what the conviction EWMA consumes)
                        self.overlay_gate.record_outcome(sym, c / w)
                    except Exception:
                        pass

    @staticmethod
    def _book_realized_vol(targets: Dict[str, float], all_bars: Dict[str, pd.DataFrame],
                           window: int = 63) -> Optional[float]:
        """Annualized realized vol of the book at the CURRENT target weights, estimated from the
        trailing daily returns of the held assets. Sub-daily bars (4h crypto) are collapsed to a
        daily last-close so equities and crypto share one calendar; the book return each day is
        Σ wᵢ·rᵢ. Returns None if too little overlapping history (caller then skips scaling)."""
        import numpy as np
        series: Dict[str, "pd.Series"] = {}
        for s, w in targets.items():
            if abs(w) < 1e-9:
                continue
            df = all_bars.get(s)
            if df is None or "close" not in getattr(df, "columns", []) or len(df) < 10:
                continue
            try:
                if "timestamp" in df.columns:
                    idx = pd.to_datetime(df["timestamp"])
                    close = pd.Series(df["close"].to_numpy(float), index=idx)
                    close = close.resample("1D").last().dropna()          # collapse 4h→daily
                else:
                    close = pd.Series(df["close"].to_numpy(float))
                ret = close.pct_change().dropna()
                if len(ret) >= 10:
                    series[s] = ret
            except Exception:
                continue
        if not series:
            return None
        panel = pd.DataFrame(series).tail(window * 2)
        cols = [c for c in panel.columns]
        book = None
        for c in cols:
            contrib = float(targets[c]) * panel[c].fillna(0.0)
            book = contrib if book is None else book.add(contrib, fill_value=0.0)
        if book is None:
            return None
        book = book.dropna().tail(window)
        if len(book) < 10:
            return None
        vol = float(np.std(book.to_numpy(), ddof=1) * np.sqrt(252.0))
        return vol if vol > 1e-9 else None

    def _conviction_scale(self, targets: Dict[str, float],
                          all_bars: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """Per-asset conviction multiplier in [1/cap, cap] from SIGNED trend strength — how far the
        price sits beyond its trend EMA, in units of return vol, in the DIRECTION of the position.
        A long in a strong uptrend (or a short in a strong downtrend) → multiplier > 1 (bigger bet);
        a marginal/countertrend position → multiplier < 1. tanh-saturated so extremes can't explode.
        Fractional-Kelly in spirit: bet size ∝ conviction (edge proxy); vol-target handles variance."""
        import numpy as np
        gain = float(self.conviction_gain)
        cap = max(1.0, float(self.conviction_cap))
        out: Dict[str, float] = {}
        for s, w in targets.items():
            if abs(w) < 1e-9:
                continue
            df = all_bars.get(s)
            if df is None or "close" not in getattr(df, "columns", []) or len(df) < 60:
                continue
            try:
                close = pd.Series(df["close"].to_numpy(float))
                ema = close.ewm(span=50, adjust=False).mean()
                ret_std = close.pct_change().rolling(50).std().iloc[-1]
                px = float(close.iloc[-1]); e = float(ema.iloc[-1])
                if not (px > 0 and e > 0 and np.isfinite(ret_std) and ret_std > 1e-9):
                    continue
                # distance from trend in vol units, signed by the position direction
                z = ((px - e) / e) / float(ret_std)
                signed_z = float(np.sign(w)) * z
                mult = 1.0 + gain * float(np.tanh(signed_z / 2.0))
                out[s] = float(min(cap, max(1.0 / cap, mult)))
            except Exception:
                continue
        return out

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

    @staticmethod
    def _binance_all_prices(symbols: Optional[List[str]] = None) -> Dict[str, float]:
        """Binance spot last prices (symbol -> price) in ONE call. US-reachable mirror first, main
        host as fallback. Perp marks use spot (basis is negligible for a paper mark).

        Pass ``symbols`` to fetch only what we hold — the unfiltered endpoint returns ~700 rows
        (~250KB) and this runs every mark_seconds. Falls back to the full list if the filtered
        request is rejected."""
        import json as _json
        import requests
        params = {}
        if symbols:
            # Binance wants a JSON array, e.g. symbols=["BTCUSDT","ETHUSDT"]; requests URL-encodes it.
            params["symbols"] = _json.dumps([s.upper() for s in symbols], separators=(",", ":"))
        for attempt_params in ([params, {}] if params else [{}]):
            for host in ("https://data-api.binance.vision", "https://api.binance.com"):
                try:
                    r = requests.get(f"{host}/api/v3/ticker/price", params=attempt_params,
                                     headers={"User-Agent": "Mozilla/5.0"}, timeout=8)
                    if r.status_code == 200:
                        rows = r.json() or []
                        if isinstance(rows, dict):          # single-symbol shape
                            rows = [rows]
                        return {str(d.get("symbol", "")).upper(): float(d.get("price", 0.0) or 0.0)
                                for d in rows if d.get("symbol")}
                except Exception:
                    continue
        return {}

    @staticmethod
    def _alpaca_last_prices(symbols: List[str]) -> Dict[str, float]:
        """Latest trade price per US-stock symbol via Alpaca snapshots (one call). Empty on any
        failure / missing creds. During closed markets Alpaca returns the last trade (non-zero,
        unchanged) so those names simply don't move — no spurious jumps."""
        if not symbols:
            return {}
        try:
            from backend.core.config import settings
            key = (getattr(settings, "ALPACA_API_KEY", "") or "").strip()
            secret = (getattr(settings, "ALPACA_SECRET_KEY", "") or
                      getattr(settings, "ALPACA_SECRET", "") or "").strip()
            if not (key and secret and "your_" not in key.lower()):
                return {}
            import requests
            r = requests.get("https://data.alpaca.markets/v2/stocks/snapshots",
                             params={"symbols": ",".join(symbols), "feed": "iex"},
                             headers={"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret},
                             timeout=8)
            if r.status_code != 200:
                return {}
            out: Dict[str, float] = {}
            for sym, snap in (r.json() or {}).items():
                if not isinstance(snap, dict):
                    continue
                px = 0.0
                for node_key, field_key in (("latestTrade", "p"), ("latestQuote", "ap"),
                                            ("dailyBar", "c")):
                    node = snap.get(node_key) or {}
                    try:
                        px = float(node.get(field_key, 0.0) or 0.0)
                    except (TypeError, ValueError):
                        px = 0.0
                    if px > 0:
                        break
                if px > 0:
                    out[str(sym).upper()] = px
            return out
        except Exception:
            return {}

    async def _live_prices(self, symbols: List[str]) -> Dict[str, float]:
        """Best-effort LIVE last prices keyed to match the held-position symbols. Crypto (…USDT or
        yfinance …-USD) → Binance spot; US stocks/ETFs → Alpaca. Symbols we can't price are omitted,
        so mark_to_market leaves them flat this step (correct for a stock overnight)."""
        out: Dict[str, float] = {}
        syms = [s for s in symbols if s]
        if not syms:
            return out
        # ONE Binance symbol can back SEVERAL position keys — the two-sleeve book holds the perp
        # "BTCUSDT" (TS sleeve) AND the yfinance "BTC-USD" (managed sleeve), both priced off
        # Binance BTCUSDT. A 1:1 map silently dropped one of them, freezing that leg's mark.
        binance_map: Dict[str, List[str]] = {}   # BINANCE_SYMBOL -> [our position keys]
        stock_syms: List[str] = []
        for s in syms:
            su = str(s).upper()
            if su.endswith("USDT"):
                binance_map.setdefault(su, []).append(s)
            elif su.endswith("-USD"):
                binance_map.setdefault(su.replace("-USD", "USDT"), []).append(s)
            elif su.startswith("^"):
                continue                    # ^VIX etc — a signal input, never a holding
            else:
                stock_syms.append(su)
        if binance_map:
            try:
                allpx = await asyncio.to_thread(self._binance_all_prices, list(binance_map))
                for b, keys in binance_map.items():
                    px = allpx.get(b, 0.0)
                    if px > 0:
                        for key in keys:
                            out[key] = px
            except Exception as e:
                logger.debug("live_crypto_prices_failed", error=str(e)[:100])
        if stock_syms:
            try:
                sp = await asyncio.to_thread(self._alpaca_last_prices, stock_syms)
                out.update(sp)
            except Exception as e:
                logger.debug("live_stock_prices_failed", error=str(e)[:100])
        return out

    async def mark_and_publish_once(self) -> None:
        """Fast intra-rebalance revaluation: mark HELD positions at LIVE prices and republish the
        dashboard view + status — WITHOUT recomputing targets, routing, journaling, or overlay
        learning (those stay on the daily rebalance cadence). This is what makes the dashboard net
        worth move in real time between rebalances."""
        book = self.portfolio
        positions = dict(getattr(book, "positions", {}) or {})
        held = [s for s, w in positions.items() if abs(float(w)) > 1e-9]
        if not held or not hasattr(book, "mark_to_market"):
            return
        prices = await self._live_prices(held)
        if not prices:
            return
        now = __import__("time").time()
        record_hist = (now - self._last_hist_ts) >= self.history_interval_seconds
        try:
            book.mark_to_market(prices, now, record_history=record_hist)
        except TypeError:                     # a custom book without the record_history kwarg
            book.mark_to_market(prices, now)
            record_hist = True
        if record_hist and getattr(book, "last_mark_repriced", True):
            self._last_hist_ts = now
        # Dashboard view every mark (that's the point); heavy state only on the history cadence.
        # NOTE: status/heartbeat is published by run() on EVERY loop iteration, not here — this
        # method returns early when prices can't be fetched, and the engine must not look dead to
        # the dashboard just because a quote fetch failed.
        await self._publish_portfolio(light=True, persist_state=record_hist)

    async def _publish_portfolio(self, *, light: bool = False, persist_state: bool = True) -> None:
        """Publish the paper book's dashboard view (``strategy:portfolio``), the FULL restorable
        book state (``strategy:paperbook:state`` — so a restart resumes instead of resetting), and
        the entity graph's live state (``strategy:graph``) to Redis. Best-effort — a Redis hiccup
        never breaks a rebalance, and a live (non-paper) book simply has no view/state.

        ``light=True`` (the fast mark loop) publishes ONLY the dashboard view: the entity graph is
        refreshed per-rebalance and the scanner is TTL-cached, so re-serialising them every 20s is
        pure cost. ``persist_state=False`` also skips the full restorable state (which embeds the
        whole net-worth curve); the fast loop persists it on the slower history cadence instead.
        That is lossless — ``last_prices`` is snapshotted with the equity, so the next mark after a
        crash recovers the whole move since the last persisted point."""
        try:
            import json as _json
            from backend.memory.redis_client import get_redis
            r = await get_redis()
            if hasattr(self.portfolio, "portfolio_view"):
                view = self.portfolio.portfolio_view()
                view["paper"] = bool(self.paper)
                view["updated_at"] = __import__("time").time()
                await r.set("strategy:portfolio", _json.dumps(view))
            if persist_state and self.paper and hasattr(self.portfolio, "to_state"):
                await r.set("strategy:paperbook:state", _json.dumps(self.portfolio.to_state()))
            if light:
                return
            if self.entity_graph is not None and hasattr(self.entity_graph, "view"):
                g = self.entity_graph.view()
                g["updated_at"] = __import__("time").time()
                await r.set("strategy:graph", _json.dumps(g))
            if self.scanner is not None and hasattr(self.scanner, "publish"):
                try:
                    await self.scanner.publish(r)          # strategy:anomalies for the copilot + UI
                except Exception:
                    pass
        except Exception as e:
            logger.debug("strategy_portfolio_publish_failed", error=str(e)[:100])

    async def _restore_paper_book(self) -> None:
        """On startup, restore the paper book from Redis so a backend restart resumes the running
        equity/positions instead of resetting to the seed (and re-paying the establishment cost).
        Honours a pending reset request (paper:reset_requested) by starting fresh."""
        if not (self.paper and hasattr(self.portfolio, "load_state")):
            return
        try:
            import json as _json
            from backend.memory.redis_client import get_redis
            r = await get_redis()
            reset = await r.get("paper:reset_requested")
            if reset:
                logger.warning("paper_book_reset_pending_skip_restore")
                await r.delete("strategy:paperbook:state")
                # CONSUME the flag. Nothing else clears it when NN_AGENT_ENABLED=false (only
                # nn_agent did), so leaving it set would wipe the book on EVERY restart. That is
                # why _reset_paper_state() had to fall back to a fragile 60s TTL.
                await r.delete("paper:reset_requested")
                return
            raw = await r.get("strategy:paperbook:state")
            if raw:
                state = _json.loads(raw if isinstance(raw, str) else raw.decode())
                # SANITY GUARD: never adopt a book holding symbols this agent doesn't trade. A
                # foreign position means the state was written by something else (a unit test
                # sharing this Redis, a different config) — restoring it would silently import
                # that book's equity and marks. Start fresh instead of inheriting nonsense.
                known = set(self.universe) | set(self.ts_universe or [])
                foreign = {s for s, w in (state.get("positions") or {}).items() if s not in known}
                if foreign:
                    logger.error("paper_book_restore_rejected_foreign_symbols",
                                 foreign=sorted(foreign)[:8],
                                 state_equity=round(float(state.get("equity", 0.0)), 2),
                                 note="starting a FRESH book; the persisted state was not written "
                                      "by this agent's universe")
                    await r.delete("strategy:paperbook:state")
                    return
                self.portfolio.load_state(state)
                logger.warning("paper_book_restored", equity=round(float(self.portfolio.equity), 2),
                               positions=len(getattr(self.portfolio, "positions", {})),
                               history=len(getattr(self.portfolio, "net_worth_history", [])))
        except Exception as e:
            logger.warning("paper_book_restore_failed", error=str(e)[:120])

    async def _publish_status(self, *, last_rebalance: float, gross: float, blocked) -> None:
        """Publish a live heartbeat + status (``strategy:status`` + heartbeat ping) so the dashboard
        engine banner reflects THIS book, not the (disabled) NN agent."""
        try:
            import json as _json, time as _t
            from backend.memory.redis_client import get_redis, HeartbeatClient
            r = await get_redis()
            # TTL must comfortably outlive the ping interval (one ping per mark). At the default
            # 10s TTL vs a 20s mark the heartbeat was expired half the time and /api/agent/status
            # fell back to the "Starting up agent core..." stub.
            await HeartbeatClient(r).ping("strategy_agent",
                                          ttl_seconds=int(max(30.0, self.mark_seconds * 3.0)))
            status = {
                "mode": "PAPER" if self.paper else "LIVE",
                # REAL process start. The API used to synthesize `last_rebalance - 1e6` to force the
                # banner out of its warm-up window, which made the dashboard report hundreds of
                # hours of uptime. The banner now skips warm-up for this engine explicitly.
                "started_at": float(self._started_at),
                "last_rebalance": last_rebalance,
                "next_rebalance": last_rebalance + float(self.rebalance_seconds),
                "rebalance_seconds": float(self.rebalance_seconds),
                "gross": round(float(gross), 4),
                "blocked": blocked,
                "sleeves": 2 if self.ts_universe else 1,
                "equity": round(float(getattr(self.portfolio, "equity", 0.0)), 2),
                "drawdown": round(float(getattr(self.portfolio, "current_drawdown", 0.0)), 4),
                "book_vol_target": float(self.book_vol_target),
                "sleeve_vol_target": float(self.target_vol),
                "max_leverage": float(self.book_max_leverage),
                "updated_at": _t.time(),
            }
            await r.set("strategy:status", _json.dumps(status))
        except Exception as e:
            logger.debug("strategy_status_publish_failed", error=str(e)[:100])

    async def _route(self, sym: str, target_w: float, delta: float, equity: float) -> None:
        if self.paper:
            await self.portfolio.apply(sym, target_w, delta)
        else:
            await self.live_router(sym, target_w, delta)   # operator-supplied; never auto-wired

    async def run(self) -> None:
        mode = "PAPER" if self.paper else "LIVE(operator-router)"
        self._started_at = __import__("time").time()   # uptime counts from HERE, not a fake far-past
        await self._restore_paper_book()      # resume across restarts (no more equity wipe)
        logger.info("strategy_agent_started", universe=self.universe, mode=mode,
                    rebalance_seconds=self.rebalance_seconds, mark_seconds=self.mark_seconds)
        # Two cadences from ONE loop that wakes every mark_seconds: a full rebalance (target-weight
        # change + routing) at most every rebalance_seconds, and a cheap live re-mark + republish on
        # every other wake so the dashboard net worth moves in real time. _last_rebalance_ts starts
        # at 0, so the first iteration runs a full rebalance immediately (as before).
        while not self._stop.is_set():
            now = __import__("time").time()
            blocked = None
            try:
                if now - self._last_rebalance_ts >= self.rebalance_seconds:
                    # Mark the rebalance on the SAME live quotes the fast loop uses (no saw-tooth).
                    held = [s for s, w in (getattr(self.portfolio, "positions", {}) or {}).items()
                            if abs(float(w)) > 1e-9]
                    live = await self._live_prices(held) if held else {}
                    plan = await self.rebalance_once(live_prices=live)
                    self._last_rebalance_ts = now
                    self._last_hist_ts = now
                    blocked = plan.get("blocked")
                    # Full publish (view + restorable state + graph + anomaly flags). Lives HERE,
                    # not in rebalance_once, so that method never touches Redis (see note there).
                    await self._publish_portfolio()
                else:
                    await self.mark_and_publish_once()
            except Exception as e:
                logger.error("strategy_loop_error", error=str(e)[:200])
            # Heartbeat + status on EVERY iteration, whatever happened above. Publishing this only
            # on a successful mark meant one failed quote fetch (or a stock-only book overnight)
            # let the heartbeat lapse, and /api/agent/status fell back to "Starting up agent
            # core..." — i.e. a live engine reporting itself dead.
            try:
                gross = float(sum(abs(float(w))
                                  for w in (getattr(self.portfolio, "positions", {}) or {}).values()))
                await self._publish_status(last_rebalance=self._last_rebalance_ts,
                                           gross=gross, blocked=blocked)
            except Exception as e:
                logger.debug("strategy_status_publish_skipped", error=str(e)[:80])
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self.mark_seconds)
            except asyncio.TimeoutError:
                pass

    def stop(self) -> None:
        self._stop.set()
