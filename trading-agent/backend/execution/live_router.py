"""Live order router for the StrategyAgent's two-sleeve target-weight book.

Implements the ``live_router(symbol, target_weight, delta)`` callable the StrategyAgent invokes when
``paper=False``. It converts a weight DELTA into a USD notional (delta × book equity), classifies the
symbol's venue, and routes the order:

    crypto perps (e.g. BTCUSDT)  → Binance USD-M futures  (long+short, leverage — the TS sleeve)
    everything else (ETFs, *-USD spot) → Alpaca           (the managed-beta sleeve)

Safety: this is only ever wired in when ``settings.any_live_enabled()`` is true (PAPER_TRADING=false
AND real credentials). Per venue, each broker independently refuses a REAL order unless ITS own keys
are valid — so Alpaca can be live while Binance stays on testnet, and vice-versa. A per-order notional
cap bounds the damage of any single mistake. ``reduce_only`` is set whenever the order shrinks the
position (moving toward zero), so de-risking never accidentally opens new exposure.
"""
from __future__ import annotations

from typing import Optional, Set

try:
    import structlog
    logger = structlog.get_logger("live_router")
except Exception:  # pragma: no cover
    import logging
    logger = logging.getLogger("live_router")


def is_perp(symbol: str, perp_symbols: Optional[Set[str]] = None) -> bool:
    """A Binance-futures perp: explicitly listed, or a USDT-margined symbol (…USDT) that is NOT a
    yfinance-style spot ticker (…-USD)."""
    s = str(symbol).upper()
    if perp_symbols and s in {p.upper() for p in perp_symbols}:
        return True
    return s.endswith("USDT")


class LiveRouter:
    def __init__(self, *, equity: float, alpaca=None, binance=None,
                 perp_symbols: Optional[Set[str]] = None, max_order_notional: float = 5_000.0,
                 min_order_notional: float = 1.0, perp_order_style: str = "passive",
                 limit_timeout_s: float = 30.0, slippage_ledger=None,
                 slice_threshold: float = 0.0, slice_children: int = 6,
                 slice_window_s: float = 900.0):
        self.equity = float(equity)
        self.alpaca = alpaca
        self.binance = binance
        self.perp_symbols = set(perp_symbols or [])
        self.max_order_notional = float(max_order_notional)
        self.min_order_notional = float(min_order_notional)
        # TWAP child-order slicing: an order whose |notional| exceeds ``slice_threshold`` is
        # chopped into ``slice_children`` equal children spread evenly over ``slice_window_s``.
        # Between the passive limit style and full RL execution, this is the deterministic rung
        # that caps market impact once position sizes grow with leverage — no learning, no
        # reliability risk. Perps only (Alpaca sizes here are small relative to equity depth).
        # OPT-IN: 0 disables (the default for direct construction — slicing awaits real
        # inter-child sleeps, which callers/tests must knowingly sign up for); production gets
        # the threshold from settings via make_live_router.
        self.slice_threshold = float(slice_threshold)
        self.slice_children = max(1, int(slice_children))
        self.slice_window_s = float(slice_window_s)
        # Execution BEFORE leverage: perp orders default to passive post-only limits with a
        # market sweep on timeout — leverage multiplies turnover, and paying the spread on
        # every rebalance at 2× turnover silently eats the CAGR the leverage was buying.
        self.perp_order_style = str(perp_order_style)
        self.limit_timeout_s = float(limit_timeout_s)
        if slippage_ledger is None:
            try:
                from backend.execution.slippage_ledger import SlippageLedger
                slippage_ledger = SlippageLedger()
            except Exception:   # pragma: no cover — ledger is telemetry, never blocks routing
                slippage_ledger = None
        self.ledger = slippage_ledger

    def _notional(self, delta: float) -> float:
        n = delta * self.equity
        # clamp magnitude to the per-order cap (keeps sign)
        if abs(n) > self.max_order_notional:
            n = self.max_order_notional if n > 0 else -self.max_order_notional
        return n

    async def __call__(self, symbol: str, target_weight: float, delta: float) -> dict:
        notional = self._notional(delta)
        if abs(notional) < self.min_order_notional:
            return {"status": "skipped", "reason": "below min notional", "symbol": symbol}
        # reduce_only when the order shrinks the position (current = target - delta)
        current = target_weight - delta
        reduce_only = abs(target_weight) < abs(current)
        venue = "binance" if is_perp(symbol, self.perp_symbols) else "alpaca"
        broker = self.binance if venue == "binance" else self.alpaca
        if broker is None:
            logger.warning("live_router_no_broker", symbol=symbol, venue=venue)
            return {"status": "error", "reason": f"no {venue} broker", "symbol": symbol}

        # TWAP slice large perp orders; small orders / equities route in one shot.
        if (venue == "binance" and self.slice_threshold > 0
                and abs(notional) > self.slice_threshold and self.slice_children > 1):
            children = await self._twap(broker, symbol, notional, reduce_only)
            self._record_slippage(symbol, venue, notional, children[-1] if children else None)
            logger.info("live_router_twap", symbol=symbol, notional=round(notional, 2),
                        children=len(children))
            return {"venue": venue, "notional": notional, "reduce_only": reduce_only,
                    "sliced": True, "children": children}

        res = await self._route_one(broker, venue, symbol, notional, reduce_only)
        logger.info("live_router_routed", symbol=symbol, venue=venue, notional=round(notional, 2),
                    reduce_only=reduce_only, status=(res or {}).get("status"))
        self._record_slippage(symbol, venue, notional, res)
        return {"venue": venue, "notional": notional, "reduce_only": reduce_only, "result": res}

    async def _route_one(self, broker, venue: str, symbol: str, notional: float,
                         reduce_only: bool) -> dict:
        if venue == "binance":
            try:
                return await broker.submit_notional(symbol, notional, reduce_only=reduce_only,
                                                    style=self.perp_order_style,
                                                    limit_timeout_s=self.limit_timeout_s)
            except TypeError:   # duck-typed broker without style support → plain market
                return await broker.submit_notional(symbol, notional, reduce_only=reduce_only)
        return await broker.submit_notional(symbol, notional, reduce_only=reduce_only)

    async def _twap(self, broker, symbol: str, notional: float, reduce_only: bool) -> list:
        """Split ``notional`` into equal children spaced over the window; each child uses the
        configured passive style, so the slicer and the spread-earning execution compose."""
        import asyncio
        k = self.slice_children
        child = notional / k
        gap = self.slice_window_s / k
        out = []
        for i in range(k):
            res = await self._route_one(broker, "binance", symbol, child, reduce_only)
            out.append(res)
            self._record_slippage(symbol, "binance", child, res)
            if i < k - 1:
                await asyncio.sleep(max(0.0, gap))
        return out

    def _record_slippage(self, symbol: str, venue: str, notional: float, res) -> None:
        """Telemetry only — never raises into the routing path."""
        if self.ledger is None or not isinstance(res, dict):
            return
        dp, fp = res.get("decision_price"), res.get("fill_price")
        if not dp:
            return
        try:
            self.ledger.record(symbol=symbol, venue=venue,
                               side=("BUY" if notional > 0 else "SELL"),
                               notional=abs(notional), decision_price=dp, fill_price=fp,
                               style=str(res.get("style", "market")),
                               status=str(res.get("status", "")))
        except Exception as e:  # pragma: no cover
            logger.warning("slippage_record_failed", symbol=symbol, error=str(e)[:80])


def make_live_router(*, equity: float, perp_symbols: Optional[Set[str]] = None,
                     max_order_notional: float = 5_000.0) -> Optional[LiveRouter]:
    """Build a LiveRouter from settings, or None if no venue is live-enabled (caller keeps paper).
    Each broker is constructed only when ITS credentials make it live."""
    from backend.core.config import settings
    alpaca = binance = None
    if settings.alpaca_live_enabled():
        from backend.execution.alpaca_broker import AlpacaBroker
        alpaca = AlpacaBroker(paper=False)
    if settings.binance_futures_live_enabled():
        from backend.execution.binance_futures_broker import BinanceFuturesBroker
        binance = BinanceFuturesBroker(paper=False)
    if alpaca is None and binance is None:
        return None
    logger.info("live_router_built", alpaca=alpaca is not None, binance=binance is not None,
                equity=equity)
    return LiveRouter(equity=equity, alpaca=alpaca, binance=binance, perp_symbols=perp_symbols,
                      max_order_notional=max_order_notional,
                      perp_order_style=str(getattr(settings, "EXECUTION_PERP_STYLE", "passive")),
                      limit_timeout_s=float(getattr(settings, "EXECUTION_LIMIT_TIMEOUT_S", 30.0)),
                      slice_threshold=float(getattr(settings, "EXECUTION_SLICE_THRESHOLD_USD", 2000.0)),
                      slice_children=int(getattr(settings, "EXECUTION_SLICE_CHILDREN", 6)),
                      slice_window_s=float(getattr(settings, "EXECUTION_SLICE_WINDOW_S", 900.0)))
