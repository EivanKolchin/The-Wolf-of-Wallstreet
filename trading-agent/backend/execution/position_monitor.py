"""Background position monitor — applies the model-emitted stop-loss / take-profit /
trailing stop to every open trade and closes via the execution engine when hit.

Cycle 19.3: also runs a momentum-reversal exit (talib MACD + volume spike) so
positions close on sharp trend changes — not just when a hard SL/TP/trail is hit.

Replaces the dead ``position_manager.py`` (which was never started and had a `break`
that killed its loop after one pass). Runs as an asyncio task inside the NN-agent
process and shares the agent's ``open_trades`` dict by reference, so it sees the
same Trade rows the agent opens.
"""
from __future__ import annotations

import asyncio
from typing import Any, Dict, Optional

import structlog

try:
    import talib                                       # type: ignore
    HAS_TALIB = True
except Exception:
    HAS_TALIB = False

logger = structlog.get_logger(__name__)


class PositionMonitor:
    def __init__(self, open_trades: Dict[str, Any], execution_engine: Any,
                 poll_interval: float = 3.0,
                 on_breach_recalc=None,
                 market_feed: Optional[Any] = None,
                 reversal_macd_drop: float = 0.5,
                 reversal_vol_multiple: float = 2.0):
        """`on_breach_recalc(symbol, trade, reason) -> dict | None` (async or sync) lets the
        agent instantly recalculate SL/TP/trail on a boundary breach (Phase 9). Returning a
        dict with any of {stop_loss, take_profit, trailing_stop} updates the trade IN-PLACE
        and skips closing; returning None falls through to the normal close.

        ``market_feed`` is optional — when supplied (and talib is installed) we
        run a momentum-reversal exit on each tick using the most recent rolling
        OHLCV. Without it, the monitor falls back to SL/TP/trailing only.
        """
        self.open_trades = open_trades
        self.execution_engine = execution_engine
        self.poll_interval = poll_interval
        self.on_breach_recalc = on_breach_recalc
        self.market_feed = market_feed
        self.reversal_macd_drop = float(reversal_macd_drop)
        self.reversal_vol_multiple = float(reversal_vol_multiple)
        self._running = False

    def stop(self) -> None:
        self._running = False

    async def run(self) -> None:
        if not hasattr(self.execution_engine, "get_price") or not hasattr(self.execution_engine, "close_position"):
            logger.warning("position_monitor_disabled_engine_unsupported")
            return
        self._running = True
        logger.info("position_monitor_started", poll_interval=self.poll_interval)
        while self._running:
            try:
                await self._tick()
            except Exception as e:
                logger.error("position_monitor_tick_error", error=str(e))
            await asyncio.sleep(self.poll_interval)

    async def _check_reversal(self, symbol: str) -> bool:
        """Cycle 19.3: talib-based momentum reversal detector. Computes MACD
        histogram + volume MA on the rolling OHLCV the market feed already
        caches; returns True iff (prev_hist - curr_hist) > drop_threshold AND
        current volume > vol_multiple × 20-bar volume MA. Cheap (~2 ms)."""
        if self.market_feed is None or not HAS_TALIB:
            return False
        try:
            df = await self.market_feed.get_dataframe(symbol)
        except Exception:
            return False
        if df is None or len(df) < 35:
            return False
        try:
            import numpy as np
            close = df["close"].astype(float).values
            volume = df["volume"].astype(float).values
            _macd, _sig, macd_hist = talib.MACD(close)
            vol_ma = talib.SMA(volume, timeperiod=20)
            # Ignore the warmup NaNs.
            if not (np.isfinite(macd_hist[-2]) and np.isfinite(macd_hist[-1])
                    and np.isfinite(vol_ma[-1]) and vol_ma[-1] > 0):
                return False
            macd_drop = float(macd_hist[-2] - macd_hist[-1])
            vol_spike = float(volume[-1]) > self.reversal_vol_multiple * float(vol_ma[-1])
            return macd_drop > self.reversal_macd_drop and vol_spike
        except Exception as e:
            logger.debug("reversal_check_error", symbol=symbol, error=str(e)[:120])
            return False

    async def _tick(self) -> None:
        for symbol, trade in list(self.open_trades.items()):
            try:
                price = await self.execution_engine.get_price(symbol)
            except Exception:
                price = None
            if price is None:
                continue

            # Momentum-reversal exit FIRST — closes ahead of an SL hit so
            # losing positions exit at a better price during sharp turns.
            if await self._check_reversal(symbol):
                logger.warning("reversal_exit_triggered", symbol=symbol)
                try:
                    await self.execution_engine.close_position(symbol, reason="momentum_reversal")
                except Exception as e:
                    logger.error("reversal_close_failed", symbol=symbol, error=str(e))
                continue

            reason = self._exit_reason(trade, float(price))
            if reason:
                # Phase 9: optional instant recalc — adjust SL/TP/trail in-place instead of closing
                if self.on_breach_recalc is not None:
                    try:
                        result = self.on_breach_recalc(symbol, trade, reason)
                        if hasattr(result, "__await__"):
                            result = await result  # support async callbacks
                    except Exception as e:
                        logger.warning("breach_recalc_failed", symbol=symbol, error=str(e))
                        result = None
                    if result:
                        if "stop_loss" in result:
                            trade.stop_loss = float(result["stop_loss"])
                        if "take_profit" in result:
                            trade.take_profit = float(result["take_profit"])
                        if "trailing_stop" in result:
                            trade.trailing_stop = float(result["trailing_stop"])
                        logger.info("boundary_recalculated", symbol=symbol, reason=reason, levels=result)
                        continue

                logger.info("exit_triggered", symbol=symbol, reason=reason, price=float(price))
                try:
                    await self.execution_engine.close_position(symbol, reason=reason)
                except Exception as e:
                    logger.error("monitor_close_failed", symbol=symbol, error=str(e))

    def _exit_reason(self, trade: Any, price: float):
        direction = trade.direction.value if hasattr(getattr(trade, "direction", None), "value") else str(getattr(trade, "direction", "long"))
        entry = float(getattr(trade, "entry_price", 0.0) or 0.0)
        sl = float(getattr(trade, "stop_loss", 0.0) or 0.0)
        tp = float(getattr(trade, "take_profit", 0.0) or 0.0)
        trail_frac = float(getattr(trade, "trailing_stop", 0.0) or 0.0)

        if direction == "long":
            hp = getattr(trade, "highest_price_seen", None) or entry
            if price > hp:
                trade.highest_price_seen = price
                hp = price
            trail_limit = hp * (1 - trail_frac) if trail_frac > 0 else 0.0
            if sl > 0 and price <= sl:
                return "stop_loss"
            if tp > 0 and price >= tp:
                return "take_profit"
            if trail_limit > 0 and hp > entry and price <= trail_limit:
                return "trailing_stop"
        else:  # short
            lp = getattr(trade, "lowest_price_seen", None) or entry
            if price < lp:
                trade.lowest_price_seen = price
                lp = price
            trail_limit = lp * (1 + trail_frac) if trail_frac > 0 else 0.0
            if sl > 0 and price >= sl:
                return "stop_loss"
            if tp > 0 and price <= tp:
                return "take_profit"
            if trail_limit > 0 and lp < entry and price >= trail_limit:
                return "trailing_stop"
        return None
