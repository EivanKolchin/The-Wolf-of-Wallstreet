import asyncio
import logging
from typing import Dict, Any
import json

from backend.memory.redis_client import get_redis

logger = logging.getLogger(__name__)

class PositionManager:
    def __init__(
        self,
        open_trades: Dict[str, Any],
        cex_execution_engine: Any,
        defi_execution_engine: Any = None,
        db_session_factory: Any = None,
        trailing_pct: float = 0.05,
        macd_drop_threshold: float = 0.5,
        volume_spike_multiple: float = 2.0,
        poll_interval: float = 3.0
    ):
        """
        Manages open positions, dynamically updating trailing stop losses
        and executing closure logic when stops or reversals trigger.
        """
        self.open_trades = open_trades
        self.cex_execution_engine = cex_execution_engine
        self.defi_execution_engine = defi_execution_engine
        self.db_session_factory = db_session_factory
        self.trailing_pct = trailing_pct
        self.macd_drop_threshold = macd_drop_threshold
        self.volume_spike_multiple = volume_spike_multiple
        self.poll_interval = poll_interval
        self._monitoring_task: asyncio.Task | None = None
        self._is_running = False

    def start_monitoring(self):
        """Start the continuous asynchronous monitoring loop."""
        if not self._is_running:
            self._is_running = True
            self._monitoring_task = asyncio.create_task(self._monitor_loop())
            logger.info("PositionManager monitoring loop started.")

    def stop_monitoring(self):
        """Stop the monitoring loop."""
        self._is_running = False
        if self._monitoring_task:
            self._monitoring_task.cancel()
            logger.info("PositionManager monitoring loop stopped.")

    async def _monitor_loop(self):
        """
        Continuously polls prices for open trades to adjust the highest price
        seen and trigger trailing stops if breached.
        """
        while self._is_running:
            try:
                redis_client = await get_redis()
                portfolio_live_state = {"unrealized_pnl": 0.0, "total_value_locked": 0.0, "positions": []}
                for symbol, trade in list(self.open_trades.items()):
                    # Use CEX engine for price feed (common)
                    current_price = await self.cex_execution_engine.get_current_price(symbol)

                    if current_price is None or current_price == 0:
                        continue

                    # Update highest price seen natively dynamically
                    if getattr(trade, 'highest_price_seen', None) is None:
                        trade.highest_price_seen = trade.entry_price

                    if current_price > trade.highest_price_seen:
                        trade.highest_price_seen = current_price
                        # Persist to DB
                        if self.db_session_factory:
                            from backend.memory.database import Trade
                            async with self.db_session_factory() as session:
                                t = await session.get(Trade, trade.id)
                                if t:
                                    t.highest_price_seen = current_price
                                    await session.commit()

                    # Calculate the dynamic trailing stop limit
                    trailing_stop_limit = trade.highest_price_seen * (1.0 - self.trailing_pct)
                    trade.trailing_stop_limit = trailing_stop_limit  # Storing this for UI visibility

                    # Add to live state
                    size_usd = getattr(trade, 'size_usd', 0.0)
                    direction_mult = 1.0 if (hasattr(trade.direction, 'value') and trade.direction.value == "long") or trade.direction == "long" else -1.0
                    unrealized = ((current_price - trade.entry_price) / trade.entry_price) * size_usd * direction_mult
                    portfolio_live_state["unrealized_pnl"] += unrealized
                    portfolio_live_state["total_value_locked"] += size_usd
                    
                    asset_size = size_usd / trade.entry_price if trade.entry_price > 0 else 0
                    
                    portfolio_live_state["positions"].append({
                        "symbol": symbol,
                        "unrealized": unrealized,
                        "size_usd": size_usd,
                        "asset_size": asset_size,
                        "entry_price": float(trade.entry_price),
                        "current_price": float(current_price),
                        "direction": str(trade.direction)
                    })

                    # Trigger closure if price dumps below our dynamic trailing stop limit
                    if current_price <= trailing_stop_limit:
                        logger.info(f"[{symbol}] Trailing stop triggered. Current: {current_price}, Stop: {trailing_stop_limit}")

                        # Route closure to the correct engine
                        is_defi = getattr(trade, "is_defi", False)
                        if is_defi and self.defi_execution_engine:
                            await self.defi_execution_engine.close_position(trade.asset)
                        else:
                            await self.cex_execution_engine.close_position(trade.asset)

                        # Remove from open_trades as it's being closed
                        self.open_trades.pop(symbol, None)

                if redis_client:
                    await redis_client.set("portfolio:live_state", json.dumps(portfolio_live_state))
                    
            except Exception as e:
                logger.error(f"Error in PositionManager monitoring loop: {e}", exc_info=True)

            await asyncio.sleep(self.poll_interval)

    def check_reversal(self, symbol: str, df: Any) -> bool:
        """
        Check for a reversal signal based on MACD histogram momentum and volume spikes.
        Requires a 2-row OHLCV DataFrame containing `macd_hist` and `volume_norm` columns.
        
        Args:
            symbol (str): The asset symbol being evaluated.
            df (pd.DataFrame): 2-row pandas DataFrame.
            
        Returns:
            bool: True if reversal conditions are met, False otherwise.
        """
        if df is None or len(df) < 2:
            return False

        try:
            # We want the most recent row (idx 1) and the previous row (idx 0)
            prev_row = df.iloc[0]
            curr_row = df.iloc[1]

            macd_drop = prev_row['macd_hist'] - curr_row['macd_hist']
            volume_spike = curr_row['volume_norm'] > self.volume_spike_multiple

            # Condition 1: MACD Histogram dropped more than the configured threshold
            # Condition 2: Normalized volume indicates a massive spike
            if macd_drop > self.macd_drop_threshold and volume_spike:
                logger.warning(f"[{symbol}] Reversal detected! MACD Drop: {macd_drop:.3f}, Volume Spike: {curr_row['volume_norm']:.2f}x")
                return True

        except KeyError as e:
            logger.error(f"DataFrame missing required column for reversal check: {e}")
        except Exception as e:
            logger.error(f"Error checking reversal for {symbol}: {e}")

        return False
