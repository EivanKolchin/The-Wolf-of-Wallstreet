import asyncio
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class PositionManager:
    def __init__(
        self,
        open_trades: Dict[str, Any],
        execution_engine: Any,
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
        self.execution_engine = execution_engine
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
                for symbol, trade in list(self.open_trades.items()):
                    # Assume execution_engine has a method to get the current price for a symbol
                    current_price = await self.execution_engine.get_current_price(symbol)

                    if current_price is None:
                        continue

                    # Update highest price seen natively dynamically
                    if not hasattr(trade, 'highest_price_seen') or trade.highest_price_seen is None:
                        trade.highest_price_seen = trade.entry_price

                    if current_price > trade.highest_price_seen:
                        trade.highest_price_seen = current_price

                    # Calculate the dynamic trailing stop limit
                    trailing_stop_limit = trade.highest_price_seen * (1.0 - self.trailing_pct)
                    trade.trailing_stop_limit = trailing_stop_limit  # Storing this for UI visibility

                    # Trigger closure if price dumps below our dynamic trailing stop limit
                    if current_price <= trailing_stop_limit:
                        logger.info(f"[{symbol}] Trailing stop triggered. Current: {current_price}, Stop: {trailing_stop_limit}")
                        await self.execution_engine.close_position(trade)
                        # Optionally remove the trade if your engine doesn't automatically drop it from open_trades
                        # self.open_trades.pop(symbol, None)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in PositionManager monitoring loop: {e}")

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
