import structlog
from datetime import datetime, timezone

from backend.agents.nn_agent import TradeDecision

logger = structlog.get_logger(__name__)

class RiskManager:
    HARD_LIMITS = {
        "max_portfolio_drawdown_pct": 15.0,
        "max_daily_loss_pct": 5.0,
        "max_single_position_pct": 20.0,
        "min_nn_confidence": 0.52,
        "max_trades_per_hour": 20,
        "min_position_usd": 12.0,
        "max_position_usd": 5000.0,
        "min_signal_classes_agreeing": 1,
    }

    def __init__(self, initial_portfolio_value: float = 10000.0):
        self.portfolio_value_usd = initial_portfolio_value
        self.peak_portfolio_value = initial_portfolio_value
        self.daily_pnl_usd = 0.0
        self.trades_this_hour = 0
        
        now = datetime.utcnow()
        self.last_hour_reset = now.replace(minute=0, second=0, microsecond=0)
        self.last_day_reset = now.replace(hour=0, minute=0, second=0, microsecond=0)
        
        self.is_halted = False

    def _reset_counters_if_needed(self):
        now = datetime.utcnow()
        
        # Hourly reset
        if now.replace(minute=0, second=0, microsecond=0) > self.last_hour_reset:
            self.trades_this_hour = 0
            self.last_hour_reset = now.replace(minute=0, second=0, microsecond=0)
            
        # Daily reset (Midnight UTC)
        if now.replace(hour=0, minute=0, second=0, microsecond=0) > self.last_day_reset:
            self.daily_pnl_usd = 0.0
            self.last_day_reset = now.replace(hour=0, minute=0, second=0, microsecond=0)

    def approve(self, decision: TradeDecision, portfolio_state: dict) -> tuple[bool, str]:
        if self.is_halted:
            return False, "HALTED: max drawdown exceeded — manual reset required"
            
        if decision.direction == "hold":
            return False, "HOLD decision — no trade"
            
        if decision.nn_confidence < self.HARD_LIMITS["min_nn_confidence"]:
            return False, f"Low confidence: {decision.nn_confidence:.3f}"
            
        self._reset_counters_if_needed()
        
        if self.trades_this_hour >= self.HARD_LIMITS["max_trades_per_hour"]:
            return False, "Trade rate limit"
            
        available_cash = portfolio_state.get("available_cash", 0.0)
        
        # Enforce max single position percent limit
        size_pct = min(decision.size_pct, self.HARD_LIMITS["max_single_position_pct"] / 100.0)
        position_usd = size_pct * available_cash
        
        if position_usd < self.HARD_LIMITS["min_position_usd"]:
            return False, "Below min notional"
            
        if position_usd > self.HARD_LIMITS["max_position_usd"]:
            return False, "Above max position"
            
        if self.peak_portfolio_value > 0:
            current_drawdown_pct = ((self.peak_portfolio_value - self.portfolio_value_usd) / self.peak_portfolio_value) * 100
            if current_drawdown_pct > self.HARD_LIMITS["max_portfolio_drawdown_pct"]:
                self.is_halted = True
                logger.error("max_drawdown_exceeded_halting_trading", drawdown_pct=current_drawdown_pct)
                return False, "MAX DRAWDOWN EXCEEDED — HALTED"
                
        # Calculate daily pnl limit dynamically
        start_of_day_value = self.portfolio_value_usd - self.daily_pnl_usd 
        if start_of_day_value > 0:
            daily_pnl_pct = (self.daily_pnl_usd / start_of_day_value) * 100
            if daily_pnl_pct < -self.HARD_LIMITS["max_daily_loss_pct"]:
                return False, "Daily loss limit"
                
        self.trades_this_hour += 1
        return True, "APPROVED"

    def update_portfolio(self, portfolio_state: dict) -> None:
        new_val = portfolio_state.get("total_value_usd", self.portfolio_value_usd)
        
        self.daily_pnl_usd += (new_val - self.portfolio_value_usd)
        self.portfolio_value_usd = new_val
        
        if self.portfolio_value_usd > self.peak_portfolio_value:
            self.peak_portfolio_value = self.portfolio_value_usd

    def reset_halt(self) -> None:
        self.is_halted = False
        self.peak_portfolio_value = self.portfolio_value_usd # resets drawdown baseline
        logger.info("trading_halt_manually_reset")

    def get_status(self) -> dict:
        current_drawdown_pct = 0.0
        if self.peak_portfolio_value > 0:
            current_drawdown_pct = ((self.peak_portfolio_value - self.portfolio_value_usd) / self.peak_portfolio_value) * 100
            
        start_of_day_value = self.portfolio_value_usd - self.daily_pnl_usd
        daily_pnl_pct = 0.0
        if start_of_day_value > 0:
            daily_pnl_pct = (self.daily_pnl_usd / start_of_day_value) * 100

        return {
            "portfolio_value_usd": self.portfolio_value_usd,
            "peak_portfolio_value": self.peak_portfolio_value,
            "daily_pnl_usd": self.daily_pnl_usd,
            "daily_pnl_pct": daily_pnl_pct,
            "trades_this_hour": self.trades_this_hour,
            "current_drawdown_pct": current_drawdown_pct,
            "is_halted": self.is_halted,
            "limits": self.HARD_LIMITS
        }