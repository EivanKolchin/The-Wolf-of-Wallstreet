from collections import deque
from datetime import datetime

import numpy as np
import structlog

from backend.agents.nn_agent import TradeDecision
from backend.core.config import settings

logger = structlog.get_logger(__name__)


class RiskManager:
    """Portfolio risk controls + Bayesian position sizing.

    Limits are loaded from settings (editable via the Advanced Options UI). Sizing
    uses fractional Kelly on the LOWER credible bound of the model's edge (from
    MC-dropout), so uncertainty automatically shrinks the bet. A Monte-Carlo
    CVaR projection blocks new trades when recent tail risk is too high.
    """

    def __init__(self, initial_portfolio_value: float = 10000.0):
        self.limits = self._load_limits()
        self.HARD_LIMITS = self.limits  # backward-compat alias (same dict reference)
        self.portfolio_value_usd = initial_portfolio_value
        self.peak_portfolio_value = initial_portfolio_value
        self.daily_pnl_usd = 0.0
        self.trades_this_hour = 0

        now = datetime.utcnow()
        self.last_hour_reset = now.replace(minute=0, second=0, microsecond=0)
        self.last_day_reset = now.replace(hour=0, minute=0, second=0, microsecond=0)

        self.is_halted = False
        self.recent_returns: deque = deque(maxlen=300)  # realized pnl_pct history (MC-CVaR)
        self.last_cvar_pct = 0.0

    @staticmethod
    def _load_limits() -> dict:
        s = settings
        return {
            "max_portfolio_drawdown_pct": float(getattr(s, "RISK_MAX_DRAWDOWN_PCT", 15.0)),
            "max_daily_loss_pct": float(getattr(s, "RISK_MAX_DAILY_LOSS_PCT", 5.0)),
            "max_single_position_pct": float(getattr(s, "RISK_MAX_POSITION_PCT", 20.0)),
            "min_nn_confidence": float(getattr(s, "RISK_MIN_CONFIDENCE", 0.0)),
            "max_trades_per_hour": int(getattr(s, "RISK_MAX_TRADES_PER_HOUR", 2000)),
            "min_position_usd": float(getattr(s, "RISK_MIN_POSITION_USD", 12.0)),
            "max_position_usd": float(getattr(s, "RISK_MAX_POSITION_USD", 5000.0)),
            "cvar_limit_pct": float(getattr(s, "RISK_CVAR_LIMIT_PCT", 10.0)),
            "min_signal_classes_agreeing": 1,
        }

    # ---------------------------------------------------------- sizing (Bayesian)
    def kelly_size(self, edge_mean: float, edge_std: float, atr_pct: float = 0.0):
        """Fractional Kelly on the lower credible bound of the edge.

        Cycle 19.6: the variance regulariser is now scaled by current ATR so
        sizing shrinks in volatile regimes and expands in calm ones. The old
        constant ``+0.02`` was regime-independent — fine for a single calm-vol
        crypto asset, wrong for a universe spanning low-vol stocks (BE ~0.4%)
        and high-vol crypto / hot stocks (3%+).

        ``atr_pct`` is the symbol's ATR as a fraction of price (e.g. 0.025 =
        2.5%). Caller may omit; the floor at 0.005 keeps behaviour close to the
        legacy value when the caller doesn't know ATR.

        Returns a position fraction in [min, max_single_position_pct], or None if
        Kelly sizing is disabled (NN_KELLY_FRACTION <= 0 -> use the model's size)."""
        frac = float(getattr(settings, "NN_KELLY_FRACTION", 0.5))
        if frac <= 0:
            return None
        max_pct = self.limits["max_single_position_pct"] / 100.0
        z = 1.0  # ~84% one-sided lower bound
        edge_lcb = abs(float(edge_mean)) - z * max(0.0, float(edge_std))
        if edge_lcb <= 0:
            return 0.02  # edge not robust under uncertainty -> minimum size
        # ATR-scaled variance floor — 0.5% min keeps behaviour bounded for
        # calm regimes; the 2.0x coefficient gives a meaningful shrink for
        # high-vol regimes (3% ATR → 6% floor → ~3x position cut).
        variance_floor = max(0.005, 2.0 * float(atr_pct or 0.0))
        variance = float(edge_std) ** 2 + variance_floor
        size = frac * (edge_lcb / variance)
        return float(min(max(size, 0.02), max_pct))

    # --------------------------------------------- Statistical R:R boundary floors
    def enforce_exit_floors(self, sl_frac: float, tp_frac: float, recent_vol: float) -> tuple[float, float]:
        """Vol-aware SL floor + minimum reward:risk. Prevents degenerate tiny stops and
        bad R:R: SL is bumped to max(0.3%, k·recent_return_std); TP is bumped to at least
        `NN_MIN_RR_RATIO * SL`. Used to give every order a 'statistically significant' boundary."""
        from backend.core.config import settings as _s
        k = float(getattr(_s, "NN_BOUNDARY_K_SIGMA", 1.0))
        min_rr = float(getattr(_s, "NN_MIN_RR_RATIO", 1.5))
        floor = max(0.003, k * float(recent_vol or 0.0))
        sl = max(float(sl_frac or 0.0), floor)
        tp = max(float(tp_frac or 0.0), sl * min_rr)
        return float(sl), float(tp)

    # ------------------------------------------------------------- Monte-Carlo CVaR
    def record_return(self, pnl_pct: float) -> None:
        self.recent_returns.append(float(pnl_pct))

    def monte_carlo_cvar(self, n_sims: int = 1000, horizon: int = 10, alpha: float = 0.95) -> float:
        """Bootstrap recent realized returns to project tail loss (CVaR) over the
        next `horizon` trades. Returns expected tail loss as a positive percent."""
        rets = np.asarray(self.recent_returns, dtype=float)
        if rets.size < 10:
            return 0.0
        rng = np.random.default_rng()
        draws = rng.choice(rets, size=(n_sims, horizon), replace=True).sum(axis=1)
        losses = -draws
        var = np.quantile(losses, alpha)
        tail = losses[losses >= var]
        cvar = float(tail.mean()) if tail.size else float(var)
        return max(0.0, cvar * 100.0)

    def approve(self, decision: TradeDecision, portfolio_state: dict) -> tuple[bool, str]:
        if self.is_halted:
            return False, "HALTED: max drawdown exceeded — manual reset required"

        if decision.direction == "hold":
            return False, "HOLD decision — no trade"

        if decision.nn_confidence < self.limits["min_nn_confidence"]:
            return False, f"Low confidence: {decision.nn_confidence:.3f}"

        self._reset_counters_if_needed()

        if self.trades_this_hour >= self.limits["max_trades_per_hour"]:
            return False, "Trade rate limit"

        available_cash = portfolio_state.get("available_cash", 0.0)
        size_pct = min(decision.size_pct, self.limits["max_single_position_pct"] / 100.0)
        position_usd = size_pct * available_cash

        if position_usd < self.limits["min_position_usd"]:
            return False, "Below min notional"
        if position_usd > self.limits["max_position_usd"]:
            return False, "Above max position"

        if self.peak_portfolio_value > 0:
            current_drawdown_pct = ((self.peak_portfolio_value - self.portfolio_value_usd) / self.peak_portfolio_value) * 100
            if current_drawdown_pct > self.limits["max_portfolio_drawdown_pct"]:
                self.is_halted = True
                logger.error("max_drawdown_exceeded_halting_trading", drawdown_pct=current_drawdown_pct)
                return False, "MAX DRAWDOWN EXCEEDED — HALTED"

        start_of_day_value = self.portfolio_value_usd - self.daily_pnl_usd
        if start_of_day_value > 0:
            daily_pnl_pct = (self.daily_pnl_usd / start_of_day_value) * 100
            if daily_pnl_pct < -self.limits["max_daily_loss_pct"]:
                return False, "Daily loss limit"

        # Monte-Carlo CVaR tail-risk gate
        self.last_cvar_pct = self.monte_carlo_cvar()
        if self.last_cvar_pct > self.limits["cvar_limit_pct"] > 0:
            return False, f"CVaR tail-risk limit ({self.last_cvar_pct:.1f}% > {self.limits['cvar_limit_pct']:.1f}%)"

        self.trades_this_hour += 1
        return True, "APPROVED"

    def _reset_counters_if_needed(self):
        now = datetime.utcnow()
        if now.replace(minute=0, second=0, microsecond=0) > self.last_hour_reset:
            self.trades_this_hour = 0
            self.last_hour_reset = now.replace(minute=0, second=0, microsecond=0)
        if now.replace(hour=0, minute=0, second=0, microsecond=0) > self.last_day_reset:
            self.daily_pnl_usd = 0.0
            self.last_day_reset = now.replace(hour=0, minute=0, second=0, microsecond=0)

    def update_portfolio(self, portfolio_state: dict) -> None:
        new_val = portfolio_state.get("total_value_usd", self.portfolio_value_usd)
        self.daily_pnl_usd += (new_val - self.portfolio_value_usd)
        self.portfolio_value_usd = new_val
        if self.portfolio_value_usd > self.peak_portfolio_value:
            self.peak_portfolio_value = self.portfolio_value_usd

    def reset_halt(self) -> None:
        self.is_halted = False
        self.peak_portfolio_value = self.portfolio_value_usd
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
            "cvar_pct": self.last_cvar_pct,
            "is_halted": self.is_halted,
            "limits": self.limits,
        }
