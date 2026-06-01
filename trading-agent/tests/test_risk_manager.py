import pytest
import sys
from pathlib import Path
from types import SimpleNamespace

# Add project root to path
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

from backend.risk.manager import RiskManager


def _decision(direction: str = "long", size_pct: float = 0.05, conf: float = 0.8):
    return SimpleNamespace(direction=direction, size_pct=size_pct, nn_confidence=conf)


def test_approve_normal_trade():
    rm = RiskManager(initial_portfolio_value=1000.0)
    approved, reason = rm.approve(_decision(), {"available_cash": 1000.0})
    assert approved is True
    assert reason == "APPROVED"


def test_reject_low_confidence():
    rm = RiskManager(initial_portfolio_value=1000.0)
    rm.HARD_LIMITS["min_nn_confidence"] = 0.7
    approved, reason = rm.approve(_decision(conf=0.6), {"available_cash": 1000.0})
    assert approved is False
    assert "Low confidence" in reason


def test_reject_above_max_position():
    rm = RiskManager(initial_portfolio_value=10000.0)
    # size clamps to the max single-position pct; at $50k cash the clamped
    # notional (15-20% = $7.5k-$10k) exceeds the $5k max-notional cap regardless
    # of the exact configured pct, so this stays valid as the cap is tuned.
    approved, reason = rm.approve(_decision(size_pct=0.9), {"available_cash": 50000.0})
    assert approved is False
    assert reason == "Above max position"


def test_halt_on_max_drawdown():
    rm = RiskManager(initial_portfolio_value=10000.0)
    rm.portfolio_value_usd = 8000.0
    approved, reason = rm.approve(_decision(), {"available_cash": 10000.0})
    assert approved is False
    assert "DRAWDOWN" in reason
    assert rm.is_halted is True


def test_halt_blocks_all_subsequent_trades():
    rm = RiskManager(initial_portfolio_value=1000.0)
    rm.is_halted = True
    approved, reason = rm.approve(_decision(), {"available_cash": 1000.0})
    assert approved is False
    assert "HALTED" in reason


def test_rate_limit():
    rm = RiskManager(initial_portfolio_value=1000.0)
    rm.HARD_LIMITS["max_trades_per_hour"] = 1
    ok1, _ = rm.approve(_decision(), {"available_cash": 1000.0})
    ok2, reason2 = rm.approve(_decision(), {"available_cash": 1000.0})
    assert ok1 is True
    assert ok2 is False
    assert reason2 == "Trade rate limit"


def test_daily_loss_limit():
    rm = RiskManager(initial_portfolio_value=1000.0)
    rm.daily_pnl_usd = -100.0
    approved, reason = rm.approve(_decision(), {"available_cash": 1000.0})
    assert approved is False
    assert reason == "Daily loss limit"
