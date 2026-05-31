"""Cycle 23 — the phantom-drawdown halt that made "the backend not do much".

Root cause: ``RiskManager()`` defaulted ``peak_portfolio_value`` to 10_000 while
the agent's real book is ``INITIAL_USDC_AMOUNT`` (1_000). The first ``approve()``
then read ``(10000-1000)/10000 = 90%`` drawdown > the 15% limit and latched
``is_halted=True`` *forever*, rejecting every subsequent decision with
"HALTED: max drawdown exceeded — manual reset required". The live decision log
showed 200/200 recent decisions rejected for exactly that reason with an empty
trades table — i.e. a pure phantom, not real losses.

Fixes under test:
  * main.py constructs ``RiskManager(initial_portfolio_value=INITIAL_USDC_AMOUNT)``.
  * nn_agent.run() re-baselines value+peak to the real book at startup.
  * reset_halt() clears the latch.
"""
import sys
from datetime import datetime
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))


def _decision(direction="long", size_pct=0.1, conf=0.6):
    from backend.agents.nn_agent import TradeDecision
    return TradeDecision(
        symbol="BTCUSDT", direction=direction, size_pct=size_pct,
        nn_confidence=conf, nn_probs={"long": conf, "short": 0.2, "hold": 0.2},
        regime="trending", active_news=None, timestamp=datetime.utcnow(),
    )


def test_seeded_to_initial_cash_does_not_phantom_halt():
    """The fix: peak==value==starting cash → 0% drawdown → approves, no latch."""
    from backend.risk.manager import RiskManager
    rm = RiskManager(initial_portfolio_value=1000.0)
    rm.update_portfolio({"total_value_usd": 1000.0})
    approved, reason = rm.approve(_decision(size_pct=0.1), {"available_cash": 1000.0})
    assert rm.is_halted is False
    assert approved is True, f"unexpected rejection: {reason}"


def test_old_default_init_would_have_phantom_halted():
    """Regression doc: the OLD code path (default 10k peak vs a 1k book) latches
    halted on the very first approve — this is the bug we removed in main.py."""
    from backend.risk.manager import RiskManager
    rm = RiskManager()  # default 10_000 peak — the pre-fix construction
    rm.update_portfolio({"total_value_usd": 1000.0})  # real book is ~1k
    approved, reason = rm.approve(_decision(), {"available_cash": 1000.0})
    assert approved is False
    assert rm.is_halted is True
    assert "DRAWDOWN" in reason.upper()


def test_real_15pct_drawdown_still_halts():
    """The fix must NOT defang the breaker: a genuine >15% drop from peak still
    trips the halt. Peak 1000 → value 800 = 20% drawdown → halt."""
    from backend.risk.manager import RiskManager
    rm = RiskManager(initial_portfolio_value=1000.0)
    rm.update_portfolio({"total_value_usd": 800.0})
    approved, reason = rm.approve(_decision(), {"available_cash": 800.0})
    assert approved is False
    assert rm.is_halted is True


def test_reset_halt_clears_latch_and_rebaselines_peak():
    from backend.risk.manager import RiskManager
    rm = RiskManager(initial_portfolio_value=1000.0)
    rm.update_portfolio({"total_value_usd": 800.0})
    rm.approve(_decision(), {"available_cash": 800.0})
    assert rm.is_halted is True
    rm.reset_halt()
    assert rm.is_halted is False
    # peak re-baselined to current value so the drawdown breaker won't instantly
    # re-trip (0% drawdown from the new high-water mark).
    assert rm.peak_portfolio_value == pytest.approx(rm.portfolio_value_usd)
    # Clear the (independent, correctly-active) daily-loss state so we isolate the
    # drawdown latch: with a fresh day and 0% drawdown the agent approves again.
    rm.daily_pnl_usd = 0.0
    approved, reason = rm.approve(_decision(), {"available_cash": 800.0})
    assert approved is True, f"unexpected rejection after reset: {reason}"
