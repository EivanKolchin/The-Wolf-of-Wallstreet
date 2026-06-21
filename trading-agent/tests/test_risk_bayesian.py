"""Phase 5 tests: Bayesian fractional-Kelly sizing + Monte-Carlo CVaR gate."""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from backend.risk.manager import RiskManager
import backend.core.config as cfg


def test_kelly_size_shrinks_with_uncertainty():
    rm = RiskManager()
    confident = rm.kelly_size(edge_mean=0.2, edge_std=0.01)
    uncertain = rm.kelly_size(edge_mean=0.2, edge_std=0.40)
    assert confident is not None
    # IR = 0.2/0.4 = 0.5 < 1.5 gate -> should return None
    assert uncertain is None
    max_pct = rm.limits["max_single_position_pct"] / 100.0
    assert 0.0 <= confident <= max_pct


def test_kelly_disabled_returns_none(monkeypatch):
    monkeypatch.setattr(cfg.settings, "NN_KELLY_FRACTION", 0.0)
    rm = RiskManager()
    assert rm.kelly_size(0.2, 0.01) is None


def test_kelly_no_edge_returns_none():
    rm = RiskManager()
    # IR = 0.0/0.1 = 0.0 < 1.5 -> None (no information ratio, no bet)
    assert rm.kelly_size(0.0, 0.1) is None
    # weak edge: IR = 0.05/0.1 = 0.5 < 1.5 -> None (not robust enough)
    assert rm.kelly_size(-0.05, 0.1) is None


def test_kelly_strong_negative_edge_sizes_a_short():
    # kelly_size returns a position MAGNITUDE; direction is decided upstream. A confident
    # negative edge (IR = 0.1/0.05 = 2.0) is a valid SHORT and should be sized, not skipped.
    rm = RiskManager()
    sz = rm.kelly_size(-0.1, 0.05)
    assert sz is not None and sz > 0


def test_kelly_low_ir_gate():
    rm = RiskManager()
    # IR = 0.03/0.02 = 1.5 -> exactly at gate, should return size
    at_gate = rm.kelly_size(0.03, 0.02)
    assert at_gate is not None
    # IR = 0.029/0.02 = 1.45 < 1.5 -> None
    below_gate = rm.kelly_size(0.029, 0.02)
    assert below_gate is None


def test_monte_carlo_cvar_positive_on_losses():
    rm = RiskManager()
    for _ in range(60):
        rm.record_return(-0.02)
    assert rm.monte_carlo_cvar() > 0.0


def test_cvar_gate_blocks_when_over_limit():
    rm = RiskManager()
    rm.limits["cvar_limit_pct"] = 1.0  # very tight
    for _ in range(60):
        rm.record_return(-0.05)
    decision = SimpleNamespace(direction="long", nn_confidence=0.9, size_pct=0.1)
    ok, reason = rm.approve(decision, {"available_cash": 1000.0})
    assert ok is False and "CVaR" in reason


def test_limits_are_config_driven(monkeypatch):
    monkeypatch.setattr(cfg.settings, "RISK_MAX_POSITION_PCT", 7.5)
    rm = RiskManager()
    assert rm.limits["max_single_position_pct"] == 7.5
