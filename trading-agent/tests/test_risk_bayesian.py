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
    assert confident is not None and uncertain is not None
    assert confident >= uncertain  # higher uncertainty -> smaller (or floor) bet
    max_pct = rm.limits["max_single_position_pct"] / 100.0
    assert 0.02 <= uncertain <= max_pct
    assert 0.02 <= confident <= max_pct


def test_kelly_disabled_returns_none(monkeypatch):
    monkeypatch.setattr(cfg.settings, "NN_KELLY_FRACTION", 0.0)
    rm = RiskManager()
    assert rm.kelly_size(0.2, 0.01) is None


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
