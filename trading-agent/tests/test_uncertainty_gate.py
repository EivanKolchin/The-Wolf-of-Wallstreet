"""Workstream C test: uncertainty gate (edge must dominate MC-dropout spread)."""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from backend.agents.nn_agent import passes_uncertainty_gate  # noqa: E402


def test_high_conviction_passes():
    # strong edge, low spread -> ratio 5.0 >= 1.0
    assert passes_uncertainty_gate(0.05, 0.01, 1.0) is True


def test_low_conviction_blocked():
    # weak edge vs large spread -> ratio 0.2 < 1.0
    assert passes_uncertainty_gate(0.01, 0.05, 1.0) is False


def test_gate_zero_disables():
    assert passes_uncertainty_gate(0.0, 0.5, 0.0) is True
    assert passes_uncertainty_gate(0.001, 0.5, 0.0) is True


def test_threshold_boundary():
    # comfortably above the gate (ratio ~2.0) passes
    assert passes_uncertainty_gate(0.02, 0.01, 1.0) is True
    # just under the gate fails (and the +eps denominator keeps exact-equal
    # ratios marginally conservative, i.e. a hair below the gate -> hold)
    assert passes_uncertainty_gate(0.0199, 0.02, 1.0) is False
    assert passes_uncertainty_gate(0.02, 0.02, 1.0) is False  # eps tips equal-ratio to hold
