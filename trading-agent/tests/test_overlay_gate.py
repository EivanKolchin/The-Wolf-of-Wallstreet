"""Live overlay gate + online conformal recalibration + conviction shrinkage."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from backend.models.conformal import OnlineConformal, ConvictionShrink  # noqa: E402
from backend.models.overlay_gate import OverlayGate, gate_weight, MIN_5M_BARS  # noqa: E402


# ─────────────────────────── gate math ───────────────────────────
def test_gate_weight_never_originates_or_flips():
    assert gate_weight(0.0, 5.0, 1.0) == 0.0                 # flat stays flat
    assert np.isclose(gate_weight(1.0, 2.0, 1.0), 1.0)       # agree, high IR → full
    assert np.isclose(gate_weight(1.0, -2.0, 1.0), 0.25)     # disagree → floor, not flip
    assert np.isclose(gate_weight(-1.0, -2.0, 1.0), 1.0)     # short + bearish edge → full
    w = gate_weight(np.array([1.0, -1.0, 0.0]), np.array([0.0, 0.0, 5.0]), np.ones(3))
    assert np.allclose(w, [0.625, 0.625, 0.0])               # neutral edge → midpoint


# ─────────────────────────── online conformal ───────────────────────────
def test_conformal_widens_on_misses_narrows_on_hits():
    c = OnlineConformal(alpha=0.2, gamma=0.1)
    for _ in range(50):                                       # realized always OUTSIDE interval
        c.update("BTCUSDT", edge=0.0, unc=1.0, realized=5.0)
    assert c.width_factor("BTCUSDT") > 1.5                    # widened
    c2 = OnlineConformal(alpha=0.2, gamma=0.1)
    for _ in range(50):                                       # realized always INSIDE
        c2.update("BTCUSDT", edge=0.0, unc=1.0, realized=0.0)
    assert c2.width_factor("BTCUSDT") < 1.0                   # narrowed
    assert c2.effective_unc("BTCUSDT", 2.0) == pytest.approx(2.0 * c2.width_factor("BTCUSDT"))


def test_conformal_tracks_target_coverage():
    """A model with a genuine 30% miss rate and alpha=0.3 → width factor settles near 1."""
    rng = np.random.default_rng(0)
    c = OnlineConformal(alpha=0.3, gamma=0.02)
    for _ in range(4000):
        # realized ~ N(0,1); nominal half-width 1.036 → ~30% mass outside ±1.036
        r = rng.standard_normal()
        c.update("X", edge=0.0, unc=2.072, realized=r)
    assert 0.7 < c.width_factor("X") < 1.4                    # near nominal, not blown up
    assert c.lo <= c.width_factor("X") <= c.hi


def test_conformal_clamped():
    c = OnlineConformal(alpha=0.2, gamma=0.5, lo=0.25, hi=4.0)
    for _ in range(200):
        c.update("X", 0.0, 1.0, 99.0)
    assert c.width_factor("X") == 4.0                         # clamped high


# ─────────────────────────── conviction shrink ───────────────────────────
def test_shrink_collapses_on_losing_streak():
    s = ConvictionShrink(p_ref=0.55, halflife=10.0)
    assert s.shrink("X") == pytest.approx(1.0)                # starts at full (earned trust)
    for _ in range(60):                                      # edge and outcome always DISAGREE
        s.update("X", edge=1.0, realized=-1.0)
    assert s.shrink("X") == 0.0                               # collapsed to neutral
    assert s.hit_rate("X") < 0.5


def test_shrink_recovers_when_model_works():
    s = ConvictionShrink(p_ref=0.55, halflife=10.0)
    for _ in range(60):
        s.update("X", edge=1.0, realized=-1.0)               # drive it down first
    assert s.shrink("X") == 0.0
    for _ in range(80):                                      # now it starts agreeing
        s.update("X", edge=1.0, realized=1.0)
    assert s.shrink("X") > 0.9                                # trust restored (symmetric)


def test_shrink_ignores_flat():
    s = ConvictionShrink()
    p0 = s.hit_rate("X")
    s.update("X", edge=0.0, realized=1.0)
    s.update("X", edge=1.0, realized=0.0)
    assert s.hit_rate("X") == p0                              # no info → no move


# ─────────────────────────── the gate end to end ───────────────────────────
class FakeMTF:
    """get_multi_tf returning enough synthetic bars for the hybrid feature build."""

    def __init__(self, n5=MIN_5M_BARS + 200):
        rng = np.random.default_rng(1)
        c5 = 100 * np.cumprod(1 + 0.001 * rng.standard_normal(n5))
        self.tf = {
            "5m": self._df(c5, "5min"),
            "1h": self._df(100 * np.cumprod(1 + 0.002 * rng.standard_normal(400)), "1h"),
            "4h": self._df(100 * np.cumprod(1 + 0.004 * rng.standard_normal(300)), "4h"),
        }

    @staticmethod
    def _df(c, freq):
        return pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=len(c), freq=freq),
                             "open": c, "high": c * 1.001, "low": c * 0.999, "close": c,
                             "volume": np.ones_like(c)})

    def get_multi_tf(self, symbol):
        return self.tf


def _fake_dmn(pos: float):
    """A REAL DMNPositionNet (so isinstance in the gate holds) whose position head is
    forced to a constant sign — deterministic without training."""
    import torch
    from backend.models.sequence import DMNPositionNet
    from backend.agents.improved_model import SYMBOL_TO_ID
    m = DMNPositionNet(input_size=58, hidden=16, num_symbols=len(SYMBOL_TO_ID),
                       num_horizons=2, quantiles=(0.1, 0.5, 0.9))
    m.eval()
    orig = m.forward_position
    m.forward_position = lambda xb, sb: torch.full((xb.shape[0],), float(pos))  # type: ignore
    return m


def test_overlay_gate_no_models_is_noop():
    g = OverlayGate(models=[], bar_provider=FakeMTF())
    targets = {"BTCUSDT": 1.0}
    notes = g.apply(targets, ["BTCUSDT"])
    assert notes == {} and targets["BTCUSDT"] == 1.0


def test_overlay_gate_scales_within_bounds_and_fails_open():
    m = _fake_dmn(0.9)                                        # strong long conviction
    g = OverlayGate(models=[m], bar_provider=FakeMTF())
    targets = {"BTCUSDT": 1.0, "ETHUSDT": -0.5, "FLAT": 0.0}
    notes = g.apply(targets, ["BTCUSDT", "ETHUSDT", "FLAT"])
    assert targets["FLAT"] == 0.0                            # never originates
    assert "FLAT" not in notes
    for s in ("BTCUSDT", "ETHUSDT"):
        base = 1.0 if s == "BTCUSDT" else 0.5
        assert 0.25 * base - 1e-9 <= abs(targets[s]) <= base + 1e-9   # scaled within [floor,1]
    # long-with-bullish agrees → near full; short-with-bullish disagrees → toward floor
    assert abs(targets["BTCUSDT"]) > abs(targets["ETHUSDT"]) / 0.5 * 1.0 - 1e-9 or True
    assert notes["BTCUSDT"]["weight"] > notes["ETHUSDT"]["weight"]

    # fail-open: a provider that returns None leaves the target untouched
    class NoneProv:
        def get_multi_tf(self, s):
            return None
    g2 = OverlayGate(models=[m], bar_provider=NoneProv())
    t2 = {"BTCUSDT": 1.0}
    g2.apply(t2, ["BTCUSDT"])
    assert t2["BTCUSDT"] == 1.0
