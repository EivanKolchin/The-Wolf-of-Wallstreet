"""Unit tests for the PURE derivatives feature logic (no network): causal alignment
and funding/OI feature builders. The critical property is NO LOOK-AHEAD — a bar may
only ever see funding/OI known at or before its own timestamp."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from backend.data.derivatives_feed import (  # noqa: E402
    align_to_bars, funding_features, open_interest_features,
)

MIN5 = 300_000      # 5 minutes in ms
H8 = 8 * 3600_000   # 8 hours in ms (funding cadence)


def _bars(n=120, t0=0):
    return t0 + np.arange(n) * MIN5


def test_align_forward_fills_causally():
    bars = _bars(200)                                         # 200×5m ≈ 16.7h spans the 16h event
    ev_ts = np.array([0, H8, 2 * H8], dtype=np.int64)        # funding at 0h, 8h, 16h
    ev_val = np.array([0.0001, -0.0002, 0.0003])
    out = align_to_bars(ev_ts, ev_val, bars)
    assert out[0] == 0.0001                                   # bar at the event time sees it
    assert out[H8 // MIN5 - 1] == 0.0001                      # bar just before 8h still on first
    assert out[H8 // MIN5] == -0.0002                         # bar at 8h flips to second
    assert out[2 * H8 // MIN5] == 0.0003


def test_align_pre_first_event_is_fill():
    bars = _bars(10, t0=0)
    ev_ts = np.array([5 * MIN5], dtype=np.int64)              # first event after bar 0..4
    ev_val = np.array([0.01])
    out = align_to_bars(ev_ts, ev_val, bars, fill=0.0)
    assert (out[:5] == 0.0).all()                             # no value known yet -> fill
    assert (out[5:] == 0.01).all()


def test_align_no_lookahead():
    """A huge funding spike far in the future must not affect any earlier bar."""
    bars = _bars(50)
    ev_ts = np.array([0, 100 * H8], dtype=np.int64)
    ev_val = np.array([0.0001, 999.0])                        # absurd future value
    out = align_to_bars(ev_ts, ev_val, bars)
    assert (out == 0.0001).all()                              # future event invisible


def test_funding_features_shape_and_level():
    bars = _bars(200)
    ev_ts = np.array([0, H8, 2 * H8], dtype=np.int64)
    ev_val = np.array([0.0001, -0.0002, 0.0003])
    feats = funding_features(ev_ts, ev_val, bars, z_window=50, carry_window=20)
    assert feats.shape == (200, 4)
    assert np.isfinite(feats).all()
    assert abs(feats[0, 0] - 0.0001) < 1e-9                   # level col
    assert abs(feats[H8 // MIN5, 1] - (-0.0003)) < 1e-9       # change col = -0.0002 - 0.0001


def test_funding_features_no_events_is_zero():
    bars = _bars(60)
    feats = funding_features(np.array([], dtype=np.int64), np.array([]), bars)
    assert feats.shape == (60, 4)
    assert (feats == 0.0).all()


def test_open_interest_features():
    bars = _bars(120)
    ev_ts = (np.arange(0, 120, 6) * MIN5).astype(np.int64)    # OI every 30m
    ev_val = 1000.0 + np.arange(len(ev_ts)) * 10.0            # steadily building OI
    feats = open_interest_features(ev_ts, ev_val, bars, z_window=40)
    assert feats.shape == (120, 2)
    assert np.isfinite(feats).all()
