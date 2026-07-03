"""On-chain stablecoin features — causal daily→bar alignment, momentum math, no look-ahead."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.data.onchain_feed import (  # noqa: E402
    align_daily_to_bars, stablecoin_features, STABLECOIN_FEATURE_LABELS,
)

DAY_MS = 86_400_000


def test_align_is_causal_forward_fill():
    days = np.array([0, DAY_MS, 2 * DAY_MS], dtype=np.int64)
    vals = np.array([10.0, 20.0, 30.0])
    bars = np.array([-1, 0, DAY_MS // 2, DAY_MS, 2 * DAY_MS + 5], dtype=np.int64)
    out = align_daily_to_bars(days, vals, bars)
    assert np.isnan(out[0])                       # before first day → NaN
    assert out[1] == 10.0                          # exactly on day 0
    assert out[2] == 10.0                          # mid-day 0 → still day-0 value (no peek)
    assert out[3] == 20.0
    assert out[4] == 30.0


def test_features_shape_and_labels():
    days = np.arange(60, dtype=np.int64) * DAY_MS
    mcap = 100.0 + np.arange(60) * 2.0             # steadily minting
    bars = np.arange(0, 60 * DAY_MS, DAY_MS // 2, dtype=np.int64)
    out = stablecoin_features(days, mcap, bars)
    assert out.shape == (len(bars), 3)
    assert len(STABLECOIN_FEATURE_LABELS) == 3
    assert out.dtype == np.float32


def test_momentum_positive_when_minting():
    days = np.arange(60, dtype=np.int64) * DAY_MS
    mcap = 100.0 * (1.01 ** np.arange(60))         # +1%/day compounding mint
    bars = days.copy()
    out = stablecoin_features(days, mcap, bars, mom_short=7, mom_long=30)
    # last bar: 7d and 30d momentum both clearly positive
    assert out[-1, 0] > 0 and out[-1, 1] > 0
    # short-window momentum on a compounding series < long-window cumulative → accel negative-ish;
    # the point is only that accel = short − long is computed and finite
    assert np.isfinite(out[-1, 2])


def test_warmup_and_short_history_safe():
    days = np.arange(5, dtype=np.int64) * DAY_MS   # fewer than mom_long
    mcap = np.array([1, 2, 3, 4, 5.0])
    bars = days.copy()
    out = stablecoin_features(days, mcap, bars, mom_short=7, mom_long=30)
    assert out.shape == (5, 3)
    assert np.all(out == 0.0)                       # too short → all zeros (no crash)


def test_burning_gives_negative_momentum():
    days = np.arange(60, dtype=np.int64) * DAY_MS
    mcap = 200.0 * (0.99 ** np.arange(60))         # burning 1%/day
    out = stablecoin_features(days, mcap, days.copy())
    assert out[-1, 0] < 0 and out[-1, 1] < 0
