"""Tests for the lean Phase-2 hybrid feature contract: the ACTIVE (pruned) index set,
the multi-scale momentum features, and build_hybrid_matrix. The audit showed ~41 of 90
features are dead (stubbed offline); the hybrid trains on the 49 live ones + 9 momentum."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from signals import feature_spec as fs  # noqa: E402
from backend.features import pipeline as pl  # noqa: E402


def _ohlcv(n=2200, seed=5):
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.standard_normal(n) * 0.4)
    close = np.maximum(close, 1.0)
    high = close + np.abs(rng.standard_normal(n)) * 0.3
    low = close - np.abs(rng.standard_normal(n)) * 0.3
    op = close + rng.standard_normal(n) * 0.2
    ts = pd.date_range("2022-01-01", periods=n, freq="5min")
    return pd.DataFrame({"timestamp": ts, "open": op, "high": high, "low": low,
                         "close": close, "volume": rng.uniform(10, 1000, n)})


def test_active_indices_drop_the_stubbed_blocks():
    active = set(fs.ACTIVE_FEATURE_INDICES)
    assert len(fs.ACTIVE_FEATURE_INDICES) == 49           # 90 - 41 dead
    # stubbed-offline blocks must be excluded …
    for i in list(range(fs.ORDERBOOK.start, fs.ORDERBOOK.stop)) + \
            list(range(fs.NEWS.start, fs.NEWS.stop)) + \
            list(range(fs.MACRO.start, fs.MACRO.stop)) + \
            list(range(fs.NEWS_EMBED_START, fs.NEWS_EMBED_END)) + \
            list(range(fs.EARNINGS_START, fs.INPUT)) + [fs.SPREAD, 16, 34]:
        assert i not in active
    # … and real trend/HTF features must be kept
    for name in ("htf_4h_ema21_dist", "vwap_dist", "ema200_dist", "atr_norm", "rsi"):
        assert fs.FEATURE_NAMES.index(name) in active


def test_momentum_features_shape_and_causality():
    df = _ohlcv()
    m = pl.momentum_features(df)
    assert m.shape == (len(df), 9)
    assert np.isfinite(m).all()
    # warm-up: the longest lookback (mom_2016, col 3) is masked to 0 before 2016 bars
    assert (m[:2016, 3] == 0.0).all()


def test_momentum_rising_series_is_positive():
    df = _ohlcv()
    df["close"] = np.linspace(100, 300, len(df))          # monotonic rise
    df["high"] = df["close"] + 0.1; df["low"] = df["close"] - 0.1
    m = pl.momentum_features(df)
    # mom_288 (col 1) and range_1440 (col 7) should be positive late in a rising series
    assert m[-1, 1] > 0 and m[-1, 7] > 0.5


def test_build_hybrid_matrix_width_and_names():
    df5 = _ohlcv()
    df1h = df5.set_index("timestamp").resample("1h").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna().reset_index()
    df4h = df5.set_index("timestamp").resample("4h").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna().reset_index()
    H = pl.build_hybrid_matrix(df5, df1h, df4h)
    assert H.shape == (len(df5), 58)
    assert len(pl.HYBRID_FEATURE_NAMES) == 58
    assert np.isfinite(H).all()
    # last 9 columns are the momentum block
    assert pl.HYBRID_FEATURE_NAMES[-9:] == pl.MOMENTUM_NAMES
