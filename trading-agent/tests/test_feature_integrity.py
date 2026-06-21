"""Phase A correctness: the offline feature builder must be CAUSAL (no look-ahead) and
the revived features (VWAP feat 10, BB-width feat 18) must carry real signal, not be dead
or leak the future. These guard the fixes in scripts/pretrain.py build_feature_matrix and
the shared backend/signals/feature_spec.rolling_vwap_distance helper."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

import importlib.util

_spec = importlib.util.spec_from_file_location("pre", str(ROOT / "scripts" / "pretrain.py"))
pre = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pre)

from backend.signals import feature_spec as fs  # noqa: E402


def _synthetic_df(n=400, seed=0):
    rng = np.random.default_rng(seed)
    ret = rng.standard_normal(n) * 0.01 + 0.0003
    close = 100.0 * np.cumprod(1.0 + ret)
    high = close * (1.0 + np.abs(rng.standard_normal(n)) * 0.003)
    low = close * (1.0 - np.abs(rng.standard_normal(n)) * 0.003)
    op = np.concatenate([[close[0]], close[:-1]])
    vol = rng.uniform(100, 1000, n)
    ts = pd.date_range("2023-01-01", periods=n, freq="5min")
    return pd.DataFrame({"timestamp": ts, "open": op, "high": high, "low": low,
                         "close": close, "volume": vol})


def test_feature_matrix_has_no_lookahead():
    """Perturbing the LAST bar must not change ANY earlier feature row — a strict, general
    no-look-ahead guarantee (catches the np.gradient centered-difference leak on feats 16/21)."""
    df = _synthetic_df()
    base = pre.build_feature_matrix(df)

    df2 = df.copy()
    last = len(df2) - 1
    df2.loc[last, "close"] *= 1.05          # shock only the final bar
    df2.loc[last, "high"] *= 1.05
    df2.loc[last, "low"] *= 0.95
    df2.loc[last, "volume"] *= 3.0
    base2 = pre.build_feature_matrix(df2)

    # Every row before the last must be bit-identical → no feature peeked at the future.
    # equal_nan=True: the fib (22-24) warmup is NaN in BOTH (min_periods=50) and is later
    # zeroed by apply_rolling_zscore's fillna — NaN!=NaN is not look-ahead.
    assert np.allclose(base[:-1], base2[:-1], atol=0, rtol=0, equal_nan=True), \
        "a feature changed on an earlier bar when only the last bar moved → look-ahead"


def test_rsi_div_and_obv_are_causal_specifically():
    """Targeted check on the two feats that used np.gradient (16 rsi_div, 21 obv slope)."""
    df = _synthetic_df(seed=3)
    a = pre.build_feature_matrix(df)
    df2 = df.copy(); df2.loc[len(df2) - 1, "close"] *= 1.1
    b = pre.build_feature_matrix(df2)
    for idx in (16, 21):
        assert np.array_equal(a[:-1, idx], b[:-1, idx])


def test_vwap_feature_is_alive_not_flat():
    """The old cumulative-from-epoch VWAP decayed to a near-constant. The rolling VWAP must
    actually vary across a long series (carry signal)."""
    df = _synthetic_df(n=2000, seed=1)
    base = pre.build_feature_matrix(df)
    vwap_feat = base[200:, 10]                       # skip warmup
    assert np.std(vwap_feat) > 1e-3                  # not a dead constant


def test_bb_width_not_dead_after_column_fix():
    """Feat 18 (BB width) was clipped to 0 by the swapped-band bug; now it must be positive."""
    df = _synthetic_df(n=600, seed=2)
    base = pre.build_feature_matrix(df)
    bb_width = base[100:, 18]
    assert np.mean(bb_width) > 0.0 and np.max(bb_width) > 0.0


def test_rolling_vwap_helper_is_causal():
    rng = np.random.default_rng(5)
    n = 500
    close = 100 * np.cumprod(1 + rng.standard_normal(n) * 0.01)
    high = close * 1.002; low = close * 0.998; vol = rng.uniform(1, 5, n)
    v1 = fs.rolling_vwap_distance(high, low, close, vol)
    close2 = close.copy(); close2[-1] *= 1.2          # move only the last bar
    high2 = high.copy(); high2[-1] *= 1.2
    v2 = fs.rolling_vwap_distance(high2, low, close2, vol)
    assert np.array_equal(v1[:-1], v2[:-1])           # earlier values unchanged


def _resample(df, rule):
    g = df.set_index("timestamp").resample(rule)
    r = pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(),
                      "close": g["close"].last(), "volume": g["volume"].sum()}).dropna().reset_index()
    return r


def test_htf_features_are_causal():
    """An HTF (1h/4h) feature for a 5m bar must use only candles that have FULLY CLOSED before
    that bar. Perturbing a 4h candle must not change the HTF features of any 5m bar that precedes
    that candle's close — the leak the signal-audit flagged (0.8 AUC at H+12)."""
    n5 = 60 * 12                                    # 60 hours of 5m bars
    t5 = pd.date_range("2023-01-01", periods=n5, freq="5min")
    rng = np.random.default_rng(0)
    c5 = 100 * np.cumprod(1 + rng.standard_normal(n5) * 0.005)
    df5 = pd.DataFrame({"timestamp": t5, "open": c5, "high": c5 * 1.001,
                        "low": c5 * 0.999, "close": c5, "volume": 1.0})
    df1h, df4h = _resample(df5, "1h"), _resample(df5, "4h")

    htf = pre.build_htf_features(df5, df1h, df4h)
    df4h2 = df4h.copy()
    j = len(df4h2) - 1                              # perturb the LAST 4h candle
    df4h2.loc[j, "close"] *= 1.5
    df4h2.loc[j, "high"] *= 1.5
    htf2 = pre.build_htf_features(df5, df1h, df4h2)

    close_time = df4h2.loc[j, "timestamp"] + df4h2["timestamp"].diff().median()
    pre_close = np.asarray(t5 < close_time)         # 5m bars before that candle closed
    # 4h HTF columns (4-7) must be identical for every bar that precedes the candle's close.
    assert np.allclose(htf[pre_close][:, 4:8], htf2[pre_close][:, 4:8]), \
        "perturbing a 4h candle changed an earlier 5m bar's HTF features → look-ahead"


def test_rolling_vwap_offline_full_equals_live_buffer_tail():
    """Offline computes on the FULL multi-year series; live computes on a short recent buffer.
    A rolling window makes them AGREE on the recent bars once ≥window history exists — this is
    the train/serve parity guarantee that the cumulative VWAP violated."""
    rng = np.random.default_rng(6)
    n = 3000
    close = 100 * np.cumprod(1 + rng.standard_normal(n) * 0.008)
    high = close * 1.0015; low = close * 0.9985; vol = rng.uniform(1, 9, n)
    full = fs.rolling_vwap_distance(high, low, close, vol, window=fs.VWAP_WINDOW)
    k = 50
    buf = fs.VWAP_WINDOW + k
    tail = fs.rolling_vwap_distance(high[-buf:], low[-buf:], close[-buf:], vol[-buf:],
                                    window=fs.VWAP_WINDOW)
    # the last k bars each have ≥window history in BOTH computations → identical
    assert np.allclose(full[-k:], tail[-k:], atol=1e-6)
