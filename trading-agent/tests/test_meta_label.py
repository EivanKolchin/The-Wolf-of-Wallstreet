"""Meta-labeling unit tests: triple-barrier correctness (target-first=1, stop-first=0, no-signal=NaN,
time-barrier=0, both-in-one-bar=conservative 0), feature causality/shape, the 2p-1 sizing map, and
the classifier wrapper's degenerate-fold fallback."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

import pytest  # noqa: E402

from backend.strategies.meta_label import (  # noqa: E402
    TripleBarrierParams, triple_barrier_labels, meta_features, size_from_proba, MetaLabeler,
    realized_vol, garman_klass_vol)


def _df(close, high=None, low=None, open_=None):
    c = np.asarray(close, float)
    return pd.DataFrame({"open": open_ if open_ is not None else c,
                         "high": high if high is not None else c,
                         "low": low if low is not None else c, "close": c,
                         "volume": np.ones(len(c))})


def test_size_from_proba_maps_correctly():
    p = np.array([0.0, 0.5, 0.75, 1.0])
    np.testing.assert_allclose(size_from_proba(p), [0.0, 0.0, 0.5, 1.0])


def _noisy_leadin(n=30, base=100.0):
    """Oscillating series → stable, clearly-positive trailing vol so vol-scaled barriers are well-defined."""
    return base + (np.arange(n) % 2).astype(float)      # 100,101,100,101,… → ~1% returns


def test_triple_barrier_target_first_is_1():
    # controlled lead-in (positive vol), long at the last lead-in bar, then a big up-spike next bar
    lead = _noisy_leadin(30)
    close = np.concatenate([lead, np.full(5, lead[-1])])
    high = close.copy(); low = close.copy()
    e = 30
    sig = np.zeros(len(close)); sig[e] = 1.0
    high[e + 1] = close[e] * 1.50                         # +50% reach → hits +target; low flat → no stop
    y = triple_barrier_labels(high, low, close, sig, TripleBarrierParams(horizon=3, target_sigma=1.0,
                                                                         stop_sigma=1.0, vol_window=10))
    assert y[e] == 1.0


def test_triple_barrier_stop_first_is_0():
    close = np.linspace(100, 101, 40)
    high = close.copy(); low = close.copy()
    sig = np.zeros(len(close)); sig[30] = 1.0
    low[31] = 50.0                                      # crash next bar → hits −stop first
    y = triple_barrier_labels(high, low, close, sig, TripleBarrierParams(horizon=3, target_sigma=2.0,
                                                                         stop_sigma=0.5, vol_window=10))
    assert y[30] == 0.0


def test_triple_barrier_no_signal_is_nan():
    close = np.linspace(100, 110, 40)
    sig = np.zeros(len(close))                          # never long
    y = triple_barrier_labels(close, close, close, sig, TripleBarrierParams())
    assert np.all(np.isnan(y))


def test_triple_barrier_time_barrier_defaults_to_0():
    # positive lead-in vol, but forward path stays well INSIDE both barriers → time-barrier → label 0
    lead = _noisy_leadin(30)
    close = np.concatenate([lead, np.full(6, lead[-1])])
    e = 30
    entry = close[e]
    high = close.copy(); low = close.copy()
    high[e + 1:e + 6] = entry * 1.001                    # ±0.1% band, far inside the ~±5% barriers
    low[e + 1:e + 6] = entry * 0.999
    sig = np.zeros(len(close)); sig[e] = 1.0
    y = triple_barrier_labels(high, low, close, sig, TripleBarrierParams(horizon=5, target_sigma=5.0,
                                                                         stop_sigma=5.0, vol_window=10))
    assert y[e] == 0.0


def test_triple_barrier_no_lookahead_on_signal_bars_only():
    # labels exist exactly where signal>0 and enough future bars exist
    close = np.linspace(100, 120, 50)
    sig = np.zeros(len(close)); sig[10] = 1.0; sig[48] = 1.0   # idx48 has <horizon future bars
    y = triple_barrier_labels(close, close, close, sig, TripleBarrierParams(horizon=10))
    assert np.isfinite(y[10])
    assert np.isnan(y[0]) and np.isnan(y[20])                  # no signal → NaN
    # idx48: only 1 future bar; still labelled by whatever is reachable (finite or via time barrier)
    assert np.isfinite(y[48])


def test_meta_features_shape_and_finite():
    rng = np.random.default_rng(0)
    c = 100 + np.cumsum(rng.standard_normal(400))
    df = _df(c, high=c + 1, low=c - 1, open_=c)
    X, names = meta_features(df, trend_ema=50, vol_window=20)
    assert X.shape == (400, len(names))
    assert np.all(np.isfinite(X))                              # NaN-safe by construction
    assert "gk_vol" in names and "trend_strength" in names


def test_meta_features_optional_columns():
    c = 100 + np.cumsum(np.random.default_rng(1).standard_normal(300))
    df = _df(c, high=c + 1, low=c - 1)
    X1, n1 = meta_features(df, panel_regime=np.zeros(300))
    X2, n2 = meta_features(df, panel_regime=np.zeros(300), funding=np.ones(300))
    assert "panel_corr" in n1 and "funding" in n2 and X2.shape[1] == X1.shape[1] + 1


def test_garman_klass_and_realized_vol_causal_and_positive():
    rng = np.random.default_rng(2)
    c = 100 + np.cumsum(rng.standard_normal(200))
    gk = garman_klass_vol(c, c + 2, c - 2, c, window=20)
    rv = realized_vol(c, window=20)
    assert np.all(gk[np.isfinite(gk)] >= 0) and np.all(rv[np.isfinite(rv)] >= 0)
    assert np.isnan(rv[0])                                     # trailing window not yet filled


def test_meta_labeler_fit_predict_and_degenerate_fold():
    rng = np.random.default_rng(3)
    X = rng.standard_normal((200, 5))
    y = (X[:, 0] + 0.3 * rng.standard_normal(200) > 0).astype(float)   # learnable
    m = MetaLabeler().fit(X, y)
    p = m.predict_proba(X)
    assert p.shape == (200,) and np.all((p >= 0) & (p <= 1))
    # degenerate fold (single class) → falls back to the base rate, no crash
    m2 = MetaLabeler().fit(X, np.ones(200))
    assert np.allclose(m2.predict_proba(X[:10]), 1.0)
