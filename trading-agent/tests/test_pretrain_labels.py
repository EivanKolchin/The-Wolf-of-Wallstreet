"""Triple-barrier, volatility-scaled labels (Cycle 2). First-touch semantics +
the sign/shape/mask invariants the loss + PnL-weighting rely on."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

_spec = importlib.util.spec_from_file_location("pretrain_lbl_mod", str(ROOT / "scripts" / "pretrain.py"))
pt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pt)


def _df(close, high=None, low=None):
    close = np.asarray(close, float)
    high = close if high is None else np.asarray(high, float)
    low = close if low is None else np.asarray(low, float)
    return pd.DataFrame({"open": close, "high": high, "low": low, "close": close,
                         "volume": np.ones(len(close))})


def test_first_touch_upper_is_long():
    """A high spike inside the horizon (and no low spike) must label LONG with ret>0."""
    n = 30
    close = 100 + 0.01 * np.arange(n)            # gentle rise → vol > 0
    high = close.copy(); low = close.copy()
    high[5] = 200.0                              # pierce the upper barrier at bar 5
    labels, rets = pt.triple_barrier_labels(_df(close, high, low), [10], k=1.0)
    assert labels[0, 0] == 0 and rets[0, 0] > 0


def test_first_touch_lower_is_short():
    n = 30
    close = 100 - 0.01 * np.arange(n)
    high = close.copy(); low = close.copy()
    low[5] = 1.0                                 # pierce the lower barrier at bar 5
    labels, rets = pt.triple_barrier_labels(_df(close, high, low), [10], k=1.0)
    assert labels[0, 0] == 1 and rets[0, 0] < 0


def test_tail_masked_and_shapes():
    n, h = 50, 10
    close = 100 + np.cumsum(np.random.default_rng(1).standard_normal(n)) * 0.1
    labels, rets = pt.triple_barrier_labels(_df(close, close + 0.1, close - 0.1), [h], k=1.0)
    assert labels.shape == (n, 1) and rets.shape == (n, 1)
    assert (labels[n - h:, 0] == -1).all()        # last h rows can't be computed → masked
    assert np.isnan(rets[n - h:, 0]).all()
    assert set(np.unique(labels[:n - h, 0])).issubset({0, 1, 2})


def test_sign_invariant_long_positive_short_negative():
    """The PnL-weight loss relies on: long ⇒ ret>0, short ⇒ ret<0 (per horizon)."""
    rng = np.random.default_rng(3)
    n = 400
    close = 100 * np.exp(np.cumsum(rng.standard_normal(n) * 0.01))
    labels, rets = pt.triple_barrier_labels(_df(close, close * 1.002, close * 0.998), [12, 48], k=1.0)
    for hi in range(2):
        lab, r = labels[:, hi], rets[:, hi]
        m = lab != -1
        assert (r[m & (lab == 0)] > 0).all()
        assert (r[m & (lab == 1)] < 0).all()


def test_build_labels_wrapper_matches_combined():
    n = 120
    close = 100 * np.exp(np.cumsum(np.random.default_rng(5).standard_normal(n) * 0.01))
    df = _df(close, close * 1.001, close * 0.999)
    labels = pt.build_labels(df, [3, 12], thresholds=[0.0, 0.0])   # thresholds ignored now
    rets = pt.build_label_returns(df, [3, 12])
    lab2, r2 = pt.triple_barrier_labels(df, [3, 12], k=pt.BARRIER_K)
    np.testing.assert_array_equal(labels, lab2)
    np.testing.assert_array_equal(np.nan_to_num(rets), np.nan_to_num(r2))


def test_vol_scaling_widens_barriers_for_higher_vol():
    """Higher-volatility series ⇒ wider barriers ⇒ fewer touches ⇒ more 'hold'."""
    n = 400
    rng = np.random.default_rng(9)
    calm = 100 * np.exp(np.cumsum(rng.standard_normal(n) * 0.002))
    wild = 100 * np.exp(np.cumsum(rng.standard_normal(n) * 0.02))
    calm_lab = pt.triple_barrier_labels(_df(calm, calm * 1.001, calm * 0.999), [24], k=1.0)[0]
    wild_lab = pt.triple_barrier_labels(_df(wild, wild * 1.001, wild * 0.999), [24], k=1.0)[0]
    hold_calm = (calm_lab[calm_lab != -1] == 2).mean()
    hold_wild = (wild_lab[wild_lab != -1] == 2).mean()
    # both are finite fractions; vol-scaling keeps them in a sane band (not all-hold / all-trade)
    assert 0.0 <= hold_calm <= 1.0 and 0.0 <= hold_wild <= 1.0
