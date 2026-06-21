"""A4: the signal-audit harness. Tests the pure helpers (no data/sklearn needed) plus a
sklearn-guarded check that the baseline AUC detects a planted edge and reads ~0.5 on noise."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

import importlib.util

_spec = importlib.util.spec_from_file_location("sigaudit", str(ROOT / "scripts" / "signal_audit.py"))
sa = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sa)


def test_forward_returns_values_and_tail_mask():
    close = np.array([1, 2, 4, 8, 16], float)
    fr = sa.forward_returns(close, 2)
    assert np.isclose(fr[0], 3.0) and np.isclose(fr[1], 3.0) and np.isclose(fr[2], 3.0)
    assert np.isnan(fr[3]) and np.isnan(fr[4])          # last h bars have no future


def test_rank_ic_detects_signal_and_ignores_noise():
    rng = np.random.default_rng(0)
    fwd = rng.standard_normal(3000)
    informative = 0.6 * fwd + 0.4 * rng.standard_normal(3000)
    noise = rng.standard_normal(3000)
    assert sa.rank_ic(informative, fwd) > 0.3
    assert abs(sa.rank_ic(noise, fwd)) < 0.1


def test_rank_ic_constant_feature_is_zero():
    fwd = np.random.default_rng(1).standard_normal(500)
    assert sa.rank_ic(np.full(500, 0.5), fwd) == 0.0     # dead/constant feature → IC 0


def test_purged_split_embargo_gap():
    tr, te = sa.purged_split(1000, 0.2, embargo=48)
    assert tr.max() < te.min()
    assert te[0] - tr[-1] - 1 == 48                      # exactly `embargo` rows purged


def test_candidate_features_finite_and_causal():
    rng = np.random.default_rng(3)
    n = 2000
    close = 100 * np.cumprod(1 + rng.standard_normal(n) * 0.01)
    high, low, vol = close * 1.002, close * 0.998, rng.uniform(1, 5, n)
    cf = sa.candidate_features(high, low, close, vol)
    assert cf.shape[0] == n and cf.shape[1] >= 8
    assert np.all(np.isfinite(cf))                      # no NaN/inf may reach the model
    close2, high2 = close.copy(), high.copy()
    close2[-1] *= 1.3; high2[-1] *= 1.3                 # perturb only the last bar
    cf2 = sa.candidate_features(high2, low, close2, vol)
    assert np.allclose(cf[:-1], cf2[:-1])               # earlier rows unchanged → causal


def test_baseline_auc_detects_edge_and_noise():
    pytest.importorskip("sklearn")
    rng = np.random.default_rng(2)
    n = 4000
    y = (rng.standard_normal(n) > 0).astype(float)
    X = np.column_stack([y + 0.3 * rng.standard_normal(n), rng.standard_normal(n)])
    tr, te = sa.purged_split(n, 0.3, embargo=0)
    auc = sa._baseline_auc(X, y, tr, te)
    assert auc is not None and auc > 0.85               # planted edge is found

    yr = (rng.standard_normal(n) > 0).astype(float)
    Xr = rng.standard_normal((n, 2))
    aucr = sa._baseline_auc(Xr, yr, tr, te)
    assert aucr is None or 0.4 < aucr < 0.6             # pure noise ≈ 0.5
