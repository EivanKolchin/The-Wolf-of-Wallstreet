"""Phase 2 hybrid scaffolding tests (offline, synthetic): the target, the pinball loss,
the QuantileTCN (shapes, non-crossing quantiles, a train step that reduces loss), the
LightGBM quantile co-model, and the stacker blend. These prove the components are wired
correctly before any real training run."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from backend.models.targets import vol_normalized_forward_return  # noqa: E402
from backend.models.losses import pinball_loss  # noqa: E402
from backend.models.sequence import QuantileTCN  # noqa: E402
from backend.models.ensemble import QuantileStacker  # noqa: E402
from backend.models import HAS_LGB  # noqa: E402
from signals import feature_spec as fs  # noqa: E402

Q = (0.1, 0.5, 0.9)


# ---------------------------------------------------------------- target
def test_target_sign_and_tail_mask():
    close = np.linspace(100, 120, 300)            # steadily rising → positive fwd return
    t = vol_normalized_forward_return(close, h=12, vol_window=20)
    assert t.shape == (300,)
    assert np.isnan(t[-12:]).all()                # last h have no forward window
    assert np.nanmean(t[:-12]) > 0                # rising series → positive vol-norm return


def test_target_flat_is_near_zero():
    close = np.full(200, 100.0) + np.random.default_rng(0).standard_normal(200) * 1e-6
    t = vol_normalized_forward_return(close, h=12)
    assert abs(np.nanmean(t)) < 1.0               # no drift → centred near zero


# ---------------------------------------------------------------- pinball loss
def test_pinball_median_equals_half_mae():
    preds = torch.zeros(100, 1)
    target = torch.linspace(-1, 1, 100)
    loss = pinball_loss(preds, target, (0.5,))
    assert torch.allclose(loss, 0.5 * target.abs().mean(), atol=1e-6)


def test_pinball_asymmetric_for_high_quantile():
    # q=0.9 penalises UNDER-prediction (target>pred) ~9x more than over-prediction.
    under = pinball_loss(torch.zeros(1, 1), torch.tensor([1.0]), (0.9,))
    over = pinball_loss(torch.zeros(1, 1), torch.tensor([-1.0]), (0.9,))
    assert under > over * 5


# ---------------------------------------------------------------- QuantileTCN
def _batch(n=16, seq=60, f=None):
    f = f or fs.INPUT
    x = torch.randn(n, seq, f)
    sid = torch.randint(0, 18, (n,))
    return x, sid


def test_qtcn_shapes_and_monotonic_quantiles():
    model = QuantileTCN(input_size=fs.INPUT, num_horizons=3, quantiles=Q)
    x, sid = _batch()
    outs = model(x, sid)
    assert len(outs) == 3
    for o in outs:
        assert o.shape == (16, len(Q))
        # quantiles must be non-decreasing across the Q dim (no crossing)
        assert torch.all(o[:, 1:] >= o[:, :-1] - 1e-6)


def test_qtcn_regime_conditioning_shape():
    model = QuantileTCN(input_size=fs.INPUT, num_horizons=2, quantiles=Q, regime_dim=6)
    x, sid = _batch()
    regime = torch.zeros(16, 6); regime[:, 0] = 1.0
    outs = model(x, sid, regime=regime)
    assert len(outs) == 2 and outs[0].shape == (16, 3)


def test_qtcn_train_step_reduces_loss():
    torch.manual_seed(0)
    model = QuantileTCN(input_size=fs.INPUT, num_horizons=1, quantiles=Q)
    x, sid = _batch(n=64)
    target = (x[:, -1, 0] * 2.0).detach()         # learnable signal from a feature
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    l0 = pinball_loss(model(x, sid)[0], target, Q).item()
    for _ in range(40):
        opt.zero_grad()
        loss = pinball_loss(model(x, sid)[0], target, Q)
        loss.backward(); opt.step()
    assert loss.item() < l0                        # optimisation actually descends


def test_qtcn_edge_and_uncertainty():
    model = QuantileTCN(input_size=fs.INPUT, num_horizons=1, quantiles=Q)
    x, sid = _batch()
    edge, unc = model.edge_and_uncertainty(x, sid, horizon_idx=0)
    assert edge.shape == (16,) and unc.shape == (16,)
    assert torch.all(unc >= -1e-6)                 # p90 - p10 ≥ 0


# ---------------------------------------------------------------- GBM
@pytest.mark.skipif(not HAS_LGB, reason="lightgbm not installed")
def test_qgbm_fit_predict_tracks_signal():
    from backend.models.gbm import QuantileGBM
    rng = np.random.default_rng(1)
    X = rng.standard_normal((2000, 8))
    y = X[:, 0] * 1.5 + 0.3 * rng.standard_normal(2000)   # feature 0 carries the signal
    gbm = QuantileGBM(quantiles=Q, n_estimators=80).fit(X, y)
    p = gbm.predict(X)
    assert p.shape == (2000, 3)
    assert np.all(p[:, 1:] >= p[:, :-1] - 1e-9)            # sorted → non-crossing
    med = p[:, 1]
    assert np.corrcoef(med, y)[0, 1] > 0.6                 # median tracks the target
    assert int(np.argmax(gbm.feature_importance())) == 0   # finds the informative feature


@pytest.mark.skipif(not HAS_LGB, reason="lightgbm not installed")
def test_qgbm_save_load_roundtrip(tmp_path):
    from backend.models.gbm import QuantileGBM
    rng = np.random.default_rng(3)
    X = rng.standard_normal((600, 5)); y = X[:, 1] - 0.5 * X[:, 2]
    gbm = QuantileGBM(quantiles=Q, n_estimators=40).fit(X, y)
    p0 = gbm.predict(X)
    path = tmp_path / "qgbm.json"
    gbm.save(str(path))
    p1 = QuantileGBM.load(str(path)).predict(X)
    np.testing.assert_allclose(p0, p1, rtol=0, atol=1e-9)   # native string round-trip is exact


# ---------------------------------------------------------------- stacker
def test_stacker_blends_and_learns_weights():
    rng = np.random.default_rng(2)
    n = 500
    y = rng.standard_normal(n)
    good = np.column_stack([y - 0.05, y, y + 0.05])        # model A: accurate quantiles
    bad = np.column_stack([np.zeros(n) - 0.05, np.zeros(n), np.zeros(n) + 0.05])  # model B: useless
    st = QuantileStacker(n_models=2, quantiles=Q).fit([good, bad], y)
    assert st.weights[0] > st.weights[1]                   # learns to trust the good model
    blended = st.predict([good, bad])
    assert blended.shape == (n, 3)
    assert np.all(blended[:, 1:] >= blended[:, :-1] - 1e-9)
