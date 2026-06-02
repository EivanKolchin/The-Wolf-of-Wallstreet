"""Phase 16 tests — PnL-magnitude-weighted CE loss in the offline trainer.

The old loss treated +0.1% and +5% returns equivalently — gradient signal
dominated by chop. v2.1+ weights each sample's CE by
``clip(|future_return|/median_abs, 0.25, 4.0)`` so big moves drive learning
and chop is dampened.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "scripts"))


# ---------------------------------------------------------------- helpers
def test_pnl_magnitude_weight_clips_and_scales():
    from scripts.pretrain import pnl_magnitude_weight

    rets = np.array([0.001, 0.002, 0.01, 0.05, -0.001], dtype=np.float32)
    w = pnl_magnitude_weight(rets)
    assert w.shape == rets.shape
    assert w.dtype == np.float32
    # All inside [0.25, 4.0]
    assert (w >= 0.25 - 1e-6).all() and (w <= 4.0 + 1e-6).all()
    # Big return gets bigger weight than small return
    big_idx = int(np.argmax(np.abs(rets)))
    small_idx = int(np.argmin(np.abs(rets)))
    assert w[big_idx] > w[small_idx]


def test_pnl_magnitude_weight_handles_nan_and_zero():
    from scripts.pretrain import pnl_magnitude_weight

    rets = np.array([np.nan, 0.0, 0.005, np.inf], dtype=np.float32)
    w = pnl_magnitude_weight(rets)
    assert np.all(np.isfinite(w))
    assert (w >= 0.25 - 1e-6).all() and (w <= 4.0 + 1e-6).all()


def test_build_label_returns_matches_label_signs():
    """Returned magnitudes must have the same sign as the labels' direction."""
    import pandas as pd
    from scripts.pretrain import build_label_returns, build_labels

    close = np.array([100.0, 101.0, 102.5, 102.0, 99.0, 100.5, 100.0], dtype=np.float64)
    df = pd.DataFrame({"close": close})
    horizons = [1, 3]
    thresholds = [0.0, 0.0]

    labels = build_labels(df, horizons, thresholds)
    rets = build_label_returns(df, horizons)
    assert rets.shape == labels.shape
    # For unmasked rows: label == 0 → ret > 0; label == 1 → ret < 0; label == 2 → ret ≈ 0
    for hi in range(len(horizons)):
        for i in range(len(close) - horizons[hi]):
            if labels[i, hi] == 0:
                assert rets[i, hi] > 0
            elif labels[i, hi] == 1:
                assert rets[i, hi] < 0


# ---------------------------------------------------------------- end-to-end weighted loss
def _five_tuple_loader(X, y, R, s, w, B):
    from torch.utils.data import TensorDataset, DataLoader
    ds = TensorDataset(
        torch.from_numpy(X), torch.from_numpy(y), torch.from_numpy(R),
        torch.from_numpy(s), torch.from_numpy(w),
    )
    return DataLoader(ds, batch_size=B)


def test_train_epoch_pnl_weights_change_gradient_vs_uniform():
    """The PnL-magnitude weight must affect backprop: sharp future-return spread
    (few big moves) yields different gradients than a uniform spread. Recency
    weight is held at 1.0 so this isolates the PnL term in the new 5-tuple path."""
    from scripts.pretrain import (
        make_weighted_loss, train_epoch,
        HORIZONS, NUM_CLASSES, INPUT_SIZE, SEQ_LEN,
    )
    from agents.improved_model import ImprovedTradingLSTM

    rng = np.random.default_rng(0)
    B = 16
    X = rng.standard_normal((B, SEQ_LEN, INPUT_SIZE)).astype(np.float32)
    y = rng.integers(0, NUM_CLASSES, size=(B, len(HORIZONS))).astype(np.int64)
    sym_ids = np.zeros(B, dtype=np.int64)
    w_rec = np.ones(B, dtype=np.float32)                       # isolate the PnL term

    R_uniform = np.full((B, len(HORIZONS)), 0.01, dtype=np.float32)   # all equal → w_pnl≈1
    R_sharp   = np.full((B, len(HORIZONS)), 0.0001, dtype=np.float32)
    R_sharp[:3, :] = 0.05                                             # few big moves

    device = torch.device("cpu")

    def _grad_norm(model):
        return float(sum((p.grad.detach().norm()**2).item() for p in model.parameters() if p.grad is not None) ** 0.5)

    torch.manual_seed(0)
    model_a = ImprovedTradingLSTM().to(device)
    loss_a = make_weighted_loss(y, per_sample=True)
    opt_a = torch.optim.SGD(model_a.parameters(), lr=1e-3)
    train_epoch(model_a, _five_tuple_loader(X, y, R_uniform, sym_ids, w_rec, B),
                opt_a, loss_a, device, scaler=None, use_amp=False)
    gn_a = _grad_norm(model_a)

    torch.manual_seed(0)
    model_b = ImprovedTradingLSTM().to(device)
    loss_b = make_weighted_loss(y, per_sample=True)
    opt_b = torch.optim.SGD(model_b.parameters(), lr=1e-3)
    train_epoch(model_b, _five_tuple_loader(X, y, R_sharp, sym_ids, w_rec, B),
                opt_b, loss_b, device, scaler=None, use_amp=False)
    gn_b = _grad_norm(model_b)

    assert abs(gn_a - gn_b) > 1e-6, f"sharp grad {gn_b} ≈ uniform {gn_a} — PnL weighting had no effect"


def test_recency_weight_changes_gradient():
    """Recency weighting must also reach backprop: down-weighting old samples
    (w<1) changes the gradient vs all-ones, on identical data."""
    from scripts.pretrain import (
        make_weighted_loss, train_epoch, HORIZONS, NUM_CLASSES, INPUT_SIZE, SEQ_LEN,
    )
    from agents.improved_model import ImprovedTradingLSTM

    rng = np.random.default_rng(7)
    B = 12
    X = rng.standard_normal((B, SEQ_LEN, INPUT_SIZE)).astype(np.float32)
    y = rng.integers(0, NUM_CLASSES, size=(B, len(HORIZONS))).astype(np.int64)
    R = np.full((B, len(HORIZONS)), 0.01, dtype=np.float32)    # uniform PnL term
    s = np.zeros(B, dtype=np.int64)
    device = torch.device("cpu")

    def _gn(m):
        return float(sum((p.grad.detach().norm()**2).item() for p in m.parameters() if p.grad is not None) ** 0.5)

    torch.manual_seed(0)
    m1 = ImprovedTradingLSTM().to(device)
    train_epoch(m1, _five_tuple_loader(X, y, R, s, np.ones(B, np.float32), B),
                torch.optim.SGD(m1.parameters(), lr=1e-3), make_weighted_loss(y), device)
    g1 = _gn(m1)

    torch.manual_seed(0)
    m2 = ImprovedTradingLSTM().to(device)
    w_decayed = np.linspace(0.25, 1.0, B).astype(np.float32)   # older samples down-weighted
    train_epoch(m2, _five_tuple_loader(X, y, R, s, w_decayed, B),
                torch.optim.SGD(m2.parameters(), lr=1e-3), make_weighted_loss(y), device)
    g2 = _gn(m2)

    assert np.isfinite(g1) and np.isfinite(g2)
    assert abs(g1 - g2) > 1e-6, "recency weighting had no effect on gradients"
