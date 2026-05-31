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
def test_train_epoch_with_pnl_weights_changes_gradient_vs_unweighted():
    """The 4-tuple batch path (with future_returns) must apply per-sample weights
    so gradients differ from the 3-tuple legacy path on the same data."""
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
    # Construct future_returns where most samples are chop (~0.01%) and a few
    # are large (~5%) so the weight ratio is sharp.
    future_returns = np.full((B, len(HORIZONS)), 0.0001, dtype=np.float32)
    future_returns[:3, :] = 0.05    # big moves on first 3 samples

    device = torch.device("cpu")

    def _grad_norm(model):
        return float(sum((p.grad.detach().norm()**2).item() for p in model.parameters() if p.grad is not None) ** 0.5)

    # --- unweighted (legacy 3-tuple) ---
    torch.manual_seed(0)
    model_a = ImprovedTradingLSTM().to(device)
    loss_fns_a = make_weighted_loss(y, per_sample=False)
    opt_a = torch.optim.SGD(model_a.parameters(), lr=1e-3)
    from torch.utils.data import TensorDataset, DataLoader
    ds_a = TensorDataset(torch.from_numpy(X), torch.from_numpy(y), torch.from_numpy(sym_ids))
    loader_a = DataLoader(ds_a, batch_size=B)
    train_epoch(model_a, loader_a, opt_a, loss_fns_a, device)
    gn_a = _grad_norm(model_a)

    # --- weighted (4-tuple) ---
    torch.manual_seed(0)
    model_b = ImprovedTradingLSTM().to(device)
    loss_fns_b = make_weighted_loss(y, per_sample=True)
    opt_b = torch.optim.SGD(model_b.parameters(), lr=1e-3)
    ds_b = TensorDataset(
        torch.from_numpy(X), torch.from_numpy(y),
        torch.from_numpy(future_returns), torch.from_numpy(sym_ids),
    )
    loader_b = DataLoader(ds_b, batch_size=B)
    train_epoch(model_b, loader_b, opt_b, loss_fns_b, device)
    gn_b = _grad_norm(model_b)

    # Different gradient magnitude → the weighting actually affected backprop.
    # Allow either direction (weighted can be larger or smaller depending on chop mix)
    # but they must be meaningfully different.
    assert abs(gn_a - gn_b) > 1e-6, f"weighted grad {gn_b} ≈ unweighted {gn_a} — weighting had no effect"


def test_legacy_3_tuple_batch_still_works():
    """Backward compat: a loader that yields 3-tuples (no future_returns) must
    still train without error."""
    from scripts.pretrain import (
        make_weighted_loss, train_epoch, HORIZONS, NUM_CLASSES, INPUT_SIZE, SEQ_LEN,
    )
    from agents.improved_model import ImprovedTradingLSTM
    from torch.utils.data import TensorDataset, DataLoader

    rng = np.random.default_rng(1)
    B = 8
    X = rng.standard_normal((B, SEQ_LEN, INPUT_SIZE)).astype(np.float32)
    y = rng.integers(0, NUM_CLASSES, size=(B, len(HORIZONS))).astype(np.int64)
    sym = np.zeros(B, dtype=np.int64)
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y), torch.from_numpy(sym))
    loader = DataLoader(ds, batch_size=B)

    model = ImprovedTradingLSTM()
    loss_fns = make_weighted_loss(y, per_sample=True)  # 'none' reduction, but no weights -> falls back to mean
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)
    out = train_epoch(model, loader, opt, loss_fns, torch.device("cpu"))
    assert np.isfinite(out)
