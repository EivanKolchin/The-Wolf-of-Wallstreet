"""Recency weighting, per-symbol split loaders, and an end-to-end train/eval smoke
for the post-disk-fix pretraining pipeline."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

_spec = importlib.util.spec_from_file_location("pretrain_up_mod", str(ROOT / "scripts" / "pretrain.py"))
pt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pt)


# ───────────────────────────── recency weighting ─────────────────────────────
def test_recency_weights_newer_outweighs_older_with_floor():
    ref = pd.Timestamp("2026-06-01")
    ts = pd.to_datetime(["2026-06-01", "2025-06-01", "2024-06-01", "2010-01-01"]).values
    w = pt._recency_weights(ts, ref, halflife_years=2.0, floor=0.25)
    assert w.dtype == np.float32
    assert w[0] > w[1] > w[2]                      # newer weighs strictly more
    assert abs(w[0] - 1.0) < 1e-3                  # "today" ≈ 1.0
    assert abs(float(w[1]) - 0.5 ** 0.5) < 0.02    # 1y old ≈ 0.5^(1/halflife)
    assert float(w[3]) == 0.25                     # ancient sample clamped to floor
    assert w.min() >= 0.25 and w.max() <= 1.0


def test_recency_weights_future_timestamps_clamped_to_one():
    ref = pd.Timestamp("2026-06-01")
    ts = pd.to_datetime(["2026-12-31"]).values     # "newer than now"
    w = pt._recency_weights(ts, ref, halflife_years=2.0, floor=0.25)
    assert float(w[0]) == 1.0


# ───────────────────────────── per-symbol split ──────────────────────────────
def _array_build_out(n_per=40, seed=0, sids=(0, 1, 8)):
    rng = np.random.default_rng(seed)
    T, F, H = pt.SEQ_LEN, pt.INPUT_SIZE, len(pt.HORIZONS)
    Xs, ys, Rs, Ss, Ws = [], [], [], [], []
    for sid in sids:
        Xs.append(rng.standard_normal((n_per, T, F)).astype(np.float32))
        ys.append(rng.integers(0, pt.NUM_CLASSES, (n_per, H)).astype(np.int64))
        Rs.append(rng.standard_normal((n_per, H)).astype(np.float32))
        Ss.append(np.full(n_per, sid, np.int64))
        Ws.append(np.linspace(0.25, 1.0, n_per).astype(np.float32))
    return ("array", np.concatenate(Xs), np.concatenate(ys), np.concatenate(Rs),
            np.concatenate(Ss), np.concatenate(Ws))


def test_make_split_loaders_splits_each_symbol_not_global_tail():
    bo = _array_build_out(n_per=40, sids=(0, 1, 8))
    tr, va, y_tr, n_tr, n_va = pt.make_split_loaders(bo, batch_size=16, val_frac=0.25, embargo=0)
    # 3 symbols × 40 = 120; 25% of EACH → 30 val, 90 train.
    assert n_tr == 90 and n_va == 30
    assert len(y_tr) == 90

    batch = next(iter(tr))
    assert len(batch) == 5                          # X, y, future_returns, sym_id, recency
    bx, by, br, bs, bw = batch
    assert bx.shape[1:] == (pt.SEQ_LEN, pt.INPUT_SIZE)

    # The key fix: every symbol appears in BOTH splits (old global-tail split put
    # only the last symbol(s) in val).
    tr_syms = {int(s) for *_, ss, _ in [tuple(b) for b in tr] for s in ss}
    va_syms = {int(s) for *_, ss, _ in [tuple(b) for b in va] for s in ss}
    assert tr_syms == {0, 1, 8}
    assert va_syms == {0, 1, 8}


def test_make_split_loaders_parts_path_offsets_and_symbols(tmp_path):
    """The actual Colab path: per-symbol float16 memmaps (no assembled copy). Verify
    the val split reads the LATER rows of each file (lo offset) and both symbols are
    present in train + val. Datasets are indexed directly (no worker procs)."""
    T, F, H = pt.SEQ_LEN, pt.INPUT_SIZE, len(pt.HORIZONS)
    n = 20
    parts = []
    for sid in (0, 8):
        xp = tmp_path / f"_X_{sid}.npy"
        Xmm = np.lib.format.open_memmap(str(xp), mode="w+", dtype=np.float16, shape=(n, T, F))
        for r in range(n):
            Xmm[r] = np.float16(r)                  # row r holds the value r
        Xmm.flush()
        parts.append({"path": str(xp), "n": n,
                      "y": np.full((n, H), sid, np.int64),
                      "R": np.zeros((n, H), np.float32),
                      "s": np.full(n, sid, np.int64),
                      "w": np.ones(n, np.float32)})

    tr, va, y_tr, n_tr, n_va = pt.make_split_loaders(("parts", parts), batch_size=4, val_frac=0.25, embargo=0)
    assert n_tr == 30 and n_va == 10                # 2 symbols × 20; 25% val = 5 each
    assert len(y_tr) == 30

    # Index the underlying datasets directly (avoids DataLoader worker processes).
    val_ds = va.dataset
    val_vals = [float(val_ds[i][0].mean()) for i in range(len(val_ds))]
    assert min(val_vals) >= 15.0                    # val = rows >= cut(15): lo offset correct
    tr_ds = tr.dataset
    assert {int(tr_ds[i][3]) for i in range(len(tr_ds))} == {0, 8}   # both symbols in train
    assert {int(val_ds[i][3]) for i in range(len(val_ds))} == {0, 8} # … and val (no leak)
    assert len(tr_ds[0]) == 5                        # 5-tuple


def test_embargo_purges_train_boundary_rows():
    """Embargo removes `embargo` train rows per symbol at the val boundary (so the
    H+48 label window can't leak into val); the val set is untouched."""
    bo = _array_build_out(n_per=200, sids=(0, 1))
    _, _, _, n_tr_0, n_va_0 = pt.make_split_loaders(bo, batch_size=16, val_frac=0.2, embargo=0)
    _, _, _, n_tr_e, n_va_e = pt.make_split_loaders(bo, batch_size=16, val_frac=0.2, embargo=30)
    assert n_va_0 == n_va_e                          # validation unchanged
    assert n_tr_0 - n_tr_e == 2 * 30                 # 2 symbols × 30 purged rows


def test_default_embargo_is_max_horizon():
    bo = _array_build_out(n_per=300, sids=(0,))
    _, _, _, n_tr_def, _ = pt.make_split_loaders(bo, batch_size=16, val_frac=0.2)       # default embargo
    _, _, _, n_tr_exp, _ = pt.make_split_loaders(bo, batch_size=16, val_frac=0.2, embargo=max(pt.HORIZONS))
    assert n_tr_def == n_tr_exp


# ──────────────────────────── focal loss ─────────────────────────────────────
def test_focal_suppresses_easy_confident_samples(monkeypatch):
    """(1-p_true)^gamma must collapse the loss of easy, confident-correct samples,
    so learning concentrates on the hard ones."""
    logits = torch.tensor([[8.0, 0.0, 0.0]] * 3, dtype=torch.float32)   # all easy + correct
    targets = torch.tensor([0, 0, 0], dtype=torch.int64)
    ce = torch.nn.CrossEntropyLoss(reduction="none")
    w = torch.ones(3)

    monkeypatch.setattr(pt, "FOCAL_GAMMA", 0.0)
    plain = float(pt._apply_horizon_loss(ce, logits, targets, w))
    monkeypatch.setattr(pt, "FOCAL_GAMMA", 2.0)
    focal = float(pt._apply_horizon_loss(ce, logits, targets, w))
    assert focal < plain * 0.5 and plain > 0          # easy samples strongly suppressed


# ──────────────────────────── end-to-end smoke ───────────────────────────────
def test_train_and_eval_run_end_to_end_cpu():
    """One train + eval pass on a tiny model exercises the full 5-tuple + recency
    + loss path (AMP off on CPU) — guards against another mid-training halt."""
    bo = _array_build_out(n_per=24, seed=3)
    tr, va, y_tr, _, _ = pt.make_split_loaders(bo, batch_size=8, val_frac=0.25)
    device = torch.device("cpu")

    model = pt.ImprovedTradingLSTM(
        input_size=pt.INPUT_SIZE, hidden_size=16, num_layers=1, dropout=0.0,
        num_symbols=len(pt.SYMBOLS), symbol_embed_dim=4,
        num_horizons=len(pt.HORIZONS), num_classes=pt.NUM_CLASSES,
    ).to(device)
    loss_fns = pt.make_weighted_loss(y_tr)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    train_loss = pt.train_epoch(model, tr, opt, loss_fns, device, scaler=None, use_amp=False)
    val_loss, preds, labels, probs = pt.eval_epoch(model, va, loss_fns, device, use_amp=False)

    assert np.isfinite(train_loss) and np.isfinite(val_loss)
    assert len(preds) == len(pt.HORIZONS)
    assert preds[0].shape == labels[0].shape
    assert probs[0].shape[1] == pt.NUM_CLASSES
