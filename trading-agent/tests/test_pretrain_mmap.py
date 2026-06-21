"""On-the-fly windowing dataset (low-RAM / low-disk path). `_SeqDataset` slices each
(seq_len, F) window from the stored feature MATRIX at read time, instead of reading
a pre-expanded window — so the dataset is ~seq_len× smaller and fits in RAM."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

_spec = importlib.util.spec_from_file_location("pretrain_mmap_mod", str(ROOT / "scripts" / "pretrain.py"))
pt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pt)


def _meta(n_ends):
    H = len(pt.HORIZONS)
    return (np.zeros((n_ends, H), np.int64), np.zeros((n_ends, H), np.float32),
            np.zeros(n_ends, np.int64), np.ones(n_ends, np.float32))


def test_window_built_on_the_fly_from_feature_path(tmp_path):
    N, F, T = 200, pt.INPUT_SIZE, pt.SEQ_LEN
    feats = np.arange(N * F, dtype=np.float32).reshape(N, F)
    fp = tmp_path / "_F.npy"
    np.save(fp, feats.astype(np.float16))
    ends = np.arange(T, N)                         # every valid endpoint
    y, R, s, w = _meta(len(ends))

    ds = pt._SeqDataset(str(fp), ends, y, R, s, w)
    assert len(ds) == len(ends)
    xi, yi, ri, si, wi = ds[0]
    assert xi.dtype == torch.float32 and tuple(xi.shape) == (T, F)
    # window for ends[0]=T is feats[0:T] (float16 round-trip)
    assert np.allclose(xi.numpy(), feats[0:T].astype(np.float16).astype(np.float32))
    e = int(ends[-1])
    xl = ds[len(ds) - 1][0].numpy()
    assert np.allclose(xl, feats[e - T:e].astype(np.float16).astype(np.float32))

    bx, by, br, bs, bw = next(iter(DataLoader(ds, batch_size=4)))
    assert bx.shape == (4, T, F) and bw.shape == (4,)


def test_feature_file_opened_lazily(tmp_path):
    N, F, T = 100, pt.INPUT_SIZE, pt.SEQ_LEN
    fp = tmp_path / "_F.npy"
    np.save(fp, np.random.default_rng(0).standard_normal((N, F)).astype(np.float16))
    ends = np.arange(T, N)
    ds = pt._SeqDataset(str(fp), ends, *_meta(len(ends)))
    assert ds._F is None        # not opened until first access (fork-safe)
    _ = ds[0]
    assert ds._F is not None
