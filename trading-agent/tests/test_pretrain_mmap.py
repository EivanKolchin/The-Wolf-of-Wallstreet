"""Test the float16 memmap streaming dataset (low-RAM / low-disk Colab path).

After the disk refactor, `_SeqDataset` is fork-safe: it accepts EITHER an in-RAM
array OR a .npy PATH (opened lazily per worker), supports a `lo` offset so one
per-symbol file feeds both the train and val split, and yields a 5-tuple that
includes the recency weight.
"""
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


def _make_file(tmp_path, M):
    """A float16 .npy where row r is filled with the value r (so we can assert the
    dataset reads the correct row, including through a `lo` offset)."""
    T, F = pt.SEQ_LEN, pt.INPUT_SIZE
    xp = tmp_path / "X.npy"
    Xmm = np.lib.format.open_memmap(str(xp), mode="w+", dtype=np.float16, shape=(M, T, F))
    for r in range(M):
        Xmm[r] = np.float16(r)
    Xmm.flush()
    return str(xp), T, F


def test_seqdataset_array_input_yields_float32_5tuple(tmp_path):
    M = 12
    xp, T, F = _make_file(tmp_path, M)
    Xr = np.load(xp, mmap_mode="r")
    assert Xr.dtype == np.float16
    y = np.zeros((M, 3), np.int64)
    R = np.zeros((M, 3), np.float32)
    s = np.zeros(M, np.int64)

    ds = pt._SeqDataset(Xr, y, R, s)              # w defaults to ones
    assert len(ds) == M
    xi, yi, ri, si, wi = ds[0]                    # 5-tuple now
    assert xi.dtype == torch.float32 and tuple(xi.shape) == (T, F)
    assert float(wi) == 1.0

    bx, by, br, bs, bw = next(iter(DataLoader(ds, batch_size=4)))
    assert bx.shape == (4, T, F) and bx.dtype == torch.float32
    assert bw.shape == (4,)


def test_seqdataset_path_input_is_lazy_and_offset_correct(tmp_path):
    M, cut = 10, 7
    xp, T, F = _make_file(tmp_path, M)
    n_val = M - cut
    y = np.zeros((n_val, 3), np.int64)
    R = np.zeros((n_val, 3), np.float32)
    s = np.zeros(n_val, np.int64)
    w = np.full(n_val, 0.5, np.float32)

    # PATH input + lo=cut → item i must read file row (cut + i).
    ds = pt._SeqDataset(xp, y, R, s, w=w, lo=cut)
    assert ds._X is None                          # not opened until first access
    xi, yi, ri, si, wi = ds[0]
    assert ds._X is not None                       # opened lazily (fork-safe)
    assert float(xi.mean()) == float(cut)          # row `cut`
    assert float(wi) == 0.5
    xi2, *_ = ds[2]
    assert float(xi2.mean()) == float(cut + 2)
    assert len(ds) == n_val
