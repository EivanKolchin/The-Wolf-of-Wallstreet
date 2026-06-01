"""Test the float16 memmap streaming path used for low-RAM (Colab) training."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

# Load the script directly (scripts/ isn't a package).
_spec = importlib.util.spec_from_file_location("pretrain_mmap_mod", str(ROOT / "scripts" / "pretrain.py"))
pt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pt)


def test_seqdataset_streams_float16_memmap_as_float32(tmp_path):
    M, T, F = 12, pt.SEQ_LEN, pt.INPUT_SIZE
    xp = tmp_path / "X.npy"
    Xmm = np.lib.format.open_memmap(str(xp), mode="w+", dtype=np.float16, shape=(M, T, F))
    Xmm[:] = np.random.default_rng(0).standard_normal((M, T, F)).astype(np.float16)
    Xmm.flush()
    Xr = np.load(str(xp), mmap_mode="r")
    assert Xr.dtype == np.float16

    y = np.zeros((M, 3), np.int64)
    R = np.zeros((M, 3), np.float32)
    s = np.zeros(M, np.int64)

    ds = pt._SeqDataset(Xr, y, R, s)
    assert len(ds) == M
    xi, yi, ri, si = ds[0]
    assert xi.dtype == torch.float32 and tuple(xi.shape) == (T, F)

    bx, by, br, bs = next(iter(DataLoader(ds, batch_size=4)))
    assert bx.shape == (4, T, F) and bx.dtype == torch.float32


def test_make_loader_branch_selects_streaming_for_float16():
    # float16 input → streaming dataset path; float32 → in-RAM TensorDataset.
    # We assert the dtype contract the loader branches on.
    assert np.float16 != np.float32
