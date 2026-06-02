"""The --mmap streaming sequence builder must be byte-for-byte equivalent (within
float16 rounding) to the in-RAM build_sequences, with identical X/y/R alignment
and masking. This is what lets Colab build BTC (~9.6 GB float32) without OOM."""
import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

_spec = importlib.util.spec_from_file_location("pretrain_mmap_mod", str(ROOT / "scripts" / "pretrain.py"))
pt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pt)


def _synthetic(n=500, f=None, h=3, seed=1):
    f = f or pt.INPUT_SIZE
    rng = np.random.default_rng(seed)
    features = rng.standard_normal((n, f)).astype(np.float32)
    labels = rng.integers(0, pt.NUM_CLASSES, size=(n, h)).astype(np.int64)
    # Mask a scattered ~10% of rows (-1 in at least one horizon) to exercise the drop path.
    mask_rows = rng.choice(n, size=n // 10, replace=False)
    labels[mask_rows, rng.integers(0, h, size=mask_rows.shape[0])] = -1
    returns = rng.standard_normal((n, h)).astype(np.float32)
    return features, labels, returns


def test_stream_matches_build_sequences(tmp_path):
    features, labels, returns = _synthetic()
    X_ref, y_ref, R_ref = pt.build_sequences(features, labels, returns=returns)

    out = str(tmp_path / "_X_TEST.npy")
    y, R, n_seq = pt._build_sequences_to_file(features, labels, returns, out, chunk=64)

    # Same number of (masked-out) sequences, same labels/returns, same order.
    assert n_seq == len(X_ref)
    np.testing.assert_array_equal(y, y_ref)
    np.testing.assert_array_equal(R, R_ref)

    # X equal within float16 precision.
    X_disk = np.load(out, mmap_mode="r")
    assert X_disk.shape == X_ref.shape
    assert X_disk.dtype == np.float16
    np.testing.assert_allclose(np.asarray(X_disk, dtype=np.float32), X_ref, rtol=0, atol=1e-2)


def test_stream_handles_all_masked(tmp_path):
    features, labels, returns = _synthetic(n=200)
    labels[:] = -1  # every row masked -> zero sequences, must not crash
    out = str(tmp_path / "_X_EMPTY.npy")
    y, R, n_seq = pt._build_sequences_to_file(features, labels, returns, out)
    assert n_seq == 0 and len(y) == 0 and len(R) == 0


def test_stream_chunk_boundary(tmp_path):
    # n chosen so valid count is not a multiple of chunk -> exercises the tail slice.
    features, labels, returns = _synthetic(n=331, seed=7)
    X_ref, y_ref, _ = pt.build_sequences(features, labels, returns=returns)
    out = str(tmp_path / "_X_TAIL.npy")
    y, R, n_seq = pt._build_sequences_to_file(features, labels, returns, out, chunk=50)
    assert n_seq == len(X_ref)
    np.testing.assert_array_equal(y, y_ref)
    X_disk = np.load(out, mmap_mode="r")
    np.testing.assert_allclose(np.asarray(X_disk, dtype=np.float32), X_ref, rtol=0, atol=1e-2)
