"""The on-the-fly windows (feature matrix + end indices) must be byte-for-byte
equivalent to the reference in-RAM ``build_sequences`` — same masking, endpoints and
X/y/R alignment. That equivalence is what lets us store the small feature matrix
instead of the ~seq_len× larger expanded windows without changing the model."""
import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

_spec = importlib.util.spec_from_file_location("pretrain_seq_mod", str(ROOT / "scripts" / "pretrain.py"))
pt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pt)


def _valid_ends(labels, seq_len):
    ends = np.arange(seq_len, len(labels))
    keep = ~np.any(labels[ends - 1] == -1, axis=1)
    return ends[keep]


def test_on_the_fly_windows_match_build_sequences():
    rng = np.random.default_rng(0)
    N, F, H = 600, pt.INPUT_SIZE, len(pt.HORIZONS)
    feats = rng.standard_normal((N, F)).astype(np.float32)
    labels = rng.integers(0, pt.NUM_CLASSES, (N, H)).astype(np.int64)
    masked = rng.choice(N, N // 10, replace=False)
    labels[masked, rng.integers(0, H, masked.shape[0])] = -1
    returns = rng.standard_normal((N, H)).astype(np.float32)

    X_ref, y_ref, R_ref = pt.build_sequences(feats, labels, returns=returns)

    ends = _valid_ends(labels, pt.SEQ_LEN)
    y = labels[ends - 1]
    R = returns[ends - 1]
    ds = pt._SeqDataset(feats, ends, y, R, np.zeros(len(ends), np.int64))   # in-RAM float32

    assert len(ds) == len(X_ref)
    np.testing.assert_array_equal(y, y_ref)
    np.testing.assert_array_equal(R, R_ref)
    for k in (0, 1, 7, len(ds) // 2, len(ds) - 1):
        np.testing.assert_allclose(ds[k][0].numpy(), X_ref[k], rtol=0, atol=1e-6)


def test_empty_and_all_masked():
    H = len(pt.HORIZONS)
    labels = np.full((pt.SEQ_LEN + 5, H), -1, np.int64)     # everything masked
    assert len(_valid_ends(labels, pt.SEQ_LEN)) == 0
