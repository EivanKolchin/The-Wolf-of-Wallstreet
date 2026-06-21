"""A1/A2 tests: vectorised MC-dropout + the combined infer_with_distribution.

The batched MC must be a drop-in for the old K-iteration loop (same shapes,
non-negative spread, valid decision) and the combined call must return both a
usable InferenceResult and a (K,H) edge-sample matrix from one MC pass.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from signals import feature_spec as fs  # noqa: E402


def _model(tmp_path, monkeypatch):
    from agents.nn_model import PersistentTradingModel
    monkeypatch.setattr(PersistentTradingModel, "MODEL_PATH", tmp_path / "m.pt")
    monkeypatch.setattr(PersistentTradingModel, "CHECKPOINT_DIR", tmp_path / "ckpts")
    return PersistentTradingModel()


def test_batched_mc_shapes_and_decision(tmp_path, monkeypatch):
    from agents.nn_model import HORIZONS
    pm = _model(tmp_path, monkeypatch)
    seq = np.random.default_rng(0).standard_normal((pm.SEQUENCE_LENGTH, fs.INPUT)).astype(np.float32)

    res = pm.infer(seq, symbol_id=0, mc_samples=16)
    assert res.direction in ("long", "short", "hold")
    assert res.edge_std >= 0.0                      # MC produced a spread (or 0)

    dist = pm.infer_predictive_distribution(seq, symbol_id=0, mc_samples=16)
    assert dist["edge_samples"].shape == (16, len(HORIZONS))


def test_infer_with_distribution_matches_decision(tmp_path, monkeypatch):
    from agents.nn_model import HORIZONS
    pm = _model(tmp_path, monkeypatch)
    seq = np.random.default_rng(1).standard_normal((pm.SEQUENCE_LENGTH, fs.INPUT)).astype(np.float32)

    # Deterministic decision (mc=1) must equal the combined call's decision,
    # since both use the same no-dropout forward for the action choice.
    det = pm.infer(seq, symbol_id=3, mc_samples=1)
    res, dist = pm.infer_with_distribution(seq, symbol_id=3, mc_samples=16)

    assert res.direction == det.direction
    assert abs(res.size - det.size) < 1e-6
    assert dist["edge_samples"].shape == (16, len(HORIZONS))
    # edge stats on the combined result are derived from the MC samples at the
    # horizon actually traded (primary_horizon_idx — default H+12, NOT the H+3 noise head).
    e0 = dist["edge_samples"][:, pm.primary_horizon_idx]
    assert abs(res.edge_mean - float(np.mean(e0))) < 1e-6
    assert abs(res.edge_std - float(np.std(e0))) < 1e-6


def test_mc_samples_one_is_zero_spread(tmp_path, monkeypatch):
    pm = _model(tmp_path, monkeypatch)
    seq = np.zeros((pm.SEQUENCE_LENGTH, fs.INPUT), dtype=np.float32)
    res = pm.infer(seq, symbol_id=0, mc_samples=1)
    assert res.edge_std == 0.0                      # no MC requested -> deterministic
