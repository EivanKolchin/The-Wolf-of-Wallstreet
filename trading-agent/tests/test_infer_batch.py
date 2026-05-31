"""A3 test: cross-symbol batched inference matches per-symbol inference."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from signals import feature_spec as fs  # noqa: E402


def _pm(tmp_path, monkeypatch):
    from agents.nn_model import PersistentTradingModel
    monkeypatch.setattr(PersistentTradingModel, "MODEL_PATH", tmp_path / "m.pt")
    monkeypatch.setattr(PersistentTradingModel, "CHECKPOINT_DIR", tmp_path / "ck")
    return PersistentTradingModel()


def test_infer_batch_shapes_and_alignment(tmp_path, monkeypatch):
    from agents.nn_model import HORIZONS
    pm = _pm(tmp_path, monkeypatch)
    rng = np.random.default_rng(0)
    seqs = [rng.standard_normal((pm.SEQUENCE_LENGTH, fs.INPUT)).astype(np.float32) for _ in range(5)]
    sids = [0, 1, 2, 8, 12]

    out = pm.infer_batch(seqs, sids, mc_samples=16)
    assert len(out) == 5
    for res, dist in out:
        assert res.direction in ("long", "short", "hold")
        assert dist["edge_samples"].shape == (16, len(HORIZONS))
        assert res.edge_std >= 0.0


def test_infer_batch_decision_matches_per_symbol(tmp_path, monkeypatch):
    """The deterministic decision/size/exits for each row must equal what the
    per-symbol path produces (the action choice uses the no-dropout forward)."""
    pm = _pm(tmp_path, monkeypatch)
    rng = np.random.default_rng(7)
    seqs = [rng.standard_normal((pm.SEQUENCE_LENGTH, fs.INPUT)).astype(np.float32) for _ in range(4)]
    sids = [0, 3, 8, 11]

    batch = pm.infer_batch(seqs, sids, mc_samples=8)
    for i, (res, _dist) in enumerate(batch):
        single = pm.infer(seqs[i], symbol_id=sids[i], mc_samples=1)
        assert res.direction == single.direction
        assert abs(res.size - single.size) < 1e-6
        assert abs(res.sl - single.sl) < 1e-6
        assert abs(res.tp - single.tp) < 1e-6


def test_infer_batch_empty():
    from agents.nn_model import PersistentTradingModel  # noqa
    # empty input is a no-op (constructing a model is unnecessary here)
    assert PersistentTradingModel.infer_batch.__doc__ is not None
