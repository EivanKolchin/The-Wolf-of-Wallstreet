import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from backend.agents.nn_model import PersistentTradingModel, TradeExperience
from backend.signals import feature_spec as fs
from backend.core.config import settings


def _configure_paths(tmp_path):
    PersistentTradingModel.MODEL_PATH = tmp_path / "trading_lstm_latest.pt"
    PersistentTradingModel.CHECKPOINT_DIR = tmp_path / "checkpoints"


def test_save_and_load_roundtrip(tmp_path):
    _configure_paths(tmp_path)
    model = PersistentTradingModel()
    model.trade_count = 7
    model.cumulative_pnl = 0.42
    model.safe_checkpoint(label="roundtrip")

    reloaded = PersistentTradingModel()
    assert reloaded.trade_count == 7
    assert reloaded.cumulative_pnl == pytest.approx(0.42)


def test_trade_count_and_pnl_persist_after_update(tmp_path):
    _configure_paths(tmp_path)
    model = PersistentTradingModel()
    experience = TradeExperience(
        features_sequence=np.zeros((model.SEQUENCE_LENGTH, fs.INPUT), dtype=np.float32),
        direction_taken=0,
        actual_pnl_pct=0.01,
    )
    model.online_update(experience)
    model.safe_checkpoint(label="persist")

    restored = PersistentTradingModel()
    assert restored.trade_count >= 1
    assert restored.cumulative_pnl > 0


def test_incompatible_checkpoint_falls_back_safely(tmp_path):
    _configure_paths(tmp_path)
    # A checkpoint with no FeatureSpec metadata + junk weights must NOT corrupt the model.
    bad_state = {"model_state_dict": {"bad.weight": torch.randn(3, 3)}}
    torch.save(bad_state, PersistentTradingModel.MODEL_PATH)

    model = PersistentTradingModel()  # should detect mismatch and cold-start
    sample = np.zeros((model.SEQUENCE_LENGTH, fs.INPUT), dtype=np.float32)
    res = model.infer(sample, symbol_id=0)

    assert res.direction in {"long", "short", "hold"}
    assert 0.02 <= res.size <= 0.20
    assert set(res.probs.keys()) == {"long", "short", "hold"}


def test_online_update_changes_weights(tmp_path):
    _configure_paths(tmp_path)
    model = PersistentTradingModel()
    initial = {
        k: v.detach().clone()
        for k, v in model.model.state_dict().items()
        if v.dtype.is_floating_point
    }

    for i in range(40):  # AWR update fires once the replay buffer reaches 32
        experience = TradeExperience(
            features_sequence=np.random.normal(0, 1, (model.SEQUENCE_LENGTH, fs.INPUT)).astype(np.float32),
            direction_taken=i % 3,
            actual_pnl_pct=0.02 if i % 2 == 0 else -0.015,
            size_taken=0.1, sl_taken=0.02, tp_taken=0.04,
        )
        model.online_update(experience)

    updated = model.model.state_dict()
    assert any(not torch.allclose(initial[k], updated[k]) for k in initial.keys())


def test_infer_respects_confidence_gate(tmp_path, monkeypatch):
    _configure_paths(tmp_path)
    model = PersistentTradingModel()

    monkeypatch.setattr(settings, "NN_HOLD_PROB_MULTIPLIER", 1.0)
    monkeypatch.setattr(settings, "NN_MIN_ACTION_CONFIDENCE", 0.99)

    sample = np.zeros((model.SEQUENCE_LENGTH, fs.INPUT), dtype=np.float32)
    res = model.infer(sample, symbol_id=0)
    assert res.direction == "hold"
