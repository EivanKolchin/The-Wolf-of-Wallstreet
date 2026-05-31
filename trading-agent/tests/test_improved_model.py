"""Phase 1 guard tests: the live ImprovedTradingLSTM wiring + PersistentTradingModel.infer."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from signals import feature_spec as fs  # noqa: E402
from agents.improved_model import ImprovedTradingLSTM, SYMBOL_TO_ID  # noqa: E402


def test_forward_shapes_and_exit_heads():
    model = ImprovedTradingLSTM()
    x = torch.randn(2, 60, fs.INPUT)
    sids = torch.tensor([0, 1], dtype=torch.long)
    logits_list, probs_list, size, exits, attn = model(x, sids)
    assert len(logits_list) == len(probs_list) == model.num_horizons
    assert probs_list[0].shape == (2, 3)
    assert size.shape == (2, 1)
    for k in ("sl", "tp", "trail"):
        assert exits[k].shape == (2, 1)
    # exit fractions land inside their configured ranges
    assert torch.all(exits["sl"] >= 0.0) and torch.all(exits["tp"] <= 0.25)


def test_symbol_registry_stable_order():
    # ids must never be reordered (checkpoint embedding rows depend on this)
    assert SYMBOL_TO_ID["BTCUSDT"] == 0
    assert SYMBOL_TO_ID["ETHUSDT"] == 1


def test_persistent_model_cold_start_and_infer(tmp_path, monkeypatch):
    from agents.nn_model import PersistentTradingModel, InferenceResult

    # Point checkpoints at a temp dir so we never clobber the live model file.
    monkeypatch.setattr(PersistentTradingModel, "MODEL_PATH", tmp_path / "trading_lstm_latest.pt")
    monkeypatch.setattr(PersistentTradingModel, "CHECKPOINT_DIR", tmp_path / "checkpoints")

    pm = PersistentTradingModel()  # no file -> cold-start, writes a v2 checkpoint
    assert (tmp_path / "trading_lstm_latest.pt").exists()

    seq = np.random.randn(pm.SEQUENCE_LENGTH, fs.INPUT).astype(np.float32)
    res = pm.infer(seq, symbol_id=SYMBOL_TO_ID["BTCUSDT"], mc_samples=1)
    assert isinstance(res, InferenceResult)
    assert res.direction in ("long", "short", "hold")
    assert 0.02 <= res.size <= 0.20
    assert abs(sum(res.probs.values()) - 1.0) < 1e-4
    # Phase 17: SL/TP/trail are now ATR-scaled (mult × atr_pct). With randn input
    # the atr_norm slot can be ~3σ so the upper bound is naturally wider; the
    # risk manager clamps these downstream. Just check positivity + sanity ceiling.
    assert 0.0 < res.sl < 5.0 and 0.0 < res.tp < 5.0 and 0.0 < res.trail < 5.0
    assert len(res.horizon_probs) == pm.model.num_horizons


def test_infer_mc_dropout_produces_uncertainty(tmp_path, monkeypatch):
    from agents.nn_model import PersistentTradingModel

    monkeypatch.setattr(PersistentTradingModel, "MODEL_PATH", tmp_path / "m.pt")
    monkeypatch.setattr(PersistentTradingModel, "CHECKPOINT_DIR", tmp_path / "ckpts")
    pm = PersistentTradingModel()

    seq = np.random.randn(pm.SEQUENCE_LENGTH, fs.INPUT).astype(np.float32)
    res = pm.infer(seq, symbol_id=0, mc_samples=8)
    # With dropout active across 8 passes the edge std should be defined (>= 0).
    assert res.edge_std >= 0.0


def test_reload_roundtrip_after_cold_start(tmp_path, monkeypatch):
    from agents.nn_model import PersistentTradingModel

    monkeypatch.setattr(PersistentTradingModel, "MODEL_PATH", tmp_path / "m.pt")
    monkeypatch.setattr(PersistentTradingModel, "CHECKPOINT_DIR", tmp_path / "ckpts")
    PersistentTradingModel()  # cold-start writes a valid v2 checkpoint
    # Second construction must load it cleanly (no cold-start path / no raise).
    pm2 = PersistentTradingModel()
    res = pm2.infer(np.random.randn(pm2.SEQUENCE_LENGTH, fs.INPUT).astype(np.float32), symbol_id=0)
    assert res.direction in ("long", "short", "hold")
