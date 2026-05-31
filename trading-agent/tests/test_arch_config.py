"""A4 tests: config-driven architecture (GRU core, dropout) + regularization."""
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from signals import feature_spec as fs  # noqa: E402
from backend.core.config import settings  # noqa: E402


def test_default_core_is_lstm():
    from agents.improved_model import ImprovedTradingLSTM
    m = ImprovedTradingLSTM()
    assert isinstance(m.lstm, nn.LSTM)


def test_gru_core_builds_and_infers(monkeypatch):
    monkeypatch.setattr(settings, "NN_RNN_TYPE", "gru", raising=False)
    from agents.improved_model import ImprovedTradingLSTM
    m = ImprovedTradingLSTM()
    assert isinstance(m.lstm, nn.GRU)
    assert m.rnn_type == "gru"
    # forward works and emits the multi-horizon prob lists
    x = torch.randn(4, 60, fs.INPUT)
    sid = torch.zeros(4, dtype=torch.long)
    logits_list, probs_list, size, exits, attn = m(x, sid)
    assert len(probs_list) == m.num_horizons
    assert probs_list[0].shape == (4, 3)


def test_dropout_is_config_driven(monkeypatch):
    monkeypatch.setattr(settings, "NN_DROPOUT", 0.5, raising=False)
    from agents.improved_model import ImprovedTradingLSTM
    m = ImprovedTradingLSTM()
    assert abs(m.dropout.p - 0.5) < 1e-9


def test_label_smoothing_setting_exists():
    # The AWR path reads this; default should be a small positive smoothing.
    assert 0.0 <= float(settings.NN_LABEL_SMOOTHING) < 0.5
    assert float(settings.NN_WEIGHT_DECAY) > 0.0
