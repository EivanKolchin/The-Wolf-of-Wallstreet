"""Cycle 8: switchable TCN temporal core. Strict causality, output-contract parity
with the LSTM, and config-driven selection."""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from backend.agents.improved_model import ImprovedTradingLSTM, TemporalConvNet  # noqa: E402
from backend.signals import feature_spec as fs  # noqa: E402
from backend.core import config  # noqa: E402


def _model(trunk, monkeypatch, hidden=32):
    monkeypatch.setattr(config.settings, "NN_TRUNK", trunk)
    return ImprovedTradingLSTM(input_size=fs.INPUT, hidden_size=hidden, num_layers=2,
                               symbol_embed_dim=8).eval()


def test_tcn_module_is_strictly_causal():
    torch.manual_seed(0)
    tcn = TemporalConvNet(8, 16).eval()           # dropout off in eval → deterministic
    x = torch.randn(1, 60, 8)
    y1 = tcn(x)
    x2 = x.clone()
    x2[:, 40:, :] += 7.0                           # perturb ONLY the future (t >= 40)
    y2 = tcn(x2)
    assert torch.allclose(y1[:, :40], y2[:, :40], atol=1e-5)      # past unchanged → causal
    assert not torch.allclose(y1[:, 40:], y2[:, 40:], atol=1e-5)  # future did change


def test_tcn_receptive_field_covers_lookback():
    # dilations 1,2,4,8,16 with kernel 3 → receptive field 1 + 2*(1+2+4+8+16) = 63 ≥ 60
    tcn = TemporalConvNet(4, 8)
    rf = 1 + 2 * sum(d for d in (1, 2, 4, 8, 16))
    assert rf >= 60 and len(tcn.blocks) == 5


def test_tcn_trunk_builds_and_matches_output_contract(monkeypatch):
    m = _model("tcn", monkeypatch)
    assert m.trunk_type == "tcn" and hasattr(m, "tcn") and not hasattr(m, "lstm")
    x = torch.randn(4, 60, fs.INPUT)
    sids = torch.zeros(4, dtype=torch.long)
    logits, probs, size, exits, attn = m(x, sids)
    assert len(logits) == m.num_horizons and logits[0].shape == (4, 3)
    assert torch.allclose(probs[0].sum(-1), torch.ones(4), atol=1e-4)
    assert size.shape == (4, 1)
    for k in ("sl", "tp", "trail", "sl_mult", "tp_mult", "trail_mult", "next_candle_logret"):
        assert k in exits
    assert attn.shape == (4, 60)


def test_lstm_and_tcn_have_identical_io_shapes(monkeypatch):
    x = torch.randn(2, 60, fs.INPUT)
    sids = torch.zeros(2, dtype=torch.long)
    lstm_out = _model("lstm", monkeypatch)(x, sids)
    tcn_out = _model("tcn", monkeypatch)(x, sids)
    assert [t.shape for t in lstm_out[0]] == [t.shape for t in tcn_out[0]]   # logits
    assert lstm_out[2].shape == tcn_out[2].shape                             # size
    assert lstm_out[3]["next_candle_logret"].shape == tcn_out[3]["next_candle_logret"].shape
    assert lstm_out[4].shape == tcn_out[4].shape                             # attention


def test_default_trunk_is_lstm(monkeypatch):
    monkeypatch.setattr(config.settings, "NN_TRUNK", "lstm")
    m = ImprovedTradingLSTM(input_size=fs.INPUT, hidden_size=16)
    assert m.trunk_type == "lstm" and hasattr(m, "lstm") and not hasattr(m, "tcn")
