"""Tier 1 / 1.5 / 2 model-quality changes:
  • 1a per-horizon training loss weights (down-weight the noisy H+3 head),
  • 1b trading-score checkpoint selection (compute_metrics returns a dict, _trading_score),
  • 1.5 live orderbook train/serve guard (zeroed at inference to match offline-zero training),
  • 2  tempered class weights (less over-trading).
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

_spec = importlib.util.spec_from_file_location("pretrain_tier_mod", str(ROOT / "scripts" / "pretrain.py"))
pt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pt)

from backend.signals import feature_spec as fs  # noqa: E402


# ───────────────────────── Tier 1a — per-horizon loss weights ─────────────────────────
def test_horizon_loss_weights_valid_and_downweight_h3():
    assert len(pt.HORIZON_LOSS_WEIGHTS) == len(pt.HORIZONS)
    # The 15-min head (index 0) is down-weighted vs the profitable longer horizons.
    assert pt.HORIZON_LOSS_WEIGHTS[0] < pt.HORIZON_LOSS_WEIGHTS[1]
    assert pt._HW_SUM == float(sum(pt.HORIZON_LOSS_WEIGHTS))


def test_select_horizon_in_range():
    assert 0 <= pt.SELECT_HORIZON_IDX < len(pt.HORIZONS)


# ───────────────────────── Tier 1b — trading-score selection ─────────────────────────
def test_trading_score_rewards_correct_direction():
    labels = np.array([0, 1, 0, 1])
    good   = np.array([0, 1, 0, 1])                 # all correct directional calls
    bad    = np.array([1, 0, 1, 0])                 # all wrong
    probs  = np.tile([0.6, 0.3, 0.1], (4, 1)).astype(np.float32)
    exp_g, sh_g = pt._trading_score(good, labels, probs)
    exp_b, sh_b = pt._trading_score(bad, labels, probs)
    assert sh_g > 0 > sh_b
    assert exp_g > exp_b


def test_trading_score_all_hold_is_neutral():
    labels = np.array([0, 1, 0, 1])
    hold   = np.array([2, 2, 2, 2])                 # never trades
    probs  = np.tile([0.3, 0.3, 0.4], (4, 1)).astype(np.float32)
    exp, sh = pt._trading_score(hold, labels, probs)
    assert exp == 0.0 and sh == 0.0


def test_compute_metrics_returns_dict():
    pytest.importorskip("sklearn")   # classification_report; installed on Colab/CI, not the slim local venv
    preds  = np.array([0, 1, 2, 0])
    labels = np.array([0, 1, 2, 1])
    probs  = np.tile([0.5, 0.3, 0.2], (4, 1)).astype(np.float32)
    m = pt.compute_metrics(preds, labels, probs, horizon=12)
    assert set(m) >= {"accuracy", "expectancy", "sharpe", "win_rate", "trade_pct"}
    assert all(isinstance(v, float) for v in m.values())


# ───────────────────────── Tier 2 — tempered class weights ─────────────────────────
def test_class_weight_power_tempers_toward_uniform():
    y = np.zeros((1000, len(pt.HORIZONS)), np.int64)
    y[200:400, :] = 1
    y[400:, :] = 2                                   # 20% long, 20% short, 60% hold
    orig = pt.CLASS_WEIGHT_POWER
    try:
        pt.CLASS_WEIGHT_POWER = 1.0
        w_full = pt.make_weighted_loss(y)[0].weight.numpy()
        pt.CLASS_WEIGHT_POWER = 0.5
        w_temp = pt.make_weighted_loss(y)[0].weight.numpy()
    finally:
        pt.CLASS_WEIGHT_POWER = orig
    # Tempering compresses the spread and lifts the majority 'hold' weight (less crushed).
    assert (w_temp.max() - w_temp.min()) < (w_full.max() - w_full.min())
    assert w_temp[2] > w_full[2]


# ───────────────────────── Tier 1.5 — live orderbook guard ─────────────────────────
def _live_model():
    from backend.agents.improved_model import ImprovedTradingLSTM
    torch.manual_seed(0)
    return ImprovedTradingLSTM(input_size=fs.INPUT, hidden_size=16, num_layers=2, symbol_embed_dim=8).eval()


def test_live_model_ignores_orderbook_when_zeroed():
    from backend.agents import improved_model as im
    im._ZERO_OB_CACHE = True                         # force the guard ON
    m = _live_model()
    x1 = torch.randn(1, 60, fs.INPUT)
    x2 = x1.clone()
    x2[..., fs.ORDERBOOK] += 5.0                      # perturb ONLY the orderbook block
    sid = torch.zeros(1, dtype=torch.long)
    with torch.no_grad():
        p1 = m(x1, sid)[1][0]
        p2 = m(x2, sid)[1][0]
    assert torch.allclose(p1, p2, atol=1e-6)          # zeroed → orderbook can't change the output


def test_live_model_uses_orderbook_when_guard_off():
    from backend.agents import improved_model as im
    im._ZERO_OB_CACHE = False                         # force the guard OFF
    try:
        m = _live_model()
        x1 = torch.randn(1, 60, fs.INPUT)
        x2 = x1.clone()
        x2[..., fs.ORDERBOOK] += 5.0
        sid = torch.zeros(1, dtype=torch.long)
        with torch.no_grad():
            p1 = m(x1, sid)[1][0]
            p2 = m(x2, sid)[1][0]
        assert not torch.allclose(p1, p2, atol=1e-6)  # guard off → orderbook does affect output
    finally:
        im._ZERO_OB_CACHE = True                       # restore default for other tests
