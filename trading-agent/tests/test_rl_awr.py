"""Phase 3 tests: Advantage-Weighted Regression, fee-net reward shaping, anti-hold."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from signals import feature_spec as fs  # noqa: E402


def test_reward_is_fee_net_and_holding_penalized():
    from agents.nn_model import TradeExperience
    z = np.zeros((60, fs.INPUT), dtype=np.float32)
    quick_win = TradeExperience(features_sequence=z, direction_taken=0, actual_pnl_pct=0.01, bars_held=1.0)
    slow_win = TradeExperience(features_sequence=z, direction_taken=0, actual_pnl_pct=0.01, bars_held=50.0)
    a_loss = TradeExperience(features_sequence=z, direction_taken=0, actual_pnl_pct=-0.01, bars_held=1.0)
    # holding far beyond the horizon eats into the reward
    assert quick_win.reward > slow_win.reward
    # a loss is worse than a win
    assert a_loss.reward < quick_win.reward


def test_sortino_downside_adjustment_dampens_reward():
    """A winning trade earns LESS reward when recent realized returns have been
    volatile/drawdown-prone (high downside deviation) than in calm conditions.
    With downside_dev == 0 the shaped reward must equal the legacy .reward."""
    from agents.nn_model import TradeExperience
    z = np.zeros((60, fs.INPUT), dtype=np.float32)
    win = TradeExperience(features_sequence=z, direction_taken=0, actual_pnl_pct=0.01, bars_held=1.0)

    calm = win.shaped_reward(0.0)
    volatile = win.shaped_reward(0.03)  # 3% downside deviation
    # downside_dev == 0 reproduces the legacy property exactly
    assert calm == win.reward
    # positive PnL reward is dampened (but stays positive) under downside risk
    assert 0.0 < volatile < calm


def test_downside_deviation_uses_only_shortfalls(tmp_path, monkeypatch):
    """_downside_deviation must ignore upside (Sortino, not Sharpe): a window of
    all-positive returns yields 0 deviation; losses produce a positive value."""
    from agents.nn_model import PersistentTradingModel
    monkeypatch.setattr(PersistentTradingModel, "MODEL_PATH", tmp_path / "m.pt")
    monkeypatch.setattr(PersistentTradingModel, "CHECKPOINT_DIR", tmp_path / "ckpts")
    pm = PersistentTradingModel()

    for _ in range(40):
        pm.recent_returns.append(0.02)  # all gains
    assert pm._downside_deviation() == 0.0

    for _ in range(40):
        pm.recent_returns.append(-0.02)  # now drawdowns dominate
    assert pm._downside_deviation() > 0.0


def test_idle_pressure_reduces_hold_prob(tmp_path, monkeypatch):
    from agents.nn_model import PersistentTradingModel
    monkeypatch.setattr(PersistentTradingModel, "MODEL_PATH", tmp_path / "m.pt")
    monkeypatch.setattr(PersistentTradingModel, "CHECKPOINT_DIR", tmp_path / "ckpts")
    pm = PersistentTradingModel()

    seq = np.random.default_rng(1).standard_normal((pm.SEQUENCE_LENGTH, fs.INPUT)).astype(np.float32)
    pm.set_idle_pressure(0.0)
    hold_idle0 = pm.infer(seq, symbol_id=0).probs["hold"]
    pm.set_idle_pressure(1.0)
    hold_idle1 = pm.infer(seq, symbol_id=0).probs["hold"]
    # under idle pressure the agent is less willing to hold
    assert hold_idle1 <= hold_idle0 + 1e-6


def test_awr_update_runs_and_trains_value_baseline(tmp_path, monkeypatch):
    from agents.nn_model import PersistentTradingModel, TradeExperience
    monkeypatch.setattr(PersistentTradingModel, "MODEL_PATH", tmp_path / "m.pt")
    monkeypatch.setattr(PersistentTradingModel, "CHECKPOINT_DIR", tmp_path / "ckpts")
    pm = PersistentTradingModel()

    before = [p.detach().clone() for p in pm.value_baseline.parameters()]
    rng = np.random.default_rng(0)
    for i in range(40):  # _awr_update fires once buffer >= 32 at trade_count % 10 == 0
        ex = TradeExperience(
            features_sequence=rng.standard_normal((pm.SEQUENCE_LENGTH, fs.INPUT)).astype(np.float32),
            direction_taken=int(i % 3),
            actual_pnl_pct=float(rng.standard_normal() * 0.01),
            symbol_id=0, size_taken=0.1, sl_taken=0.02, tp_taken=0.04, bars_held=float(i % 5),
        )
        pm.online_update(ex)

    after = list(pm.value_baseline.parameters())
    assert any(not torch.equal(b, a) for b, a in zip(before, after)), "value baseline should train"

    res = pm.infer(rng.standard_normal((pm.SEQUENCE_LENGTH, fs.INPUT)).astype(np.float32), symbol_id=0)
    assert res.direction in ("long", "short", "hold")
