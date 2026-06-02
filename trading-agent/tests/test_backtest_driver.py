"""Cycle 4: the backtest driver's model→signal→engine wiring (no data download —
tiny model + synthetic features), so the path that runs post-training is verified."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

_spec = importlib.util.spec_from_file_location("backtest_drv_mod", str(ROOT / "scripts" / "backtest.py"))
bt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bt)

from backend.backtest.engine import directional_signal, run_backtest  # noqa: E402


def test_probs_for_starts_and_backtest_wiring():
    device = torch.device("cpu")
    model = bt.pre.ImprovedTradingLSTM(
        input_size=bt.pre.INPUT_SIZE, hidden_size=8, num_layers=1, dropout=0.0,
        num_symbols=len(bt.pre.SYMBOLS), symbol_embed_dim=4,
        num_horizons=len(bt.pre.HORIZONS), num_classes=bt.pre.NUM_CLASSES,
    ).to(device).eval()

    feats = np.random.default_rng(0).standard_normal((300, bt.pre.INPUT_SIZE)).astype(np.float32)
    starts = np.arange(0, 300 - bt.pre.SEQ_LEN)
    probs = bt.probs_for_starts(model, feats, starts, sym_id=0, device=device, batch=64)

    assert probs.shape == (len(starts), bt.pre.NUM_CLASSES)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-4)     # valid probability rows (softmax)

    close = 100 * np.cumprod(1 + np.random.default_rng(1).standard_normal(len(starts)) * 0.001)
    sig = directional_signal(probs[:, 0], probs[:, 1], min_confidence=0.34, min_edge=0.0)
    res = run_backtest(close, sig, fee_bps=10, slippage_bps=5)
    assert res.equity.shape == (len(starts),)
    assert set(np.unique(res.position)).issubset({-1.0, 0.0, 1.0})
    for key in ("sharpe", "sortino", "max_drawdown", "total_return", "num_trades", "hit_rate"):
        assert key in res.metrics
