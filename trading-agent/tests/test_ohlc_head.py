"""Phase 18 v2 tests — small OHLC-delta head on the shared trunk.

Until the offline trainer is rerun on next-K_FUTURE log-returns these weights
are randomly initialised, so we only verify:
  - the head exists with the right output shape,
  - param count is bounded (small head — must not bloat the trunk),
  - outputs are finite + bounded by tanh.
"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from signals import feature_spec as fs  # noqa: E402
from agents.improved_model import ImprovedTradingLSTM, K_FUTURE  # noqa: E402


def test_next_candle_head_shape_and_bounds():
    model = ImprovedTradingLSTM()
    x = torch.randn(4, 60, fs.INPUT)
    sids = torch.zeros(4, dtype=torch.long)
    _, _, _, exits, _ = model(x, sids)
    assert "next_candle_logret" in exits
    out = exits["next_candle_logret"]
    assert out.shape == (4, K_FUTURE, 4), f"expected (4, {K_FUTURE}, 4), got {tuple(out.shape)}"
    # Tanh-bounded.
    assert torch.all(out >= -1.0 - 1e-6) and torch.all(out <= 1.0 + 1e-6)
    assert torch.all(torch.isfinite(out))


def test_next_candle_head_is_small():
    """The head must be lightweight (≈ 2-3k params) so it doesn't bloat the trunk."""
    model = ImprovedTradingLSTM()
    n_params = sum(p.numel() for p in model.next_candle_head.parameters())
    # 64*32+32 (Linear1) + 32*(4*K_FUTURE) + (4*K_FUTURE) (Linear2)
    expected = (64 * 32 + 32) + (32 * 4 * K_FUTURE + 4 * K_FUTURE)
    assert n_params == expected, f"head param count drifted: {n_params} vs expected {expected}"
    assert n_params < 5000, f"head too large: {n_params} params"


def test_rolling_up_logret_to_ohlc_close_track_drifts_in_expected_direction():
    """Forward → reshape (K_FUTURE, 4) as (open, high, low, close) log-returns.
    Until trained, just verify the rollup math is well-defined (no NaNs)."""
    model = ImprovedTradingLSTM()
    x = torch.randn(2, 60, fs.INPUT)
    sids = torch.zeros(2, dtype=torch.long)
    _, _, _, exits, _ = model(x, sids)
    logret = exits["next_candle_logret"].detach().numpy()
    entry = 100.0
    # Roll up close-log-returns: close[k] = entry * exp(sum of log-returns up to k+1).
    closes = entry * np.exp(np.cumsum(logret[..., 3], axis=1))
    assert closes.shape == (2, K_FUTURE)
    assert np.all(np.isfinite(closes))
    assert (closes > 0).all()
