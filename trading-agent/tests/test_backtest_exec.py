"""Cycle 5: execution-aware backtest — vol-targeted sizing, ATR trailing stop,
move-to-breakeven, partial scale-out, exposure cap. Deterministic scenarios."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from backend.backtest.engine import run_exec_backtest, atr_from_ohlc  # noqa: E402


def _ohlc(close):
    close = np.asarray(close, float)
    return close, close + 0.1, close - 0.1   # close, high, low


def test_vol_targeting_scales_size_inversely_to_vol():
    n = 20
    close, high, low = _ohlc(np.full(n, 100.0))
    atr = np.ones(n)
    sig = np.ones(n)                       # hold long the whole way (no stop: far)
    common = dict(max_size=1.0, stop_atr=99, trail_atr=99, breakeven_atr=999,
                  tp1_atr=999, scale_out_frac=0.0, fee_bps=0, slippage_bps=0)
    lo = run_exec_backtest(close, high, low, atr, sig, forecast_vol=np.full(n, 0.005),
                           target_vol=0.01, **common)   # 0.01/0.005=2 → capped at 1.0
    hi = run_exec_backtest(close, high, low, atr, sig, forecast_vol=np.full(n, 0.02),
                           target_vol=0.01, **common)   # 0.01/0.02 = 0.5
    assert lo.metrics["avg_exposure"] > hi.metrics["avg_exposure"] + 0.3
    assert abs(hi.metrics["avg_exposure"] - 0.5) < 1e-6


def test_trailing_stop_banks_profit_on_reversal():
    up = np.linspace(100, 110, 11)
    down = np.linspace(110, 100, 11)[1:]
    close, high, low = _ohlc(np.concatenate([up, down]))
    atr = np.ones(len(close))
    sig = np.ones(len(close))
    res = run_exec_backtest(close, high, low, atr, sig, forecast_vol=None, max_size=1.0,
                            stop_atr=2.0, trail_atr=2.0, breakeven_atr=999, tp1_atr=999,
                            scale_out_frac=0.0, fee_bps=0, slippage_bps=0)
    assert res.equity[-1] > 1.0            # exited near the top, didn't ride it all the way down
    assert (res.position == 0).any()       # the trailing stop fired


def test_scale_out_halves_position():
    close, high, low = _ohlc(np.linspace(100, 120, 21))
    atr = np.ones(21)
    sig = np.ones(21)
    res = run_exec_backtest(close, high, low, atr, sig, forecast_vol=None, max_size=1.0,
                            stop_atr=99, trail_atr=99, breakeven_atr=999, tp1_atr=2.0,
                            scale_out_frac=0.5, fee_bps=0, slippage_bps=0)
    assert abs(res.position.max() - 1.0) < 1e-9      # full size early
    assert 0.4 < res.position[-1] < 0.6              # halved after +2*ATR


def test_breakeven_reduces_loss_vs_far_stop():
    close, high, low = _ohlc(np.array([100, 101.5, 101, 100, 99, 98, 97], float))
    atr = np.ones(len(close))
    sig = np.ones(len(close))
    far = run_exec_backtest(close, high, low, atr, sig, forecast_vol=None, stop_atr=10,
                            trail_atr=10, breakeven_atr=999, tp1_atr=999, scale_out_frac=0,
                            fee_bps=0, slippage_bps=0)
    be = run_exec_backtest(close, high, low, atr, sig, forecast_vol=None, stop_atr=10,
                           trail_atr=10, breakeven_atr=1.0, tp1_atr=999, scale_out_frac=0,
                           fee_bps=0, slippage_bps=0)
    assert be.equity[-1] > far.equity[-1]            # breakeven exited near entry, cut the loss


def test_no_short_when_disabled():
    close, high, low = _ohlc(np.linspace(100, 90, 11))
    atr = np.ones(len(close))
    sig = -np.ones(len(close))
    res = run_exec_backtest(close, high, low, atr, sig, forecast_vol=None, allow_short=False,
                            fee_bps=0, slippage_bps=0)
    assert np.allclose(res.position, 0.0) and np.allclose(res.equity, 1.0)


def test_atr_from_ohlc_is_positive_and_shaped():
    rng = np.random.default_rng(0)
    close = 100 + np.cumsum(rng.standard_normal(200)) * 0.1
    atr = atr_from_ohlc(close + 0.2, close - 0.2, close, window=14)
    assert atr.shape == close.shape and (atr > 0).all()
