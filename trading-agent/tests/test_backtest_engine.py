"""Cycle 4: the cost-aware backtest engine — the measuring stick. Deterministic
scenarios pin down PnL accounting, cost/turnover, no-look-ahead, and the metrics."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from backend.backtest.engine import (  # noqa: E402
    run_backtest, compute_metrics, directional_signal, _max_drawdown,
)


def test_long_in_uptrend_profits():
    close = np.array([100, 101, 102, 103, 104], float)
    r = run_backtest(close, np.ones(5), fee_bps=0, slippage_bps=0)
    assert r.equity[-1] > 1.0 and r.metrics["total_return"] > 0


def test_flat_signal_no_cost_no_trades():
    close = np.array([100, 101, 99, 102, 100], float)
    r = run_backtest(close, np.zeros(5), fee_bps=10, slippage_bps=5)
    assert np.allclose(r.equity, 1.0)
    assert r.metrics["num_trades"] == 0 and r.metrics["total_cost"] == 0.0


def test_costs_charged_on_turnover():
    close = np.full(4, 100.0)                       # flat price → only costs move equity
    sig = np.array([1, 1, 0, 1], float)             # enter, hold, exit, re-enter
    r = run_backtest(close, sig, fee_bps=10, slippage_bps=0)
    assert abs(r.metrics["turnover"] - 3.0) < 1e-9  # 1 (enter) + 1 (exit) + 1 (re-enter)
    assert r.equity[-1] < 1.0                       # pure cost drag


def test_short_in_downtrend_profits_when_allowed():
    close = np.array([100, 99, 98, 97, 96], float)
    r = run_backtest(close, -np.ones(5), fee_bps=0, slippage_bps=0, allow_short=True)
    assert r.equity[-1] > 1.0


def test_allow_short_false_clips_to_flat():
    close = np.array([100, 99, 98, 97], float)
    r = run_backtest(close, -np.ones(4), fee_bps=0, slippage_bps=0, allow_short=False)
    assert np.allclose(r.position, 0.0) and np.allclose(r.equity, 1.0)


def test_no_lookahead_last_signal_irrelevant():
    close = np.array([100, 101, 102, 103], float)
    a = run_backtest(close, np.array([1, 1, 1, 0.0]), fee_bps=0, slippage_bps=0)
    b = run_backtest(close, np.array([1, 1, 1, 1.0]), fee_bps=0, slippage_bps=0)
    assert np.allclose(a.equity, b.equity)          # last bar's signal acts on a non-existent next bar


def test_max_drawdown_value():
    eq = np.array([1.0, 1.2, 0.9, 1.1])
    assert abs(_max_drawdown(eq) - (0.9 / 1.2 - 1.0)) < 1e-9


def test_hit_rate_two_trades():
    close = np.array([100, 110, 110, 110, 99, 99], float)
    sig = np.array([1, 0, 0, 1, 0, 0], float)       # one winning long, one losing long
    r = run_backtest(close, sig, fee_bps=0, slippage_bps=0)
    assert r.metrics["num_trades"] == 2
    assert abs(r.metrics["hit_rate"] - 0.5) < 1e-9
    assert r.trades[0]["net"] > 0 and r.trades[1]["net"] < 0


def test_sharpe_and_sortino_positive_for_upward_drift():
    rng = np.random.default_rng(0)
    # Upward drift WITH real downside bars, so Sortino is well-defined (> 0).
    bar = rng.standard_normal(400) * 0.002 + 0.0006
    close = 100 * np.cumprod(1 + bar)
    r = run_backtest(close, np.ones(len(close)), fee_bps=0, slippage_bps=0)
    assert (r.net_returns < 0).any()                # downside exists → Sortino defined
    assert r.metrics["sharpe"] > 0 and r.metrics["sortino"] > 0
    assert r.metrics["max_drawdown"] <= 0.0


# ───────────────────────── directional_signal gate ───────────────────────────
def test_directional_signal_gate():
    p_long = np.array([0.8, 0.2, 0.45, 0.5])
    p_short = np.array([0.1, 0.7, 0.40, 0.5])
    sig = directional_signal(p_long, p_short, min_confidence=0.5, min_edge=0.1)
    # bar0: long (conf .8, edge .7) ; bar1: short ; bar2: below conf → flat ; bar3: edge 0 → flat
    assert list(sig) == [1.0, -1.0, 0.0, 0.0]


def test_directional_signal_no_short_when_disabled():
    p_long = np.array([0.1, 0.9])
    p_short = np.array([0.8, 0.0])
    sig = directional_signal(p_long, p_short, min_confidence=0.5, min_edge=0.1, allow_short=False)
    assert list(sig) == [0.0, 1.0]
