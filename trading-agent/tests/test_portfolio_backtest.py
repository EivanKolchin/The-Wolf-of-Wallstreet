"""Phase A tests: the strategy framework + multi-strategy portfolio backtester. Covers the
properties that matter for trustworthy results — causal (no look-ahead) PnL, causal
vol-targeting that actually scales toward the target, per-strategy attribution, the
diversification correlation matrix, and the Deflated Sharpe multiple-testing adjustment."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from backend.strategies.base import Strategy, StrategySpec, positions_to_signals  # noqa: E402
from backend.backtest.portfolio import (  # noqa: E402
    portfolio_backtest, vol_target_scale, strategy_net_returns, drawdown_degear,
)
from backend.backtest.engine import deflated_sharpe_ratio  # noqa: E402

BARS_YR = 105_120


def _ohlcv(n=3000, seed=0, drift=0.0):
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.standard_normal(n) * 0.5 + drift)
    close = np.maximum(close, 1.0)
    return pd.DataFrame({"open": close, "high": close + 0.5, "low": close - 0.5,
                         "close": close, "volume": rng.uniform(1, 100, n)})


class _AlwaysLong(Strategy):
    def generate_positions(self, data):
        return {s: np.ones(len(df)) for s, df in data.items()}


class _UpMomentum(Strategy):
    """Causal: position[t] = sign(close[t]-close[t-1]); backtester applies it to ret[t+1]."""
    def generate_positions(self, data):
        out = {}
        for s, df in data.items():
            c = df["close"].to_numpy()
            out[s] = self._safe_positions(np.sign(np.diff(c, prepend=c[:1])), len(c))
        return out


def _spec(name):
    return StrategySpec(name=name, asset_class="crypto", timeframe="1h")


def test_portfolio_runs_attributes_and_correlation():
    data = {"BTCUSDT": _ohlcv(seed=1), "ETHUSDT": _ohlcv(seed=2)}
    strategies = {"always_long": _AlwaysLong(_spec("always_long")),
                  "up_mom": _UpMomentum(_spec("up_mom"))}
    res = portfolio_backtest(strategies, data, bars_per_year=BARS_YR, market_symbol="BTCUSDT")
    assert res.equity.shape == res.net_returns.shape
    assert set(res.strategy_metrics) == {"always_long", "up_mom"}
    for m in ("sharpe", "reg_alpha", "beta", "max_drawdown", "leverage_mean"):
        assert m in res.metrics
    assert res.correlation.shape == (2, 2)            # diversification view
    assert np.isfinite(res.net_returns).all()


def test_strategy_net_returns_no_lookahead():
    # A position that is +1 ONLY at bar k must earn the return from k -> k+1 (not k-1 -> k).
    close = np.array([100, 101, 103, 102, 104, 100], float)
    data = {"X": pd.DataFrame({"close": close, "high": close, "low": close,
                               "open": close, "volume": np.ones(6)})}
    pos = np.zeros(6); pos[2] = 1.0
    net = strategy_net_returns({"X": pos}, data, fee_bps=0.0, slippage_bps=0.0)
    ret = close[1:] / close[:-1] - 1.0                # bar returns (len 5, index 1..5)
    # held during bar 3 (k=2 -> 3): earns ret at index 3 = 102/103-1
    assert np.isclose(net[3], ret[2])
    assert np.isclose(net[2], 0.0)                    # nothing earned the bar the position is set


def test_vol_target_is_causal_and_scales_to_target():
    rng = np.random.default_rng(3)
    net = rng.standard_normal(6000) * 0.01            # ~constant per-bar vol
    lev = vol_target_scale(net, target_ann_vol=0.12, bars_per_year=BARS_YR, window=96)
    assert lev[0] == 0.0                              # shifted: bar 0 can't use its own vol
    scaled = net * lev
    ann_vol = scaled[200:].std() * np.sqrt(BARS_YR)   # skip warm-up
    assert 0.06 < ann_vol < 0.20                      # lands near the 0.12 target


def test_vol_target_higher_target_more_leverage():
    rng = np.random.default_rng(4); net = rng.standard_normal(4000) * 0.01
    lo = vol_target_scale(net, 0.08, BARS_YR).mean()
    hi = vol_target_scale(net, 0.20, BARS_YR).mean()
    assert hi > lo


def test_drawdown_degear_caps_a_losing_run():
    net = np.full(500, -0.01)                       # a steady losing stream
    dg = drawdown_degear(net, dd_threshold=0.10, floor=0.25)
    eq_raw = np.cumprod(1.0 + net)
    eq_dg = np.cumprod(1.0 + dg)
    dd_raw = float((eq_raw / np.maximum.accumulate(eq_raw) - 1).min())
    dd_dg = float((eq_dg / np.maximum.accumulate(eq_dg) - 1).min())
    assert dd_dg > dd_raw                           # de-geared drawdown is SHALLOWER (less negative)


def test_drawdown_degear_leaves_a_winner_untouched():
    net = np.full(200, 0.01)                        # never in drawdown
    np.testing.assert_allclose(drawdown_degear(net, 0.10, 0.25), net)   # scale stays 1.0


def test_deflated_sharpe_penalises_more_trials():
    base = deflated_sharpe_ratio(0.05, n_obs=2000, n_trials=1)
    many = deflated_sharpe_ratio(0.05, n_obs=2000, n_trials=1000)
    assert base > many                                # more configs tried ⇒ more deflation
    assert deflated_sharpe_ratio(0.10, 2000, 1000) > deflated_sharpe_ratio(0.05, 2000, 1000)
    assert 0.0 <= many <= 1.0


def test_positions_to_signals_bridge():
    data = {"BTCUSDT": _ohlcv(n=50)}
    pos = {"BTCUSDT": np.r_[np.zeros(49), -0.5]}
    sigs = positions_to_signals("s", data, pos)
    assert len(sigs) == 1 and sigs[0].side == -1 and abs(sigs[0].strength - 0.5) < 1e-9
