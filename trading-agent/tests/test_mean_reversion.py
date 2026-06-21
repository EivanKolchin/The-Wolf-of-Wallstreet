"""Phase B4 tests: regime-gated mean-reversion. The defining property is the regime gate —
it fades extremes in a RANGE but stands aside in a TREND (the stat-arb failure mode). Also
checks the fade direction, exit-at-mean, causality, and portfolio integration."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from backend.strategies.mean_reversion import MeanReversion, MeanReversionParams  # noqa: E402
from backend.backtest.portfolio import portfolio_backtest  # noqa: E402

BARS_1H_YR = 24 * 365


def _df(close):
    close = np.asarray(close, float)
    rng = np.random.default_rng(0)
    hi = close + np.abs(rng.standard_normal(len(close))) * 0.2 + 0.1
    lo = close - np.abs(rng.standard_normal(len(close))) * 0.2 - 0.1
    return pd.DataFrame({"open": close, "high": hi, "low": lo, "close": close,
                         "volume": np.ones(len(close))})


def _params(adx_max=0.0):
    # adx_max=0 disables the gate (isolate the fade logic); >0 enables the regime gate
    return MeanReversionParams(z_window=40, entry_z=2.0, exit_z=0.5, stop_z=5.0, adx_max=adx_max)


def test_fades_an_oversold_dip_long_then_exits():
    # ranging series with a sharp oversold dip that reverts to the mean
    base = 100 + np.sin(np.arange(400) * 0.3) * 0.5         # gentle range
    base[200:210] -= 6.0                                    # sharp dip (oversold)
    base[210:230] = 100.0                                   # revert to mean
    pos = MeanReversion(_params(adx_max=0.0)).generate_positions({"X": _df(base)})["X"]
    assert (pos[200:212] == 1.0).any()                      # went long into the oversold dip
    assert pos[230] == 0.0                                  # exited once back at the mean


def test_regime_gate_trades_range_not_trend():
    # The gate's job: fade extremes in a RANGE, stand aside in a TREND. Compare the SAME
    # gated strategy on an oscillating range vs a strong trend — it should trade the range
    # far more than the trend (where high ADX blocks entries — the stat-arb failure mode).
    rng = np.random.default_rng(3)
    n = 800
    rng_series = 100 + rng.standard_normal(n) * 3.0                                    # ranging (no trend)
    trend_series = np.linspace(100, 230, n) + np.cumsum(rng.standard_normal(n)) * 0.4  # trending
    strat = MeanReversion(_params(adx_max=20.0))
    active_range = float((np.abs(strat.generate_positions({"X": _df(rng_series)})["X"]) > 0).mean())
    active_trend = float((np.abs(strat.generate_positions({"X": _df(trend_series)})["X"]) > 0).mean())
    assert active_range > 0.0                       # it DOES fade extremes in a range
    assert active_trend < active_range              # but trades far less in a trend (gate works)


def test_causal_no_position_during_warmup():
    pos = MeanReversion(_params()).generate_positions({"X": _df(100 + np.random.default_rng(1).standard_normal(300))})["X"]
    assert (pos[:20] == 0.0).all()


def test_integrates_with_portfolio_backtest():
    rng = np.random.default_rng(2)
    data = {s: _df(np.maximum(100 + np.cumsum(rng.standard_normal(2000) * 0.3), 1.0))
            for s in ("BTCUSDT", "ETHUSDT")}
    res = portfolio_backtest({"mr": MeanReversion(_params(adx_max=20.0))}, data,
                             bars_per_year=BARS_1H_YR, market_symbol="BTCUSDT")
    assert np.isfinite(res.net_returns).all()
    assert "sharpe" in res.metrics
