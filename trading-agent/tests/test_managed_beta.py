"""ManagedBeta tests — the trend overlay's contract: long in uptrends, step out below trend, and
(the point of the whole pivot) cut the drawdown of a crash versus buy-and-hold. Plus strict
causality: a position can't react to a move that hasn't happened yet."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from backend.strategies.managed_beta import ManagedBeta, ManagedBetaParams  # noqa: E402


def _series(close, start="2010-01-01"):
    c = np.asarray(close, float)
    ts = pd.date_range(start, periods=len(c), freq="1D")
    return pd.DataFrame({"timestamp": ts, "open": c, "high": c, "low": c, "close": c,
                         "volume": np.ones(len(c))})


def _bar_ret(c):
    r = np.zeros(len(c)); r[1:] = c[1:] / c[:-1] - 1.0
    return r


def _max_dd(net):
    eq = np.cumprod(1.0 + net)
    return float((eq / np.maximum.accumulate(eq) - 1).min())


def test_long_in_uptrend_flat_below_trend():
    up = np.linspace(100, 200, 300)
    dn = np.linspace(200, 100, 300)
    mb = ManagedBeta(ManagedBetaParams(trend_ema=50, up_exposure=1.0, down_exposure=0.0))
    pos = mb.generate_positions({"UP": _series(up), "DN": _series(dn)})
    assert pos["UP"][-1] == 1.0          # sustained uptrend → fully invested
    assert pos["DN"][-1] == 0.0          # sustained downtrend → in cash


def test_trend_filter_cuts_the_crash_drawdown():
    # rise, then a severe crash: the EMA filter should step out and avoid most of the drop
    rise = np.linspace(100, 200, 250)
    crash = np.linspace(200, 80, 60)
    close = np.r_[rise, crash]
    mb = ManagedBeta(ManagedBetaParams(trend_ema=50, down_exposure=0.0))
    pos = mb.generate_positions({"X": _series(close)})["X"]
    ret = _bar_ret(close)
    managed = np.zeros(len(ret)); managed[1:] = pos[:-1] * ret[1:]
    bh_dd = _max_dd(ret)
    mg_dd = _max_dd(managed)
    assert mg_dd > bh_dd                 # managed drawdown is SHALLOWER (closer to 0)
    assert bh_dd < -0.3                  # sanity: buy-and-hold really did crash hard


def test_down_exposure_short_leg_profits_in_a_crash():
    close = np.r_[np.linspace(100, 200, 250), np.linspace(200, 80, 60)]
    mb = ManagedBeta(ManagedBetaParams(trend_ema=50, down_exposure=-1.0))
    pos = mb.generate_positions({"X": _series(close)})["X"]
    assert pos[-1] == -1.0               # decisively below trend → short (trend-follow)


def test_positions_are_causal_no_lookahead():
    base = np.r_[np.linspace(100, 180, 200), np.linspace(180, 170, 50)]
    k = 220
    crashed = base.copy(); crashed[k + 1:] = np.linspace(170, 60, len(base) - k - 1)  # change only the FUTURE
    mb = ManagedBeta(ManagedBetaParams(trend_ema=50))
    p_base = mb.generate_positions({"X": _series(base)})["X"]
    p_crash = mb.generate_positions({"X": _series(crashed)})["X"]
    # positions up to and including k must be identical — they cannot see the post-k divergence
    np.testing.assert_allclose(p_base[:k + 1], p_crash[:k + 1])


def test_short_series_returns_flat():
    mb = ManagedBeta(ManagedBetaParams(trend_ema=200))
    pos = mb.generate_positions({"X": _series(np.linspace(100, 110, 50))})["X"]
    assert (pos == 0.0).all()            # fewer bars than warm-up → no positions
