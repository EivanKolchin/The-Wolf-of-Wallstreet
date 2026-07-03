"""Tests for the daily factor-research mechanics: 12-1 momentum SKIP, outer panel alignment
(staggered IPO dates), and the equity-daily loader's graceful behaviour. The full factor run
needs yfinance + network (owner runs it); these cover the pure logic offline."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from backend.strategies.cross_sectional import (  # noqa: E402
    CrossSectionalMomentum, XSectionalMomentumParams, LowVolatility, LowVolParams,
)
from backend.backtest.portfolio import align_panel, portfolio_backtest  # noqa: E402
from backend.data.equity_daily import load_daily, FALLBACK_UNIVERSE  # noqa: E402


def _dly(close, start="2010-01-01"):
    close = np.asarray(close, float)
    ts = pd.date_range(start, periods=len(close), freq="1D")
    return pd.DataFrame({"timestamp": ts, "open": close, "high": close + 0.5,
                         "low": close - 0.5, "close": close, "volume": np.ones(len(close))})


def test_skip_momentum_differs_and_is_causal():
    n = 600
    up_then_crash = np.r_[np.linspace(100, 200, n - 20), np.linspace(200, 90, 20)]    # 12-1 winner, 0-1 LOSER
    steady_up = np.linspace(100, 240, n)
    steady_down = np.linspace(220, 90, n)
    down_then_rally = np.r_[np.linspace(150, 90, n - 20), np.linspace(90, 200, 20)]   # 12-1 loser, 0-1 WINNER
    data = {"A": _dly(up_then_crash), "B": _dly(steady_up), "C": _dly(steady_down), "D": _dly(down_then_rally)}

    # quantile 0.5 → k=2 of 4 per side, so the ranking flip on A is expressed in its position
    p_skip = XSectionalMomentumParams(lookback=252, skip=21, hold=21, quantile=0.5)
    p_noskip = XSectionalMomentumParams(lookback=252, skip=0, hold=21, quantile=0.5)
    pos_skip = CrossSectionalMomentum(p_skip).generate_positions(data)
    pos_noskip = CrossSectionalMomentum(p_noskip).generate_positions(data)

    # skip ignores the recent reversal → A (12-1 winner) should be long under skip but not no-skip
    assert pos_skip["A"][-1] > pos_noskip["A"][-1]
    # causal: nothing before the lookback window fills
    assert (pos_skip["A"][:252] == 0.0).all()


def test_outer_align_handles_staggered_listing():
    a = _dly(np.linspace(100, 120, 300), start="2010-01-01")          # lists earlier
    b = _dly(np.linspace(50, 70, 300), start="2010-06-01")            # lists later (overlaps)
    c = _dly(np.linspace(80, 60, 300), start="2010-01-01")
    d = _dly(np.linspace(30, 45, 300), start="2010-03-01")
    aligned = align_panel({"A": a, "B": b, "C": c, "D": d}, how="outer")
    lengths = {len(v) for v in aligned.values()}
    assert len(lengths) == 1                                          # equal length = union of dates
    assert aligned["B"]["close"].isna().any()                        # NaN-padded before B's listing
    # cross-sectional ranking tolerates the NaN (ranks only listed names) and runs end-to-end
    res = portfolio_backtest({"mom": CrossSectionalMomentum(XSectionalMomentumParams(
        lookback=60, hold=10, quantile=0.25))}, aligned, bars_per_year=252, market_symbol=None)
    assert np.isfinite(res.net_returns).all()


def test_low_vol_longs_calm_names_shorts_wild_ones():
    rng = np.random.default_rng(0)
    n = 400
    calm = 100 + np.cumsum(rng.standard_normal(n) * 0.05)      # low realized vol
    calm2 = 100 + np.cumsum(rng.standard_normal(n) * 0.07)
    wild = 100 + np.cumsum(rng.standard_normal(n) * 0.9)       # high realized vol
    wild2 = 100 + np.cumsum(rng.standard_normal(n) * 1.1)
    data = {"CALM": _dly(np.maximum(calm, 1)), "CALM2": _dly(np.maximum(calm2, 1)),
            "WILD": _dly(np.maximum(wild, 1)), "WILD2": _dly(np.maximum(wild2, 1))}
    pos = LowVolatility(LowVolParams(vol_window=60, hold=21, quantile=0.5)).generate_positions(data)
    last = {s: pos[s][-1] for s in data}
    assert last["CALM"] > 0 and last["CALM2"] > 0             # long the low-vol names
    assert last["WILD"] < 0 and last["WILD2"] < 0             # short the high-vol names
    assert (pos["CALM"][:60] == 0.0).all()                   # causal warm-up


def test_equity_daily_loader_graceful_without_cache():
    # skip_download with an unknown ticker (no cache) → returns empty, no crash
    assert load_daily(["___NOPE___"], start="2010-01-01", skip_download=True) == {}
    assert len(FALLBACK_UNIVERSE) >= 40 and "NVDA" in FALLBACK_UNIVERSE
