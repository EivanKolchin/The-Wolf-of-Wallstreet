"""Phase B2 tests: cross-sectional dollar-neutral momentum + the panel alignment helper.
Verifies dollar-neutrality (Σweights≈0), that it longs winners / shorts losers, causality,
reversal mode, the alignment join, and integration with the portfolio backtester."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from backend.strategies.cross_sectional import CrossSectionalMomentum, XSectionalMomentumParams  # noqa: E402
from backend.backtest.portfolio import align_panel, portfolio_backtest  # noqa: E402

BARS_1H_YR = 24 * 365


def _df(close, ts0="2022-01-01"):
    close = np.asarray(close, float)
    ts = pd.date_range(ts0, periods=len(close), freq="1h")
    return pd.DataFrame({"timestamp": ts, "open": close, "high": close + 0.5,
                         "low": close - 0.5, "close": close, "volume": np.ones(len(close))})


def _ranked_universe(n=600):
    # A strongest up … D strongest down (monotone cross-sectional spread)
    return {
        "A": _df(np.linspace(100, 200, n)),
        "B": _df(np.linspace(100, 130, n)),
        "C": _df(np.linspace(100, 80, n)),
        "D": _df(np.linspace(100, 40, n)),
    }


def _params():
    return XSectionalMomentumParams(lookback=168, hold=24, quantile=0.30, kind="momentum")


def test_dollar_neutral_and_ranks_correctly():
    data = _ranked_universe()
    pos = CrossSectionalMomentum(_params()).generate_positions(data)
    syms = list(data)
    # at the last bar the book must be dollar-neutral and long the winner / short the loser
    last = np.array([pos[s][-1] for s in syms])
    assert abs(last.sum()) < 1e-9                      # Σ weights = 0 (dollar-neutral, β≈0)
    assert pos["A"][-1] > 0                            # strongest momentum → long
    assert pos["D"][-1] < 0                            # weakest → short


def test_no_position_before_lookback():
    data = _ranked_universe()
    pos = CrossSectionalMomentum(_params()).generate_positions(data)
    assert all((pos[s][:168] == 0.0).all() for s in data)   # causal: no bet without history


def test_reversal_mode_flips_sign():
    data = _ranked_universe()
    p = _params(); p.kind = "reversal"
    pos = CrossSectionalMomentum(p).generate_positions(data)
    assert pos["A"][-1] < 0 and pos["D"][-1] > 0       # reversal longs the loser, shorts the winner


def test_requires_aligned_equal_length():
    data = {"A": _df(np.linspace(100, 120, 600)), "B": _df(np.linspace(100, 90, 500)),
            "C": _df(np.linspace(100, 110, 600)), "D": _df(np.linspace(100, 95, 600))}
    try:
        CrossSectionalMomentum(_params()).generate_positions(data)
        assert False, "expected ValueError for unaligned lengths"
    except ValueError:
        pass


def test_align_panel_inner_joins_timestamps():
    a = _df(np.arange(100, 200.0), ts0="2022-01-01")             # 100 bars from Jan 1
    b = _df(np.arange(100, 180.0), ts0="2022-01-01 10:00")       # offset start
    aligned = align_panel({"A": a, "B": b})
    L = {len(v) for v in aligned.values()}
    assert len(L) == 1 and L.pop() > 0
    assert (aligned["A"]["timestamp"].values == aligned["B"]["timestamp"].values).all()


def test_integrates_with_portfolio_backtest_low_beta():
    rng = np.random.default_rng(0)
    n = 1200
    # 6 correlated-but-dispersed series (common market factor + idiosyncratic)
    mkt = np.cumsum(rng.standard_normal(n) * 0.5)
    data = {}
    for i, s in enumerate(["A", "B", "C", "D", "E", "F"]):
        idio = np.cumsum(rng.standard_normal(n) * 0.5) + (i - 3) * 0.02 * np.arange(n) / n
        data[s] = _df(np.maximum(100 + mkt + idio, 1.0))
    aligned = align_panel(data)
    res = portfolio_backtest({"xs_mom": CrossSectionalMomentum(_params())}, aligned,
                             bars_per_year=BARS_1H_YR, market_symbol="A")
    assert np.isfinite(res.net_returns).all()
    assert abs(res.metrics["beta"]) < 0.5              # dollar-neutral ⇒ modest market beta
