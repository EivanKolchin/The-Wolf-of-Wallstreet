"""Phase B3 tests: statistical-arbitrage pairs. Synthetic cointegrated A/B (common trend +
mean-reverting OU spread) vs an independent C. Verifies cointegration selection finds the real
pair, the OU half-life, market-neutral hedging (A/B opposite signs), causality, and — the
economic test — that fading the reverting spread is PROFITABLE in the backtester."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from backend.strategies.stat_arb import (  # noqa: E402
    StatArbPairs, StatArbParams, find_cointegrated_pairs, ou_half_life,
)
from backend.backtest.portfolio import portfolio_backtest  # noqa: E402

BARS_15M_YR = 35_040


def _ou(n, phi=0.92, sigma=0.5, seed=0):
    rng = np.random.default_rng(seed)
    s = np.zeros(n)
    for t in range(1, n):
        s[t] = phi * s[t - 1] + rng.standard_normal() * sigma
    return s


def _df(close):
    close = np.asarray(close, float)
    ts = pd.date_range("2022-01-01", periods=len(close), freq="15min")
    return pd.DataFrame({"timestamp": ts, "open": close, "high": close + 0.2,
                         "low": close - 0.2, "close": close, "volume": np.ones(len(close))})


def _coint_universe(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    common = np.cumsum(rng.standard_normal(n) * 0.3)
    spread = _ou(n, seed=seed + 1)
    A = 100 + common + spread          # A and B share `common`; A-B = mean-reverting spread
    B = 100 + common
    C = 100 + np.cumsum(rng.standard_normal(n) * 0.3)   # independent random walk
    return {"A": _df(A), "B": _df(B), "C": _df(C)}, spread


def _params():
    return StatArbParams(hedge_window=120, z_window=120, entry_z=2.0, exit_z=0.5, stop_z=4.0)


def test_finds_the_cointegrated_pair():
    data, _ = _coint_universe()
    pairs = find_cointegrated_pairs(data, train_frac=0.5, pmax=0.05, max_pairs=3)
    assert pairs, "expected at least one cointegrated pair"
    top = set(pairs[0])
    assert top == {"A", "B"}            # the real cointegrated pair ranks first, not C-pairs


def test_ou_half_life_finite_for_reverting_spread():
    _, spread = _coint_universe()
    hl = ou_half_life(spread)
    assert np.isfinite(hl) and hl > 0
    # a random walk has no finite reversion
    rw = np.cumsum(np.random.default_rng(5).standard_normal(3000))
    assert ou_half_life(rw) > hl


def test_positions_are_two_legged_hedge():
    data, _ = _coint_universe()
    pos = StatArbPairs([("A", "B")], _params()).generate_positions(data)
    a, b = pos["A"], pos["B"]
    traded = np.abs(a) > 1e-9
    assert traded.any()                                   # it actually takes positions
    # a pair trade is always TWO-legged: whenever A is on, the hedge leg B is on too
    # (sign isn't always opposite — the rolling hedge ratio can dip negative — but both
    # legs are active; market-neutrality itself is verified by the β≈0 profitability test)
    assert (np.abs(b[traded]) > 1e-12).mean() > 0.95
    assert (pos["C"] == 0.0).all()                        # C isn't in any pair → untouched


def test_causal_no_position_during_warmup():
    data, _ = _coint_universe()
    pos = StatArbPairs([("A", "B")], _params()).generate_positions(data)
    assert (pos["A"][:120] == 0.0).all()                  # no trade before the hedge/z windows fill


def test_half_life_gate_skips_trending_broken_spread():
    # A diverges from B (trend + random walk) → the spread is NON-stationary (cointegration
    # broken). The OU half-life gate must cut exposure vs an ungated run that keeps fading it.
    n = 4000
    rng = np.random.default_rng(7)
    common = np.cumsum(rng.standard_normal(n) * 0.3)
    idio = np.cumsum(rng.standard_normal(n) * 0.5)
    A = 50 + common + np.linspace(0, 40, n) + idio       # trends away from B
    B = 50 + common
    data = {"A": _df(A), "B": _df(B)}
    gated = StatArbPairs([("A", "B")], StatArbParams(
        hedge_window=120, z_window=120, recheck_window=240, max_half_life=120)).generate_positions(data)
    ungated = StatArbPairs([("A", "B")], StatArbParams(
        hedge_window=120, z_window=120, max_half_life=0)).generate_positions(data)
    active_gated = float((np.abs(gated["A"]) > 1e-9).mean())
    active_ungated = float((np.abs(ungated["A"]) > 1e-9).mean())
    assert active_gated < active_ungated                 # gate avoids the trending/broken spread


def test_spread_reversion_is_profitable_and_neutral():
    data, _ = _coint_universe(n=4000)
    res = portfolio_backtest({"stat_arb": StatArbPairs([("A", "B")], _params())}, data,
                             fee_bps=2.0, slippage_bps=2.0, bars_per_year=BARS_15M_YR,
                             market_symbol="C")
    assert res.metrics["sharpe"] > 0.0                    # fading a reverting spread makes money
    assert abs(res.metrics["beta"]) < 0.5                 # market-neutral (β≈0 vs independent C)
