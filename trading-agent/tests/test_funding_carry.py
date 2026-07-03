"""Phase B5 tests: delta-neutral funding carry. PnL is funding accrual (via generate_returns),
not price — so it's the uncorrelated diversifier. Checks it harvests persistent funding, sits
out below threshold, is inert without funding data, and integrates with the backtester."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from backend.strategies.funding_carry import FundingCarry, FundingCarryParams  # noqa: E402
from backend.backtest.portfolio import portfolio_backtest  # noqa: E402


def _df(n, funding=None, price=None):
    price = np.full(n, 100.0) if price is None else np.asarray(price, float)
    d = {"open": price, "high": price + 0.1, "low": price - 0.1, "close": price,
         "volume": np.ones(n)}
    if funding is not None:
        d["funding_rate"] = np.asarray(funding, float)
    return pd.DataFrame(d)


def test_harvests_persistent_positive_funding():
    n = 2000
    df = _df(n, funding=np.full(n, 0.0003))             # steady +3bp/8h funding
    r = FundingCarry(FundingCarryParams(entry_rate=0.0001, periods_per_bar=1.0)).generate_returns({"BTCUSDT": df})
    assert r is not None and r.mean() > 0               # receiving side earns the funding
    assert abs(r[50] - 0.0003) < 1e-6                   # stable side → ~|funding| each bar, no churn


def test_no_harvest_below_threshold():
    n = 500
    df = _df(n, funding=np.full(n, 0.00005))            # below the 1bp entry gate
    r = FundingCarry(FundingCarryParams(entry_rate=0.0001)).generate_returns({"X": df})
    assert np.allclose(r, 0.0)


def test_inert_without_funding_column():
    assert FundingCarry().generate_returns({"X": _df(100)}) is None


def test_integrates_and_is_price_uncorrelated():
    n = 2000
    rng = np.random.default_rng(0)
    price = np.maximum(100 + np.cumsum(rng.standard_normal(n)), 1.0)
    funding = np.full(n, 0.0002)                        # persistent funding, independent of price
    df = _df(n, funding=funding, price=price)
    res = portfolio_backtest({"carry": FundingCarry(FundingCarryParams(periods_per_bar=0.125))},
                             {"BTCUSDT": df}, bars_per_year=24 * 365, market_symbol="BTCUSDT")
    assert np.isfinite(res.net_returns).all()
    assert abs(res.metrics["beta"]) < 0.3              # funding PnL carries ~no market beta
