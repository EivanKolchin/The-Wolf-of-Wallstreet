"""Fundamentals layer tests — the look-ahead firewall is the whole value of doing this honestly,
so the point-in-time logic is tested hardest: a figure must never influence a date before its SEC
filing date, and a late restatement of an old period must not override a newer period. All offline
(synthetic facts/signals) — the network fetch is the owner's to run."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from backend.data.fundamentals import (  # noqa: E402
    point_in_time_series, _annualise, _concept_facts, build_concept,
)
from backend.strategies.cross_sectional import CrossSectionalScore, ScoreParams  # noqa: E402


def _facts(rows):
    """rows: list of (end, val, filed[, days]) -> the tidy frame point_in_time_series consumes."""
    df = pd.DataFrame([(e, v, f, (d[0] if d else np.nan)) for e, v, f, *d in rows],
                      columns=["end", "val", "filed", "days"])
    df["end"] = pd.to_datetime(df["end"]); df["filed"] = pd.to_datetime(df["filed"])
    return df.sort_values(["filed", "end"]).reset_index(drop=True)


def test_point_in_time_is_strictly_causal():
    df = _facts([("2020-12-31", 100.0, "2021-02-15"),
                 ("2021-12-31", 120.0, "2022-02-15")])
    asof = pd.date_range("2021-01-01", "2022-06-01", freq="D")
    s = point_in_time_series(df, asof)
    at = lambda d: s[asof.get_loc(pd.Timestamp(d))]
    assert np.isnan(at("2021-02-01"))          # before ANY filing → unknown
    assert at("2021-02-15") == 100.0           # known on the filing date, not before
    assert at("2022-02-14") == 100.0           # still the old figure the day before the new filing
    assert at("2022-02-15") == 120.0           # new annual figure once it is filed


def test_late_restatement_of_old_period_does_not_override_newer():
    df = _facts([("2020-12-31", 100.0, "2021-02-15"),
                 ("2021-12-31", 120.0, "2022-02-15"),
                 ("2020-12-31", 105.0, "2022-05-01")])   # restates 2020, filed AFTER 2021's report
    asof = pd.date_range("2022-04-01", "2022-06-01", freq="D")
    s = point_in_time_series(df, asof)
    assert s[-1] == 120.0          # freshest period (2021) stays; stale-period restatement ignored


def test_restatement_of_current_period_takes_effect_on_its_filing():
    df = _facts([("2021-12-31", 120.0, "2022-02-15"),
                 ("2021-12-31", 130.0, "2022-05-01")])   # restates the LATEST period
    asof = pd.date_range("2022-02-01", "2022-06-01", freq="D")
    s = point_in_time_series(df, asof)
    at = lambda d: s[asof.get_loc(pd.Timestamp(d))]
    assert at("2022-03-01") == 120.0           # original
    assert at("2022-05-01") == 130.0           # restated value effective on its filing date


def test_annualise_keeps_only_year_length_periods():
    df = _facts([("2020-12-31", 9.0, "2021-02-15", 365),
                 ("2020-03-31", 2.0, "2020-05-01", 90),
                 ("2020-06-30", 4.0, "2020-08-01", 91),
                 ("2021-12-31", 10.0, "2022-02-15", 366)])
    a = _annualise(df)
    assert set(a["days"]) == {365, 366}        # quarterly (~90d) dropped, annual kept


def test_gross_profit_falls_back_to_revenue_minus_cost():
    # facts with NO GrossProfit but Revenues and CostOfRevenue (annual) → computed fallback
    facts = {"facts": {"us-gaap": {
        "Revenues": {"units": {"USD": [{"start": "2020-01-01", "end": "2020-12-31", "val": 1000.0, "filed": "2021-02-15"}]}},
        "CostOfRevenue": {"units": {"USD": [{"start": "2020-01-01", "end": "2020-12-31", "val": 600.0, "filed": "2021-02-15"}]}},
    }}}
    asof = pd.date_range("2021-02-15", "2021-03-01", freq="D")
    gp = build_concept("X", "gross_profit", asof, cik_map={}, facts=facts)
    assert np.isclose(gp[-1], 400.0)           # 1000 - 600


def test_concept_facts_uses_first_present_alias():
    facts = {"facts": {"us-gaap": {
        "NetIncomeLoss": {"units": {"USD": [{"start": "2020-01-01", "end": "2020-12-31", "val": 50.0, "filed": "2021-02-15"}]}},
    }}}
    df = _concept_facts(facts, ["NetIncomeLoss", "ProfitLoss"])
    assert len(df) == 1 and df["val"].iloc[0] == 50.0 and df["days"].iloc[0] == 365


def _ohlcv(n, start="2015-01-01"):
    ts = pd.date_range(start, periods=n, freq="1D")
    c = np.linspace(100, 110, n)
    return pd.DataFrame({"timestamp": ts, "open": c, "high": c + 1, "low": c - 1,
                         "close": c, "volume": np.ones(n)})


def test_cross_sectional_score_longs_high_shorts_low_and_respects_warmup():
    n = 60
    syms = ["A", "B", "C", "D"]
    data = {s: _ohlcv(n) for s in syms}
    # A best, D worst, every bar; quantile 0.25 over 4 names → k=1 (long A / short D)
    signal = np.tile(np.array([3.0, 2.0, 1.0, 0.0]), (n, 1))
    strat = CrossSectionalScore(signal, syms, name="q",
                                params=ScoreParams(hold=5, warmup=10, quantile=0.25))
    pos = strat.generate_positions(data)
    assert pos["A"][-1] > 0 and pos["D"][-1] < 0          # long the top, short the bottom
    assert abs(pos["B"][-1]) < 1e-9 and abs(pos["C"][-1]) < 1e-9
    assert (pos["A"][:10] == 0.0).all()                   # no positions before warm-up (causal)


def test_cross_sectional_score_higher_is_better_flag_flips_side():
    n = 60
    syms = ["A", "B", "C", "D"]
    data = {s: _ohlcv(n) for s in syms}
    signal = np.tile(np.array([3.0, 2.0, 1.0, 0.0]), (n, 1))
    flipped = CrossSectionalScore(signal, syms, name="q",
                                  params=ScoreParams(hold=5, warmup=10, quantile=0.25,
                                                     higher_is_better=False))
    pos = flipped.generate_positions(data)
    assert pos["A"][-1] < 0 and pos["D"][-1] > 0          # low-is-better → short A / long D
