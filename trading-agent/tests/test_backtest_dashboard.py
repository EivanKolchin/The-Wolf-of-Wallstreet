"""Tests for the streaming backtest dashboard's PURE compute layer (no FastAPI/network).

Covers: every single-asset strategy backtests to a sane equity+stats; the random Monte-Carlo band
is ordered (p10<=median<=p90) and reproducible; SPY reference aligns to an arbitrary calendar;
downsampling preserves endpoints; the annual-return distribution is correct; book strategies
(managed/cross-sectional) run on a synthetic universe and reindex onto a display calendar; and the
live news overlay maps GDELT tone through the real risk overlay (requests monkeypatched).
"""
import types

import numpy as np
import pandas as pd
import pytest

import scripts.backtest_dashboard as bd


def _synthetic(n=900, seed=1, drift=0.0004, vol=0.01, start="2015-01-01"):
    rng = np.random.default_rng(seed)
    rets = rng.normal(drift, vol, n)
    close = 100.0 * np.cumprod(1.0 + rets)
    dates = pd.date_range(start, periods=n, freq="D")
    high = close * (1.0 + np.abs(rng.normal(0, vol / 2, n)))
    low = close * (1.0 - np.abs(rng.normal(0, vol / 2, n)))
    return pd.DataFrame({"timestamp": dates, "open": close, "high": high, "low": low,
                         "close": close, "volume": np.full(n, 1e6)})


def test_run_strategy_shapes_and_stats():
    df = _synthetic()
    out = bd.run_strategy("faber_trend", df, np.zeros(len(df)), fee_bps=5, slippage_bps=5, allow_short=False)
    assert out["equity"].shape[0] == len(df) and out["equity"][0] > 0
    st = out["stats"]
    for k in ("sharpe", "sortino", "calmar", "cagr", "max_drawdown", "ann_vol",
              "alpha", "beta", "sr_bar", "n_obs", "annual"):
        assert k in st
    assert st["max_drawdown"] <= 0.0 and np.isfinite(st["sharpe"])


@pytest.mark.parametrize("name", list(bd.SINGLE_STRATEGIES.keys()))
def test_every_single_strategy_runs(name):
    """All ~30 single-asset strategies (zoo + extras + fibonacci) backtest cleanly."""
    df = _synthetic(seed=3)
    out = bd.run_strategy(name, df, np.zeros(len(df)), fee_bps=5, slippage_bps=5, allow_short=True)
    assert out["equity"].shape[0] == len(df)
    assert np.all(out["equity"] > 0)
    assert np.isfinite(out["stats"]["sharpe"])


def test_fibonacci_strategy_is_present_and_trades():
    assert "fib_retrace" in bd.SINGLE_STRATEGIES and bd.FAMILY["fib_retrace"] == "fibonacci"
    df = _synthetic(seed=8, n=1200)
    o, h, l, c, v = (df[k].to_numpy(float) for k in ("open", "high", "low", "close", "volume"))
    pos = bd.s_fibonacci(o, h, l, c, v)
    assert pos.shape[0] == len(df)
    assert set(np.unique(pos)).issubset({-1.0, 0.0, 1.0})
    assert np.abs(pos).sum() > 0       # it actually takes positions sometimes


def test_random_band_ordered_and_reproducible():
    df = _synthetic(seed=5)
    rb = bd.random_band(df, np.zeros(len(df)), fee_bps=5, slippage_bps=5, allow_short=True, n_runs=20, seed=11)
    for key in ("median", "p10", "p90"):
        assert rb[key].shape[0] == len(df)
    assert np.all(rb["p10"] <= rb["median"] + 1e-9)
    assert np.all(rb["median"] <= rb["p90"] + 1e-9)
    again = bd.random_band(df, np.zeros(len(df)), fee_bps=5, slippage_bps=5, allow_short=True, n_runs=20, seed=11)
    assert np.allclose(rb["median"], again["median"])


def test_spy_reference_aligns_to_arbitrary_calendar(monkeypatch):
    spy_df = _synthetic(n=500, seed=7)
    monkeypatch.setattr(bd, "load_asset", lambda symbol, yr: spy_df if symbol == "SPY" else None)
    asset_dates = pd.date_range("2015-02-01", periods=300, freq="D")
    ref = bd.spy_reference(asset_dates, 2015)
    assert ref is not None
    assert ref["equity"].shape[0] == len(asset_dates) == ref["bar_ret"].shape[0]
    assert np.all(np.isfinite(ref["equity"]))


def test_downsample_keeps_endpoints():
    idx = bd.downsample_index(5000, max_points=300)
    assert idx[0] == 0 and idx[-1] == 4999 and len(idx) <= 300
    assert np.array_equal(bd.downsample_index(120, max_points=300), np.arange(120))


def test_annual_distribution_known_values():
    d1 = pd.date_range("2020-01-01", periods=252, freq="B")
    d2 = pd.date_range("2021-01-01", periods=252, freq="B")
    dates = d1.append(d2)
    net = np.concatenate([np.full(252, 1.10 ** (1 / 252) - 1.0), np.full(252, 0.90 ** (1 / 252) - 1.0)])
    dist = bd._calendar_year_distribution(net, pd.DatetimeIndex(dates))
    assert dist["n_years"] == 2 and dist["pos_years"] == 1
    assert dist["best"] == pytest.approx(0.10, abs=1e-3)
    assert dist["worst"] == pytest.approx(-0.10, abs=1e-3)


def test_buy_hold_matches_price_path():
    df = _synthetic(seed=9)
    out = bd.buy_hold(df, np.zeros(len(df)))
    c = df["close"].to_numpy()
    assert out["stats"]["total_return"] == pytest.approx(c[-1] / c[0] - 1.0, rel=1e-6)


def test_reindex_book_rebases_and_gaps():
    full = pd.date_range("2010-01-01", periods=400, freq="D")
    book_dates = full[100:]                       # book starts later than the display window
    equity_b = np.cumprod(1.0 + np.full(len(book_dates), 0.001))
    overlay = bd.reindex_book(pd.DatetimeIndex(book_dates), equity_b, full)
    assert overlay.shape[0] == len(full)
    assert np.all(np.isnan(overlay[:100]))        # gap before the book starts
    first = np.where(np.isfinite(overlay))[0][0]
    assert overlay[first] == pytest.approx(1.0, abs=1e-9)   # rebased to 1.0 at first visible point


def test_run_book_managed_on_fake_universe(monkeypatch):
    """managed_book runs end-to-end on a synthetic 7-asset universe and returns full stats."""
    uni = {s: _synthetic(n=1500, seed=i, drift=0.0005) for i, s in enumerate(bd.MANAGED_BOOK_SYMS)}
    monkeypatch.setattr(bd, "load_universe", lambda syms, yr: {s: uni[s] for s in syms if s in uni})
    monkeypatch.setattr(bd, "load_asset", lambda symbol, yr: uni.get("SPY"))
    out = bd.run_book("managed_book", 2015, 5.0, 5.0)
    assert out is not None
    assert len(out["dates"]) == len(out["equity"])
    assert np.isfinite(out["stats"]["sharpe"]) and out["stats"]["max_drawdown"] <= 0.0


def test_run_book_xs_momentum_on_fake_universe(monkeypatch):
    uni = {s: _synthetic(n=1500, seed=i + 50) for i, s in enumerate(bd.STOCK_UNIVERSE[:20])}
    monkeypatch.setattr(bd, "load_universe", lambda syms, yr: dict(uni))
    monkeypatch.setattr(bd, "load_asset", lambda symbol, yr: next(iter(uni.values())))
    out = bd.run_book("xs_momentum", 2015, 5.0, 5.0)
    assert out is not None and np.isfinite(out["stats"]["sharpe"])
    # dollar-neutral book should carry little market beta
    assert abs(out["stats"]["beta"]) < 0.6


def test_news_overlay_maps_tone_through_risk_overlay(monkeypatch):
    """Negative GDELT tone on a LONG → the real overlay should de-gear or veto (size_scale<=1)."""
    def fake_get(url, params=None, timeout=0):
        mode = (params or {}).get("mode")
        r = types.SimpleNamespace()
        if mode == "tonechart":
            r.json = lambda: {"tonechart": [{"bin": -6, "count": 80}, {"bin": -2, "count": 40},
                                            {"bin": 1, "count": 10}]}
        else:
            r.json = lambda: {"articles": [{"title": "Selloff deepens", "url": "http://x", "domain": "x.com"}]}
        return r

    monkeypatch.setitem(__import__("sys").modules, "requests",
                        types.SimpleNamespace(get=fake_get))
    out = bd.fetch_news_risk("AAPL")
    assert out["available"] and out["n_articles"] == 130
    assert out["signed_score"] < 0                     # net-bearish tone
    assert out["size_scale"] <= 1.0                     # overlay never upsizes a long into bearish news
    assert out["action"] in ("resize", "veto", "halt_asset", "allow")
