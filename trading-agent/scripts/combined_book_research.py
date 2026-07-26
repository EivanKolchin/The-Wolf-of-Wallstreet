#!/usr/bin/env python
"""Combined-book diversification test — does adding a β≈0 stream LIFT the managed-beta Sharpe?

The whole arc proved the only durable standalone edge is the managed-beta book (disciplined beta
harvesting, β≈0.5). The fundamental law says portfolio Sharpe grows with UNCORRELATED breadth, not
name-count. The 4h regime-gated TS-momentum crypto book is β≈0 and ~0.9 standalone Sharpe on the
2022+ sample (not promotable alone — Deflated Sharpe ~0.26 — but β≈0 is exactly what the beta-heavy
managed book lacks). This script measures the REAL question: combined Sharpe vs each sleeve alone.

Honest alignment: managed-beta is daily equities/crypto (yfinance, 252 trading days); the TS-mom book
is 4h crypto (365d). We compound the 4h book to daily, then resample BOTH onto the equity trading
calendar (weekend crypto returns compounded into the next weekday — no return dropped), inner-joined
to the overlap window. Annualised at 252 (the joined calendar). No look-ahead: every sleeve is causal.

    python scripts/combined_book_research.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

if hasattr(sys.stdout, "reconfigure"):          # Windows cp1252 console → UTF-8
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from backend.data.equity_daily import load_daily                                   # noqa: E402
from backend.backtest.portfolio import (align_panel, vol_target_scale,             # noqa: E402
                                        drawdown_degear, _bar_returns, portfolio_backtest)
from backend.strategies.managed_beta import ManagedBeta, ManagedBetaParams          # noqa: E402
from backend.strategies.ts_momentum import TSMomentumBreakout, TSMomentumParams     # noqa: E402
from backend.strategies.tsmom_multiasset import TSMOMMultiAsset, TSMOMParams        # noqa: E402

PPY = 252

# managed-beta universe (the widened equal-weight book — the robust weighting)
MB_SYMBOLS = ["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "IEF", "GLD", "DBC", "TQQQ",
              "BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "ADA-USD", "AVAX-USD", "LINK-USD",
              "DOGE-USD", "LTC-USD", "BCH-USD", "SLV", "USO", "EWJ", "FXI", "EWZ",
              "HYG", "LQD", "SHY", "VNQ"]
# TS-momentum crypto universe (4h)
TS_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT", "AVAXUSDT",
              "LINKUSDT", "LTCUSDT", "BCHUSDT", "DOTUSDT", "ATOMUSDT", "ETCUSDT", "FILUSDT",
              "NEARUSDT", "UNIUSDT", "XLMUSDT", "ALGOUSDT", "AAVEUSDT", "MATICUSDT", "RENDERUSDT"]

# FX + commodity trend universe (daily ETFs) — the candidate 4th sleeve. A DIFFERENT risk
# premium (currency carry/trend + commodity trend), historically the diversifying leg of every
# managed-futures TSMOM book, so its correlation to the equity/crypto-beta sleeves should be low.
FXC_SYMBOLS = [
    "UUP", "UDN", "FXE", "FXY", "FXB", "FXF", "FXC",     # dollar + majors (EUR/JPY/GBP/CHF/CAD)
    "DBC", "USO", "UNG", "SLV", "GLD", "DBA", "CPER", "DBB",  # broad/energy/metals/ag commodity
]

COST = (2.0 + 3.0) / 1e4        # managed-beta daily cost (fee + slippage bps)


# ───────────────────────── managed-beta daily book ─────────────────────────
def _managed_net(pos, ret, cost):
    m = min(len(pos), len(ret)); pos, ret = pos[:m], ret[:m]
    gross = np.zeros(m); gross[1:] = pos[:-1] * ret[1:]
    turn = np.zeros(m); turn[0] = abs(pos[0]); turn[1:] = np.abs(pos[1:] - pos[:-1])
    return gross - turn * cost


def _causal_equal_weights(R, window=63):
    """(S,L) equal weights among assets ACTIVE at each bar (trailing vol defined), causal."""
    roll = pd.DataFrame(R.T).rolling(window, min_periods=max(10, window // 4)).std().to_numpy()
    active = (roll > 1e-9).astype(float)
    rs = active.sum(axis=1, keepdims=True)
    w = np.where(rs > 0, active / np.where(rs > 0, rs, 1.0), 0.0)
    w = pd.DataFrame(w).shift(1).to_numpy()
    return np.nan_to_num(w, nan=0.0).T


def managed_beta_daily(symbols, start="2010-01-01", target_vol=0.15, max_lev=2.0):
    """Daily net-return series (pd.Series, DatetimeIndex) of the widened equal-weight managed book."""
    data = load_daily(symbols, start=start, skip_download=True)
    if not data:
        raise RuntimeError("no managed-beta data cached")
    aligned = align_panel(data, how="outer")
    syms = list(aligned)
    idx = pd.to_datetime(aligned[syms[0]]["timestamp"].to_numpy())
    pos = ManagedBeta(ManagedBetaParams()).generate_positions(aligned)
    per_asset = [_managed_net(pos[s], _bar_returns(aligned[s]["close"].to_numpy()), COST) for s in syms]
    R = np.vstack(per_asset)
    w = _causal_equal_weights(R)
    combined = (w * R).sum(axis=0)
    net = combined * vol_target_scale(combined, target_vol, PPY, max_leverage=max_lev)
    net = drawdown_degear(net, 0.15, 0.25)
    return pd.Series(net, index=idx).dropna()


# ───────────────────── daily multi-asset TSMOM long-short book ─────────────────────
def tsmom_daily(symbols, start="2010-01-01", target_vol=0.15, max_lev=2.0, skip_download=True):
    """Daily net-return series of the MOP-style long-short TSMOM sleeve on a broad daily
    universe. Symmetric (shorts sustained downtrends — crisis alpha where the long-biased
    sleeve just sits in cash), per-asset inverse-vol scaled, then the same causal book overlay
    (equal-weight across active → vol-target → de-gear)."""
    data = load_daily(symbols, start=start, skip_download=skip_download)
    if not data:
        raise RuntimeError("no daily data cached")
    aligned = align_panel(data, how="outer")
    syms = list(aligned)
    idx = pd.to_datetime(aligned[syms[0]]["timestamp"].to_numpy())
    pos = TSMOMMultiAsset(TSMOMParams()).generate_positions(aligned)
    per_asset = [_managed_net(pos[s], _bar_returns(aligned[s]["close"].to_numpy()), COST) for s in syms]
    R = np.vstack(per_asset)
    w = _causal_equal_weights(R)
    combined = (w * R).sum(axis=0)
    net = combined * vol_target_scale(combined, target_vol, PPY, max_leverage=max_lev)
    net = drawdown_degear(net, 0.15, 0.25)
    return pd.Series(net, index=idx).dropna()


# ───────────────────────── TS-momentum 4h crypto book → daily ─────────────────────────
def _resample_4h(df5):
    d = df5.copy(); d["timestamp"] = pd.to_datetime(d["timestamp"])
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    d = d.set_index("timestamp").resample("4h").agg(agg).dropna(subset=["close"])
    return d.reset_index()


def ts_momentum_daily(symbols, start_year=2022, convex_exits=False):
    """Daily net-return series of the 4h ADX-gated TS-momentum book (compounded 4h→daily).

    Built timestamp-indexed (not bar-index): each symbol's per-bar net return is a Series on its
    own 4h clock; the equal-weight book = row-mean across symbols (outer-join, skip not-yet-live),
    then the SAME risk overlay the portfolio backtester applies (vol-target + drawdown de-gear).
    This keeps wall-clock alignment so the daily resample is correct.

    ``convex_exits``: A/B the asymmetric stop (tight initial → wide trail floored at breakeven)
    against the validated symmetric Chandelier."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("pretrain_cb", str(ROOT / "scripts" / "pretrain.py"))
    pre = importlib.util.module_from_spec(spec); spec.loader.exec_module(pre)
    strat = TSMomentumBreakout(TSMomentumParams(
        entry_channel=48, exit_channel=24, atr_mult=3.0, ema_trend=50, adx_min=25, allow_short=True,
        convex_exits=convex_exits, initial_atr_mult=1.5, trail_atr_mult=4.0, activation_atr=1.0))
    ts_cost = (10.0 + 5.0) / 1e4
    series = {}
    for s in symbols:
        try:
            df = _resample_4h(pre.load_full_history(s, start_year, 1, True)["5m"])
        except Exception as e:
            print(f"  ts skip {s}: {str(e)[:50]}"); continue
        pos = strat.generate_positions({s: df})[s]
        ret = _bar_returns(df["close"].to_numpy())
        m = min(len(pos), len(ret)); pos, ret = pos[:m], ret[:m]
        gross = np.zeros(m); gross[1:] = pos[:-1] * ret[1:]
        turn = np.zeros(m); turn[0] = abs(pos[0]); turn[1:] = np.abs(pos[1:] - pos[:-1])
        net = gross - turn * ts_cost
        series[s] = pd.Series(net, index=pd.to_datetime(df["timestamp"].to_numpy()[:m]))
    if len(series) < 4:
        raise RuntimeError("too few TS-momentum symbols cached")
    book = pd.DataFrame(series).sort_index().mean(axis=1, skipna=True)   # equal-weight, wall-clock
    arr = book.to_numpy()
    arr = arr * vol_target_scale(arr, 0.12, 6 * 365, max_leverage=2.0)   # same overlay as backtester
    arr = drawdown_degear(arr, 0.15, 0.25)
    s4 = pd.Series(arr, index=book.index)
    daily = (1.0 + s4).resample("1D").prod() - 1.0                       # compound 4h → calendar-daily
    return daily.dropna()


def _to_calendar(daily_ret: pd.Series, calendar: pd.DatetimeIndex) -> pd.Series:
    """Re-express a calendar-daily return series onto `calendar` (the equity trading days) by
    compounding the between-date returns — weekend crypto moves fold into the next trading day,
    so nothing is dropped. Causal (uses only past/closing values)."""
    eq = (1.0 + daily_ret).cumprod()
    eq = eq.reindex(eq.index.union(calendar)).ffill()
    sampled = eq.reindex(calendar).ffill()
    return sampled.pct_change().dropna()


def _stats(r: np.ndarray, ppy=PPY):
    r = np.asarray(r); r = r[np.isfinite(r)]; sd = r.std()
    eq = np.cumprod(1 + r); peak = np.maximum.accumulate(eq); dd = float(((eq - peak) / peak).min())
    return dict(sharpe=(r.mean() / sd * np.sqrt(ppy)) if sd > 1e-12 else float("nan"),
                ann_ret=r.mean() * ppy, ann_vol=sd * np.sqrt(ppy), maxdd=dd, n=len(r))


def _line(label, r):
    s = _stats(r)
    print(f"  {label:30s} Sharpe {s['sharpe']:+.2f}   CAGR {s['ann_ret']*100:+6.1f}%   "
          f"vol {s['ann_vol']*100:5.1f}%   maxDD {s['maxdd']*100:+6.1f}%   n={s['n']}")
    return s


def _skew_stats(r) -> dict:
    """Return-distribution stats that matter for a 'small red / big green' objective."""
    import numpy as np
    x = np.asarray(r.dropna(), dtype=float)
    if x.size < 20:
        return {}
    mu, sd = x.mean(), x.std(ddof=1)
    sharpe = (mu / sd * np.sqrt(252)) if sd > 0 else 0.0
    skew = float(((x - mu) ** 3).mean() / (sd ** 3)) if sd > 0 else 0.0
    pos, neg = x[x > 0].sum(), -x[x < 0].sum()
    gain_pain = float(pos / neg) if neg > 0 else float("inf")
    eq = np.cumprod(1 + x)
    dd = float((eq / np.maximum.accumulate(eq) - 1.0).min())
    avg_win = float(x[x > 0].mean()) if (x > 0).any() else 0.0
    avg_loss = float(x[x < 0].mean()) if (x < 0).any() else 0.0
    return {"sharpe": sharpe, "skew": skew, "gain_pain": gain_pain, "max_dd": dd,
            "avg_win": avg_win, "avg_loss": avg_loss,
            "win_loss_ratio": (avg_win / -avg_loss) if avg_loss < 0 else float("inf")}


def _print_ab_skew(label, r_symmetric, r_convex):
    """Print a symmetric-vs-convex A/B focused on skew / asymmetry, not just Sharpe."""
    a, b = _skew_stats(r_symmetric), _skew_stats(r_convex)
    if not a or not b:
        print(f"  [A/B {label}] insufficient data"); return
    print(f"\n  === {label}: SYMMETRIC vs CONVEX exits ===")
    print(f"  {'metric':16s} {'symmetric':>12s} {'convex':>12s}")
    for k, name in (("sharpe", "Sharpe"), ("skew", "skew"), ("gain_pain", "gain/pain"),
                    ("win_loss_ratio", "avg win/loss"), ("max_dd", "max DD")):
        print(f"  {name:16s} {a[k]:>12.3f} {b[k]:>12.3f}")
    print("  (convex WINS if skew ↑, gain/pain ↑, avg-win/loss ↑ — accepting a slightly lower "
          "hit-rate. Sharpe may move either way; the goal is asymmetry, not smoothness.)")


def main():
    import argparse
    ap = argparse.ArgumentParser(description="combined-book diversification test")
    ap.add_argument("--with-fxc", action="store_true",
                    help="add the FX/commodity daily TSMOM sleeve as a 4th candidate")
    ap.add_argument("--fxc-download", action="store_true",
                    help="allow yfinance download of any missing FX/commodity ETFs (else cache-only)")
    ap.add_argument("--convex-exits", action="store_true",
                    help="use asymmetric convex stops (tight initial → wide trail floored at breakeven) "
                         "on the TS sleeve, and print a symmetric-vs-convex A/B")
    args = ap.parse_args()

    print("building managed-beta daily book (widened, equal-weight) ...")
    mb = managed_beta_daily(MB_SYMBOLS)
    print("building TS-momentum 4h crypto book (ADX>=25, L/S) -> daily ...")
    ts = ts_momentum_daily(TS_SYMBOLS, convex_exits=args.convex_exits)
    if args.convex_exits:
        print("  [A/B] also building the SYMMETRIC-stop TS book for comparison ...")
        ts_sym = ts_momentum_daily(TS_SYMBOLS, convex_exits=False)
        _print_ab_skew("TS sleeve", ts_sym, ts)
    print("building daily multi-asset TSMOM long-short book (MOP-style) ...")
    tm = tsmom_daily(MB_SYMBOLS)

    sleeves_raw = {"mb": ("managed-beta (widened EW)", mb),
                   "ts": ("TS-momentum 4h (ADX, L/S)", ts, "calendar"),
                   "tm": ("TSMOM daily multi-asset L/S", tm)}
    if args.with_fxc:
        print("building FX/commodity daily TSMOM long-short book (the diversifier candidate) ...")
        fx = tsmom_daily(FXC_SYMBOLS, skip_download=not args.fxc_download)
        sleeves_raw["fx"] = ("FX/commodity TSMOM L/S", fx)

    # common overlap on the equity trading calendar
    daily_series = [v[1] for v in sleeves_raw.values()]
    lo = max(s.index.min() for s in daily_series)
    hi = min(s.index.max() for s in daily_series)
    cal = mb.index[(mb.index >= lo) & (mb.index <= hi)]
    mb_c = mb.reindex(cal).dropna()
    cal = mb_c.index
    aligned = {}
    for k, v in sleeves_raw.items():
        s = v[1]
        aligned[k] = (_to_calendar(s, cal) if (len(v) > 2 and v[2] == "calendar") else s).reindex(cal)
    joined = pd.DataFrame(aligned).dropna()
    cal = joined.index
    sleeves = {k: (sleeves_raw[k][0], joined[k]) for k in sleeves_raw}

    print(f"\n=== overlap {cal.min().date()} -> {cal.max().date()}  ({len(cal)} trading days) ===")
    print("\nSTANDALONE (over the overlap window):")
    stats = {k: _line(lbl, s.to_numpy()) for k, (lbl, s) in sleeves.items()}

    print("\nPAIRWISE CORRELATION (low/negative = the diversification that lifts book Sharpe):")
    print(joined.corr().round(3).to_string())

    print("\nCOMBINED books (each blend re-scaled to managed-beta's standalone vol for a fair "
          "Sharpe comparison):")
    target_vol = stats["mb"]["ann_vol"]

    def _blend(weights: dict):
        b = sum(w * sleeves[k][1].to_numpy() for k, w in weights.items())
        bv = b.std() * np.sqrt(PPY)
        return b * (target_vol / (bv + 1e-12))

    def _inv_vol(keys):
        inv = {k: 1.0 / (stats[k]["ann_vol"] + 1e-9) for k in keys}
        z = sum(inv.values())
        return {k: v / z for k, v in inv.items()}

    _line("2-sleeve mb+ts 50/50 (current)", _blend({"mb": 0.5, "ts": 0.5}))
    keys3 = ("mb", "ts", "tm")
    _line("3-sleeve equal (mb+ts+tm)", _blend({k: 1 / 3 for k in keys3}))
    if args.with_fxc:
        _line("mb+ts+fx equal", _blend({"mb": 1 / 3, "ts": 1 / 3, "fx": 1 / 3}))
        keys4 = ("mb", "ts", "tm", "fx")
        _line("4-sleeve equal", _blend({k: 0.25 for k in keys4}))
        iv = _inv_vol(keys4)
        _line(f"4-sleeve inverse-vol ({', '.join(f'{k}={v:.2f}' for k, v in iv.items())})", _blend(iv))
        print(f"\nVERDICT (fx sleeve): promote it if a book INCLUDING fx beats the best book "
              f"WITHOUT it at equal vol, AND corr(fx, others) stays <=0.3. The whole thesis of a "
              f"currency/commodity leg is low correlation to equity+crypto beta.")
    else:
        iv = _inv_vol(keys3)
        _line(f"3-sleeve inverse-vol ({', '.join(f'{k}={v:.2f}' for k, v in iv.items())})",
              _blend(iv))
        print(f"\nVERDICT: promote a 3-sleeve book if its Sharpe beats the current 2-sleeve at "
              f"equal vol AND correlations stay <=0.3. Re-run with --with-fxc to test the "
              f"FX/commodity diversifier.")


if __name__ == "__main__":
    main()
