#!/usr/bin/env python
"""Joint risk-stack calibration — the realized-PROFIT-per-time lever.

The book's expected return is Sharpe × vol: the Sharpe comes from the sleeves and their
low correlation, but the VOL (and therefore the profit per unit time) is set by the
risk stack — book vol-target, leverage cap, drawdown de-gear. Each layer was validated
alone; stacked they may systematically under-invest. This sweeps the BOOK-level knobs
jointly and reports, for each config, the number that actually matters:

    realized CAGR   subject to   maxDD ≥ -budget   AND   Sharpe > benchmark buy-&-hold

Usage (all data must already be cached — the sleeves load with skip_download):
    python scripts/risk_stack_sweep.py                       # defaults: 3-sleeve book
    python scripts/risk_stack_sweep.py --dd-budget 0.25 --weights mb=0.5,ts=0.5
    python scripts/risk_stack_sweep.py --benchmark QQQ

Caveat printed with the results: leverage above ~1× is NOT free live (perp funding /
margin interest). The sweep charges no explicit financing, so treat high-leverage rows
as upper bounds and prefer the lowest leverage that hits the DD budget.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "scripts"))

if hasattr(sys.stdout, "reconfigure"):          # Windows cp1252 console → UTF-8
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from backend.backtest.portfolio import vol_target_scale, drawdown_degear, _bar_returns  # noqa: E402
from backend.data.equity_daily import load_daily                                        # noqa: E402
import combined_book_research as cbr                                                    # noqa: E402

PPY = cbr.PPY


def _stats(r: np.ndarray, ppy: float = PPY) -> dict:
    r = np.asarray(r, float)
    r = r[np.isfinite(r)]
    sd = float(r.std())
    eq = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(eq)
    dd = float(((eq - peak) / peak).min()) if len(eq) else 0.0
    n = len(r)
    cagr = float(eq[-1] ** (ppy / n) - 1.0) if n and eq[-1] > 0 else 0.0
    return dict(sharpe=(r.mean() / sd * np.sqrt(ppy)) if sd > 1e-12 else float("nan"),
                cagr=cagr, ann_vol=sd * np.sqrt(ppy), maxdd=dd, n=n)


def build_sleeves(weights: dict) -> pd.Series:
    """Blend the requested sleeves (each already internally constructed exactly as
    validated) into ONE raw daily return series on the common calendar."""
    series = {}
    if weights.get("mb"):
        print("building managed-beta sleeve ...")
        series["mb"] = cbr.managed_beta_daily(cbr.MB_SYMBOLS)
    if weights.get("ts"):
        print("building 4h crypto TS-momentum sleeve ...")
        series["ts"] = cbr.ts_momentum_daily(cbr.TS_SYMBOLS)
    if weights.get("tm"):
        print("building daily multi-asset TSMOM sleeve ...")
        series["tm"] = cbr.tsmom_daily(cbr.MB_SYMBOLS)
    if not series:
        raise SystemExit("no sleeves requested")

    lo = max(s.index.min() for s in series.values())
    hi = min(s.index.max() for s in series.values())
    base = series.get("mb", next(iter(series.values())))
    cal = base.index[(base.index >= lo) & (base.index <= hi)]
    aligned = {}
    for k, s in series.items():
        if k == "ts":                                   # 4h book → equity calendar
            aligned[k] = cbr._to_calendar(s, cal)
        else:
            aligned[k] = s.reindex(cal)
    df = pd.DataFrame(aligned).dropna()
    blend = sum(float(weights[k]) * df[k] for k in df.columns)
    return blend


def benchmark_series(ticker: str, cal: pd.DatetimeIndex) -> np.ndarray:
    data = load_daily([ticker], start="2010-01-01", skip_download=True)
    if ticker not in data:
        raise SystemExit(f"benchmark {ticker} not in the daily cache — run a loader first")
    d = data[ticker]
    r = pd.Series(_bar_returns(d["close"].to_numpy()),
                  index=pd.to_datetime(d["timestamp"].to_numpy()))
    return r.reindex(cal).fillna(0.0).to_numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default="mb=0.5,ts=0.5",
                    help="sleeve capital weights, e.g. 'mb=0.5,ts=0.5' (mb/ts/tm). tm (daily "
                         "multi-asset TSMOM L/S) tested ~0 Sharpe over 16y — library only.")
    ap.add_argument("--dd-budget", type=float, default=0.30,
                    help="max acceptable |maxDD| for the constrained pick (default 0.30)")
    ap.add_argument("--benchmark", default="SPY", help="buy-&-hold benchmark ticker (default SPY)")
    ap.add_argument("--vols", default="0.10,0.15,0.20,0.25,0.30",
                    help="book target ann-vol grid")
    ap.add_argument("--levs", default="1.5,2.0,3.0", help="max-leverage grid")
    ap.add_argument("--degears", default="0.15:0.25,0.25:0.25,off",
                    help="drawdown-de-gear grid as thresh:floor pairs (or 'off')")
    ap.add_argument("--financing-rate", type=float, default=0.08,
                    help="annual financing rate charged on the levered portion max(L-1,0) — "
                         "margin interest / typical perp funding drag (default 8%%/yr; 0 disables)")
    args = ap.parse_args()

    weights = {}
    for part in args.weights.split(","):
        k, v = part.split("=")
        weights[k.strip()] = float(v)

    blend = build_sleeves(weights)
    cal = blend.index
    raw = blend.to_numpy()
    bench = benchmark_series(args.benchmark, cal)
    sb = _stats(bench)
    print(f"\n=== window {cal.min().date()} -> {cal.max().date()}  ({len(cal)} trading days) ===")
    print(f"benchmark {args.benchmark} buy-&-hold: Sharpe {sb['sharpe']:+.2f}  "
          f"CAGR {sb['cagr']*100:+.1f}%  maxDD {sb['maxdd']*100:+.1f}%")
    s_raw = _stats(raw)
    print(f"raw sleeve blend (no book overlay): Sharpe {s_raw['sharpe']:+.2f}  "
          f"CAGR {s_raw['cagr']*100:+.1f}%  vol {s_raw['ann_vol']*100:.1f}%  "
          f"maxDD {s_raw['maxdd']*100:+.1f}%")

    vols = [float(x) for x in args.vols.split(",")]
    levs = [float(x) for x in args.levs.split(",")]
    degears = []
    for part in args.degears.split(","):
        if part.strip().lower() == "off":
            degears.append(None)
        else:
            t, f = part.split(":")
            degears.append((float(t), float(f)))

    rows = []
    fin_daily = float(args.financing_rate) / PPY
    for tv in vols:
        for ml in levs:
            lev = vol_target_scale(raw, tv, PPY, max_leverage=ml)
            # financing: the levered portion max(L−1, 0) pays the rate every bar — this is
            # what makes high-leverage rows honest instead of upper bounds.
            net_vt = raw * lev - np.maximum(lev - 1.0, 0.0) * fin_daily
            for dg in degears:
                net = drawdown_degear(net_vt, *dg) if dg else net_vt
                s = _stats(net)
                rows.append({"target_vol": tv, "max_lev": ml,
                             "degear": (f"{dg[0]:.2f}/{dg[1]:.2f}" if dg else "off"),
                             "sharpe": s["sharpe"], "cagr": s["cagr"],
                             "ann_vol": s["ann_vol"], "maxdd": s["maxdd"],
                             "mean_lev": float(np.mean(lev))})

    df = pd.DataFrame(rows)
    with pd.option_context("display.float_format", lambda v: f"{v:+.3f}"):
        print("\nFULL GRID (sorted by CAGR):")
        print(df.sort_values("cagr", ascending=False).to_string(index=False))

    ok = df[(df["maxdd"] >= -abs(args.dd_budget)) & (df["sharpe"] > sb["sharpe"])]
    print(f"\nCONSTRAINED PICK  (maxDD within -{abs(args.dd_budget):.0%}  AND  Sharpe > "
          f"{args.benchmark} {sb['sharpe']:+.2f}):")
    if ok.empty:
        print("  none — loosen the DD budget or improve the sleeve mix first.")
    else:
        best = ok.sort_values("cagr", ascending=False).iloc[0]
        print(f"  target_vol={best.target_vol:.2f}  max_lev={best.max_lev:.1f}  "
              f"degear={best.degear}  →  CAGR {best.cagr*100:+.1f}%  "
              f"Sharpe {best.sharpe:+.2f}  maxDD {best.maxdd*100:+.1f}%  "
              f"(mean leverage {best.mean_lev:.2f})")
        print(f"  Financing charged at {args.financing_rate:.0%}/yr on the levered portion "
              f"max(L-1,0). Live, the smoothed per-symbol funding cap (backend/risk/funding_cap.py) "
              f"additionally de-levers names on the paying side of sustained funding.")


if __name__ == "__main__":
    main()
