#!/usr/bin/env python
"""Fundamental FACTOR research — value & quality, the alpha lever price-only factors couldn't reach.

The price-only daily factors (momentum/reversal/low-vol) had ~zero regression ALPHA on the liquid
S&P 500 — their Sharpe was leaked market beta. Value (cheap on earnings/book) and quality (high
gross profitability / ROE) are the factors the literature credits with genuine, low-momentum-
correlation alpha. They live in fundamentals, so this uses SEC EDGAR point-in-time data (every
figure used only ON/AFTER its filing date → strictly no look-ahead).

Each factor is a dollar-neutral cross-sectional book (β≈0 by construction); scored on walk-forward
folds + the Deflated Sharpe + regression α/β. THE test is the COMBINED book {momentum, reversal,
value, quality}: do the low-correlation fundamental factors lift the book's ALPHA (not just beta)
and its fold consistency? Beta is printed prominently — the last run taught us a dollar-neutral
book can still leak market beta, and beta is not alpha.

    python scripts/fundamental_factor_research.py --universe sp500 --start 2005-01-01 --folds 6
    python scripts/fundamental_factor_research.py --skip-download   # caches only (after one fetch)
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

from backend.data.equity_daily import load_daily, sp500_tickers, FALLBACK_UNIVERSE  # noqa: E402
from backend.data.fundamentals import point_in_time_panel, ticker_to_cik  # noqa: E402
from backend.backtest.portfolio import portfolio_backtest, align_panel  # noqa: E402
from backend.backtest.engine import deflated_sharpe_ratio  # noqa: E402
from backend.strategies.cross_sectional import (  # noqa: E402
    CrossSectionalMomentum, XSectionalMomentumParams, CrossSectionalScore, ScoreParams,
)

DAILY_BARS_PER_YEAR = 252


def _xs_zscore(panel: np.ndarray) -> np.ndarray:
    """Per-bar (cross-sectional) z-score, NaN-safe — puts different-scale ratios on a common scale
    so they can be averaged into a composite. Rows with <4 valid names stay NaN (no rebalance)."""
    p = np.asarray(panel, dtype=np.float64)
    mu = np.nanmean(p, axis=1, keepdims=True)
    sd = np.nanstd(p, axis=1, keepdims=True)
    valid = np.sum(np.isfinite(p), axis=1, keepdims=True) >= 4
    z = np.where((sd > 1e-12) & valid, (p - mu) / sd, np.nan)
    return z


def _safe_ratio(num: np.ndarray, den: np.ndarray, den_positive: bool = True) -> np.ndarray:
    """num/den with the denominator guarded (>0 if ``den_positive`` else !=0); else NaN."""
    ok = (den > 0) if den_positive else (np.abs(den) > 0)
    return np.where(ok & np.isfinite(num), num / np.where(ok, den, np.nan), np.nan)


def build_factor_signals(data: dict, asof: pd.DatetimeIndex, skip_download: bool):
    """Compute point-in-time value & quality signal panels aligned to ``data``'s symbol order."""
    syms = list(data)
    close = np.column_stack([data[s]["close"].to_numpy(np.float64) for s in syms])  # (n, S)
    f = point_in_time_panel(syms, asof, concepts=["net_income", "equity", "assets",
                                                  "gross_profit", "shares"],
                            cik_map=ticker_to_cik(), skip_download=skip_download)
    mcap_raw = close * f["shares"]
    mktcap = np.where((f["shares"] > 0) & np.isfinite(mcap_raw), mcap_raw, np.nan)  # market cap, PIT
    ep = _safe_ratio(f["net_income"], mktcap)          # earnings yield (value)
    bp = _safe_ratio(f["equity"], mktcap)              # book-to-price (value)
    gpa = _safe_ratio(f["gross_profit"], f["assets"])  # gross profitability (quality, Novy-Marx)
    roe = _safe_ratio(f["net_income"], f["equity"])    # return on equity (quality)
    value = np.nanmean(np.stack([_xs_zscore(ep), _xs_zscore(bp)]), axis=0)
    quality = np.nanmean(np.stack([_xs_zscore(gpa), _xs_zscore(roe)]), axis=0)
    coverage = float(np.mean(np.sum(np.isfinite(value), axis=1)[-252:]))   # avg names/bar, last yr
    return syms, {"value": value, "quality": quality, "ep": ep, "bp": bp, "gpa": gpa, "roe": roe}, coverage


def _report(name, res, folds, oos_frac, trials, ppy):
    net = res.net_returns
    n = len(net)
    edges = np.linspace(int(n * (1.0 - oos_frac)), n, folds + 1).astype(int)
    fold_sh = []
    for k in range(folds):
        a, b = int(edges[k]), int(edges[k + 1])
        seg = net[a:b]
        fold_sh.append(float(seg.mean() / (seg.std() + 1e-12) * np.sqrt(ppy)) if seg.size > 2 else 0.0)
    pos = sum(1 for s in fold_sh if s > 0)
    m = res.metrics
    sr_pp = float(net.mean() / (net.std() + 1e-12))
    dsr = deflated_sharpe_ratio(sr_pp, n_obs=n, n_trials=max(1, trials),
                                skew=float(pd.Series(net).skew()), kurt=float(pd.Series(net).kurt() + 3.0))
    print(f"\n=== {name} ===")
    print(f"  per-fold Sharpe : " + "  ".join(f"{s:+.2f}" for s in fold_sh) + f"   ({pos}/{folds} +)")
    print(f"  full Sharpe     : {m['sharpe']:+.2f}   ann {m['ann_return']*100:+.1f}% / vol {m['ann_vol']*100:.1f}%")
    print(f"  reg ALPHA / beta: {m['reg_alpha']*100:+.1f}%/yr / {m['beta']:+.2f}   maxDD {m['max_drawdown']*100:+.1f}%")
    print(f"  DEFLATED Sharpe : {dsr:.3f}   (P[true SR>0] after {trials} trials)")


def main():
    ap = argparse.ArgumentParser(description="Fundamental (value/quality) factor research")
    ap.add_argument("--universe", default="sp500", choices=["sp500", "fallback"])
    ap.add_argument("--start", default="2005-01-01")
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--folds", type=int, default=6)
    ap.add_argument("--oos-frac", type=float, default=0.5)
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--fee-bps", type=float, default=2.0)
    ap.add_argument("--slippage-bps", type=float, default=3.0)
    ap.add_argument("--hold", type=int, default=21)
    args = ap.parse_args()

    tickers = sp500_tickers() if args.universe == "sp500" else FALLBACK_UNIVERSE
    data = load_daily(tickers, start=args.start, skip_download=args.skip_download)
    if len(data) < 10:
        print(f"only {len(data)} tickers loaded — need a broad universe (drop --skip-download to fetch)")
        return
    data = align_panel(data, how="outer")
    n = len(next(iter(data.values())))
    asof = pd.to_datetime(next(iter(data.values()))["timestamp"])
    print(f"\nUniverse: {len(data)} names | daily | {n} bars (~{n/252:.1f} yrs from {args.start})")
    print("Fetching point-in-time fundamentals from SEC EDGAR (cached after first run)...")

    syms, sig, coverage = build_factor_signals(data, asof, args.skip_download)
    print(f"Fundamental coverage: ~{coverage:.0f} names/bar with a value score (last year)")
    ppy = DAILY_BARS_PER_YEAR
    sp = ScoreParams(hold=args.hold, warmup=252, quantile=0.2, higher_is_better=True)

    def score(nm):
        return CrossSectionalScore(sig[nm], syms, name=nm, params=sp)

    strategies = {
        "value": score("value"),
        "quality": score("quality"),
        "momentum_12_1": CrossSectionalMomentum(XSectionalMomentumParams(
            lookback=252, skip=21, hold=args.hold, quantile=0.2, kind="momentum")),
        "reversal_1m": CrossSectionalMomentum(XSectionalMomentumParams(
            lookback=21, skip=0, hold=5, quantile=0.2, kind="reversal")),
    }
    for name, strat in strategies.items():
        res = portfolio_backtest({name: strat}, data, fee_bps=args.fee_bps,
                                 slippage_bps=args.slippage_bps, bars_per_year=ppy, market_symbol=None)
        _report(name, res, args.folds, args.oos_frac, args.trials, ppy)

    # Candidate books — reversal is pure beta with NEGATIVE alpha (it drags the combined alpha to
    # ~0), so isolate the genuinely-neutral positive-alpha factors and see if dropping it pushes the
    # book's reg ALPHA clearly positive while keeping the diversification (DSR/drawdown) gains.
    books = {
        "mom+rev+value+quality (all four)": ["momentum_12_1", "reversal_1m", "value", "quality"],
        "mom+value+quality (drop reversal)": ["momentum_12_1", "value", "quality"],
        "value+quality (pure fundamentals)": ["value", "quality"],
    }
    corr_res = None
    for label, keys in books.items():
        bk = {k: strategies[k] for k in keys}
        res = portfolio_backtest(bk, data, fee_bps=args.fee_bps, slippage_bps=args.slippage_bps,
                                 bars_per_year=ppy, market_symbol=None)
        _report(f"BOOK [{label}]", res, args.folds, args.oos_frac, args.trials, ppy)
        if len(keys) == 4:
            corr_res = res
    if corr_res is not None and not corr_res.correlation.empty:
        print("\nfactor correlation (low = diversifying):\n" + corr_res.correlation.round(2).to_string())
    print("\nPROMOTE only if: most folds +, reg ALPHA>0 with |beta| small, Deflated Sharpe>~0.95.")
    print("KEY QUESTION: does value/quality add ALPHA (not beta) on top of momentum+reversal, and")
    print("lift the combined book's fold consistency? If alpha is still ~0 after fundamentals, the")
    print("honest conclusion is retail-accessible single-factor alpha on liquid US large-caps is gone.")


if __name__ == "__main__":
    main()
