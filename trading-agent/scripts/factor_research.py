#!/usr/bin/env python
"""Daily cross-sectional FACTOR research — testing the decades-robust equity anomalies at their
NATIVE horizon (monthly rebalance), on a broad, long-history universe. This is the "expand the
opportunity set" probe: the intraday edges were competed away, so go where the documented edges
actually live (Jegadeesh-Titman 12-1 momentum, short-term reversal, …) with real breadth + depth.

Each factor is a dollar-neutral cross-sectional book (β≈0); scored on walk-forward folds + the
Deflated Sharpe (multiple-testing-honest). Realistic expectation: modest Sharpe (~0.3-0.7) per
factor with deep drawdowns, but DECADES-robust — and a combination of low-correlation factors is
the textbook path to a respectable portfolio Sharpe.

    python scripts/factor_research.py --universe sp500 --start 2005-01-01 --folds 5 --trials 30
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
from backend.backtest.portfolio import portfolio_backtest, align_panel  # noqa: E402
from backend.backtest.engine import deflated_sharpe_ratio  # noqa: E402
from backend.strategies.cross_sectional import (  # noqa: E402
    CrossSectionalMomentum, XSectionalMomentumParams, LowVolatility, LowVolParams,
)

DAILY_BARS_PER_YEAR = 252


def _factors() -> dict:
    """The classic cross-sectional equity factors (daily bars, monthly-ish rebalance)."""
    return {
        # 12-1 momentum: rank by the past ~12mo return SKIPPING the last month (avoids 1mo reversal)
        "momentum_12_1": CrossSectionalMomentum(XSectionalMomentumParams(
            lookback=252, skip=21, hold=21, quantile=0.2, kind="momentum")),
        # low-volatility: long lowest-trailing-vol / short highest (price-only, low corr to momentum)
        "low_vol": LowVolatility(LowVolParams(vol_window=60, hold=21, quantile=0.2)),
        # short-term (1-month) reversal: long last month's LOSERS / short the winners
        "reversal_1m": CrossSectionalMomentum(XSectionalMomentumParams(
            lookback=21, skip=0, hold=5, quantile=0.2, kind="reversal")),
    }


# Factors combined into the BOOK (positive-expectation + diversifying). On the REAL 502-name
# S&P 500 (2005-) the standout was short-term REVERSAL (5/6 folds, DSR 0.35), NOT low-vol
# (0/6 folds, DSR 0 — dropped) — the inverse of the narrow 64-name finding. The book pairs the
# two surviving, different-horizon factors (12-1 momentum + 1-month reversal); they trade opposite
# ends of the momentum spectrum, so the run prints their correlation to confirm diversification.
COMBINED_FACTORS = ["momentum_12_1", "reversal_1m"]


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
    print(f"  reg alpha / beta: {m['reg_alpha']:+.3f} / {m['beta']:+.2f}   maxDD {m['max_drawdown']*100:+.1f}%")
    print(f"  DEFLATED Sharpe : {dsr:.3f}   (P[true SR>0] after {trials} trials)")


def main():
    ap = argparse.ArgumentParser(description="Daily cross-sectional factor research")
    ap.add_argument("--universe", default="sp500", choices=["sp500", "fallback"])
    ap.add_argument("--start", default="2005-01-01")
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--oos-frac", type=float, default=0.5)
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--fee-bps", type=float, default=2.0)
    ap.add_argument("--slippage-bps", type=float, default=3.0)
    args = ap.parse_args()

    tickers = sp500_tickers() if args.universe == "sp500" else FALLBACK_UNIVERSE
    data = load_daily(tickers, start=args.start, skip_download=args.skip_download)
    if len(data) < 10:
        print(f"only {len(data)} tickers loaded — need a broad universe (drop --skip-download to fetch)")
        return
    data = align_panel(data, how="outer")     # staggered IPO dates → union + NaN-pad
    n = len(next(iter(data.values())))
    print(f"\nUniverse: {len(data)} names | daily | {n} bars (~{n/252:.1f} yrs from {args.start})")

    factors = _factors()
    ppy = DAILY_BARS_PER_YEAR
    for name, strat in factors.items():
        res = portfolio_backtest({name: strat}, data, fee_bps=args.fee_bps,
                                 slippage_bps=args.slippage_bps, bars_per_year=ppy, market_symbol=None)
        _report(name, res, args.folds, args.oos_frac, args.trials, ppy)

    book = {k: factors[k] for k in COMBINED_FACTORS if k in factors}
    res = portfolio_backtest(book, data, fee_bps=args.fee_bps, slippage_bps=args.slippage_bps,
                             bars_per_year=ppy, market_symbol=None)
    _report(f"COMBINED book {list(book)}", res, args.folds, args.oos_frac, args.trials, ppy)
    if not res.correlation.empty:
        print("\nfactor correlation (low = good diversification):\n" + res.correlation.round(2).to_string())
    print("\nPROMOTE a factor only if: most folds positive, reg alpha>0 with |beta|≈0, Deflated Sharpe>~0.95.")
    print("These factors are decades-robust in the literature; if they DON'T show here, suspect the")
    print("data window / costs / implementation before concluding they're gone.")


if __name__ == "__main__":
    main()
