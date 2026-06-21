#!/usr/bin/env python
"""Walk-forward backtest of the STRATEGY BOOK - the promotion gate for systematic strategies.

Loads cached OHLCV (via the existing pretrain loader), resamples to the strategy timeframe,
runs the strategy portfolio through the cost-aware multi-strategy backtester, and reports the
numbers a quant actually trusts: per-fold Sharpe (regime robustness), full-period Sharpe /
regression alpha-beta / max-drawdown / turnover, the DEFLATED Sharpe (multiple-testing
honesty), and the cross-strategy correlation matrix (the diversification that makes the book).

    python scripts/backtest_portfolio.py --start-year 2022 --skip-download \
        --symbols "BTCUSDT ETHUSDT SOLUSDT XRPUSDT ADAUSDT DOGEUSDT" --folds 4 --trials 20
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

_spec = importlib.util.spec_from_file_location("pretrain_pb_mod", str(ROOT / "scripts" / "pretrain.py"))
pre = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pre)

from backend.backtest.portfolio import portfolio_backtest, align_panel  # noqa: E402
from backend.backtest.engine import deflated_sharpe_ratio  # noqa: E402
from backend.strategies.ts_momentum import TSMomentumBreakout, TSMomentumParams  # noqa: E402
from backend.strategies.cross_sectional import (  # noqa: E402
    CrossSectionalMomentum, XSectionalMomentumParams,
)
from backend.strategies.stat_arb import (  # noqa: E402
    StatArbPairs, StatArbParams, find_cointegrated_pairs,
)
from backend.strategies.mean_reversion import MeanReversion, MeanReversionParams  # noqa: E402

TF_BARS_PER_YEAR = {"5m": 105_120, "15m": 35_040, "1h": 24 * 365, "4h": 6 * 365}
# pandas offset aliases (pandas 2.x: minutes are "min", not "m" — "m" is month-end!)
_PANDAS_RULE = {"15m": "15min", "1h": "1h", "4h": "4h"}
_AGG = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}


def _resample(df5: pd.DataFrame, rule: str) -> pd.DataFrame:
    d = df5.copy()
    d["timestamp"] = pd.to_datetime(d["timestamp"])
    d = d.set_index("timestamp").resample(_PANDAS_RULE.get(rule, rule)).agg(_AGG).dropna(subset=["close"])
    return d.reset_index()


def load_data(symbols, start_year, start_month, skip_download, rule):
    data = {}
    for s in symbols:
        try:
            dfs = pre.load_full_history(s, start_year, start_month, skip_download)
            data[s] = _resample(dfs["5m"], rule) if rule != "5m" else dfs["5m"]
        except Exception as e:
            print(f"  skip {s}: {str(e)[:60]}")
    return data


def main():
    ap = argparse.ArgumentParser(description="Walk-forward backtest of the strategy book")
    ap.add_argument("--symbols", default="BTCUSDT ETHUSDT SOLUSDT XRPUSDT ADAUSDT DOGEUSDT")
    ap.add_argument("--start-year", type=int, default=2022)
    ap.add_argument("--start-month", type=int, default=1)
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--timeframe", default="1h", choices=list(TF_BARS_PER_YEAR))
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--oos-frac", type=float, default=0.5, help="tail fraction split into folds")
    ap.add_argument("--trials", type=int, default=20, help="configs tried (for Deflated Sharpe)")
    ap.add_argument("--fee-bps", type=float, default=10.0)
    ap.add_argument("--slippage-bps", type=float, default=5.0)
    ap.add_argument("--target-vol", type=float, default=0.12)
    # TS-momentum knobs (so you can sweep without code edits)
    ap.add_argument("--entry-channel", type=int, default=48)
    ap.add_argument("--exit-channel", type=int, default=24)
    ap.add_argument("--atr-mult", type=float, default=3.0)
    ap.add_argument("--ema-trend", type=int, default=100)
    ap.add_argument("--adx-min", type=float, default=0.0,
                    help="only enter momentum when ADX >= this (regime gate; e.g. 25). 0 = off.")
    ap.add_argument("--no-short", action="store_true")
    # which strategies are in the book + cross-sectional knobs
    ap.add_argument("--strategy", default="ts_momentum",
                    choices=["ts_momentum", "xs_momentum", "both", "stat_arb", "mean_reversion"])
    ap.add_argument("--xs-lookback", type=int, default=168)
    ap.add_argument("--xs-hold", type=int, default=24)
    ap.add_argument("--mr-adx-max", type=float, default=20.0,
                    help="mean-reversion: only trade when ADX < this (ranging regime)")
    ap.add_argument("--mr-entry-z", type=float, default=2.0)
    # stat-arb knobs (stocks): cointegration selection on a training prefix, then z-score reversion
    ap.add_argument("--coint-pmax", type=float, default=0.05, help="max Engle-Granger p-value to trade a pair")
    ap.add_argument("--max-pairs", type=int, default=10)
    ap.add_argument("--entry-z", type=float, default=2.0)
    ap.add_argument("--exit-z", type=float, default=0.5)
    ap.add_argument("--max-half-life", type=float, default=240.0,
                    help="stat-arb: only fade a spread reverting within this many bars (0 = gate off)")
    args = ap.parse_args()

    ppy = TF_BARS_PER_YEAR[args.timeframe]
    requested = args.symbols.split()
    data = load_data(requested, args.start_year, args.start_month,
                     args.skip_download, args.timeframe)
    if not data:
        print("no data"); return
    if len(data) < len(requested):
        print(f"\n!  only {len(data)}/{len(requested)} symbols loaded - the rest are not cached. "
              f"Drop --skip-download (or prefetch) to download them. Momentum/cross-sectional are "
              f"BREADTH strategies; too few symbols gives an unreliable, usually-negative result.")
    if args.strategy in ("xs_momentum", "both") and len(data) < 4:
        print("!  cross-sectional momentum needs >=4 symbols to form a long/short book - with fewer "
              "it stays FLAT (the all-zero result you'd see is an abstention, not a strategy verdict).")
    strategies = {}
    if args.strategy in ("ts_momentum", "both"):
        strategies["ts_momentum"] = TSMomentumBreakout(TSMomentumParams(
            entry_channel=args.entry_channel, exit_channel=args.exit_channel,
            atr_mult=args.atr_mult, ema_trend=args.ema_trend, adx_min=args.adx_min,
            allow_short=not args.no_short))
    if args.strategy in ("xs_momentum", "both"):
        strategies["xs_momentum"] = CrossSectionalMomentum(XSectionalMomentumParams(
            lookback=args.xs_lookback, hold=args.xs_hold))
        data = align_panel(data)    # cross-sectional ranking requires aligned timestamps
        print(f"(aligned {len(data)} symbols to a common timeline for cross-sectional ranking)")
    if args.strategy == "stat_arb":
        data = align_panel(data)    # pairs need a common timeline
        pairs = find_cointegrated_pairs(data, train_frac=0.5, pmax=args.coint_pmax,
                                        max_pairs=args.max_pairs)
        if not pairs:
            print("no cointegrated pairs found (try a wider universe / higher --coint-pmax)"); return
        print(f"cointegrated pairs (selected on the first 50% - no look-ahead): {pairs}")
        strategies["stat_arb"] = StatArbPairs(pairs, StatArbParams(
            entry_z=args.entry_z, exit_z=args.exit_z, max_half_life=args.max_half_life))
    if args.strategy == "mean_reversion":
        strategies["mean_reversion"] = MeanReversion(MeanReversionParams(
            entry_z=args.mr_entry_z, adx_max=args.mr_adx_max, allow_short=not args.no_short))
    market = "BTCUSDT" if "BTCUSDT" in data else None

    res = portfolio_backtest(strategies, data, fee_bps=args.fee_bps, slippage_bps=args.slippage_bps,
                             target_ann_vol=args.target_vol, bars_per_year=ppy, market_symbol=market)
    net = res.net_returns
    print(f"\nBook: {list(strategies)} | {args.timeframe} | {len(data)} symbols | {len(net)} bars\n")

    # per-fold Sharpe over the OOS tail -> regime robustness
    n = len(net)
    edges = np.linspace(int(n * (1.0 - args.oos_frac)), n, args.folds + 1).astype(int)
    fold_sh = []
    for k in range(args.folds):
        a, b = int(edges[k]), int(edges[k + 1])
        seg = net[a:b]
        sh = float(seg.mean() / (seg.std() + 1e-12) * np.sqrt(ppy)) if seg.size > 2 else 0.0
        fold_sh.append(sh)
    pos = sum(1 for s in fold_sh if s > 0)

    m = res.metrics
    sr_pp = float(net.mean() / (net.std() + 1e-12))           # per-period SR for DSR
    dsr = deflated_sharpe_ratio(sr_pp, n_obs=n, n_trials=max(1, args.trials),
                                skew=float(pd.Series(net).skew()),
                                kurt=float(pd.Series(net).kurt() + 3.0))

    print(f"{'per-fold Sharpe':22s} " + "  ".join(f"{s:+.2f}" for s in fold_sh) +
          f"   ({pos}/{args.folds} positive)")
    print(f"{'full Sharpe':22s} {m['sharpe']:+.2f}")
    print(f"{'regression alpha (ann)':22s} {m['reg_alpha']:+.3f}")
    print(f"{'beta vs ' + str(market):22s} {m['beta']:+.2f}")
    print(f"{'max drawdown':22s} {m['max_drawdown']*100:+.1f}%")
    print(f"{'ann return / vol':22s} {m['ann_return']*100:+.1f}% / {m['ann_vol']*100:.1f}%")
    print(f"{'turnover (sum)':22s} {m.get('turnover', float('nan')):.0f}")
    print(f"{'mean leverage':22s} {m['leverage_mean']:.2f}")
    print(f"{'DEFLATED Sharpe':22s} {dsr:.3f}   (P[true SR>0] after {args.trials} trials)")
    if res.strategy_metrics:
        print("\nper-strategy (standalone, pre-blend):")
        for nm, sm in res.strategy_metrics.items():
            print(f"  {nm:16s} Sharpe={sm['sharpe']:+.2f}  ann={sm['ann_return']*100:+.1f}%  "
                  f"maxDD={sm['max_drawdown']*100:+.1f}%")
    if not res.correlation.empty:
        print("\nstrategy correlation (low = good diversification):\n" + res.correlation.round(2).to_string())

    print("\nPROMOTE if: per-fold Sharpe positive in MOST folds, regression alpha > 0 with |beta| "
          "controlled, AND Deflated Sharpe > ~0.95. Otherwise iterate params / add diversifiers.")


if __name__ == "__main__":
    main()
