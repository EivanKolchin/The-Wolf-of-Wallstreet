#!/usr/bin/env python
"""Market-neutral cross-sectional crypto — the one approach that dodges the failure mode.

Every directional model we tried died the same way: it bets on "will price go up?", learns a bias
from the training regime, and is wrong when the regime flips (walk-forward killed the linear model).
A DOLLAR-NEUTRAL long-short book sidesteps that: each rebalance it ranks the alts by a relative
signal and goes LONG the strongest k / SHORT the weakest k in equal dollars. If the whole market
drops it can still profit (longs drop less than shorts) — it trades RELATIVE strength, which is far
more regime-stable than absolute direction. No model training, no new data.

Reports the portfolio Sharpe (market-neutral → Sharpe IS the skill; there's no beta to subtract)
across WALK-FORWARD time folds. Sweeps signal direction (momentum vs reversal), lookback, hold.

    python scripts/cross_sectional.py --skip-download
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

_spec = importlib.util.spec_from_file_location("pretrain_xs_mod", str(ROOT / "scripts" / "pretrain.py"))
pre = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pre)

BARS_PER_YEAR_5M = 12 * 24 * 365  # 105120
CRYPTO = "BTCUSDT ETHUSDT SOLUSDT AAVEUSDT XLMUSDT XRPUSDT ADAUSDT DOGEUSDT RENDERUSDT NEARUSDT"


def build_closes(symbols, start_year, start_month, skip_download):
    series = {}
    for sym in symbols:
        try:
            df = pre.load_full_history(sym, start_year, start_month, skip_download)["5m"]
            s = df.drop_duplicates("timestamp").set_index("timestamp")["close"].astype(float)
            series[sym] = s
            print(f"  loaded {sym}")
        except Exception as e:
            print(f"  skip {sym}: {str(e)[:40]}")
    closes = pd.DataFrame(series).sort_index()
    return closes


def portfolio_net(closes, lookback, hold, k, kind, fee_bps=15.0):
    """Dollar-neutral long-short net per-bar returns. Rebalance every `hold` bars; long the top-k /
    short the bottom-k by trailing-`lookback` return (momentum), or the reverse (reversal)."""
    ret1 = closes.pct_change()
    signal = closes.pct_change(lookback)
    n = len(closes)
    W = pd.DataFrame(np.nan, index=closes.index, columns=closes.columns)
    for i in range(lookback, n, hold):
        s = signal.iloc[i].dropna()
        if len(s) < 2 * k:
            continue
        ranked = s.sort_values().index            # ascending: weakest first
        longs, shorts = ranked[-k:], ranked[:k]
        if kind == "reversal":
            longs, shorts = shorts, longs
        w = pd.Series(0.0, index=closes.columns)
        w[longs] = 1.0 / k
        w[shorts] = -1.0 / k
        W.iloc[i] = w.values
    W = W.ffill().fillna(0.0)
    gross = (W.shift(1) * ret1).sum(axis=1)
    turnover = W.diff().abs().sum(axis=1)         # nonzero only at rebalances
    return gross - turnover * (fee_bps / 1e4)


def _sharpe(net):
    net = net.dropna().to_numpy()
    if net.size < 50 or net.std() < 1e-12:
        return 0.0
    return float(net.mean() / net.std() * np.sqrt(BARS_PER_YEAR_5M))


def main():
    ap = argparse.ArgumentParser(description="Walk-forward market-neutral cross-sectional crypto")
    ap.add_argument("--symbols", default=CRYPTO)
    ap.add_argument("--start-year", type=int, default=2022)
    ap.add_argument("--start-month", type=int, default=1)
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--k", type=int, default=3, help="names long and short (k each side)")
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    closes = build_closes(args.symbols.split(), args.start_year, args.start_month, args.skip_download)
    closes = closes.dropna(how="all")
    n = len(closes)
    print(f"\naligned {closes.shape[1]} symbols x {n} bars   (k={args.k} per side, dollar-neutral)\n")

    # configs: (signal kind, lookback bars, hold bars)
    configs = [("momentum", 12, 12), ("reversal", 12, 12), ("momentum", 48, 48),
               ("reversal", 48, 48), ("momentum", 288, 48), ("reversal", 288, 48)]
    edges = np.linspace(0.0, 1.0, args.folds + 1)
    print(f"{'signal':9s} {'look':>5s} {'hold':>5s} | " +
          " ".join(f"f{j}" for j in range(args.folds)) + " |   ALL  POSfolds")
    print("-" * 78)
    for kind, look, hold in configs:
        net = portfolio_net(closes, look, hold, args.k, kind)
        fold_sh = []
        for j in range(args.folds):
            a, b = int(edges[j] * n), int(edges[j + 1] * n)
            fold_sh.append(_sharpe(net.iloc[a:b]))
        allsh = _sharpe(net)
        pos = sum(1 for s in fold_sh if s > 0)
        cells = " ".join(f"{s:>5.1f}" for s in fold_sh)
        print(f"{kind:9s} {look:>5d} {hold:>5d} | {cells} | {allsh:>5.1f}   {pos}/{args.folds}")

    print("\nMarket-neutral → Sharpe IS the edge (no beta). Want POSITIVE Sharpe in MOST folds (robust")
    print("across regimes). If a config is positive in 4-5/5 folds → real market-neutral edge → build")
    print("that into the crypto specialist (rank-and-trade) + portfolio manager. If all flip sign by")
    print("fold → crypto has no robust cross-sectional edge either → pivot to events / new data.")


if __name__ == "__main__":
    main()
