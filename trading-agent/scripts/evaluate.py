#!/usr/bin/env python
"""Walk-forward, cost-aware out-of-sample evaluation — the HONEST promotion gate.

A single OOS backtest (scripts/backtest.py) can be one-regime luck. This evaluates a
FIXED trained checkpoint over several CONSECUTIVE windows of the held-out tail and asks
the only question that matters: is the edge POSITIVE in MOST folds (regime-robust), or
does it flip sign (luck)? For each fold + symbol it reports:

  • net ALPHA  = excess Sharpe vs buy-and-hold (after fees+slippage) — skill, not beta;
  • regression (alpha, beta) of the strategy's net returns on the asset's own returns —
    alpha>0 with |beta|≈0 is genuine timing edge; alpha≈0 with beta≈1 is a long-biased
    model riding a rising market (the trap the analysis warned about).

Nothing should be promoted to live unless median fold alpha > 0 across MOST folds.

    python scripts/evaluate.py --checkpoint models/trading_lstm_latest.pt \
        --start-year 2024 --skip-download --folds 4 --primary-horizon 1
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

# Reuse the EXACT trained-checkpoint loader + feature build + batched probs from the
# single-window backtester so this harness can never drift from it.
_spec_pre = importlib.util.spec_from_file_location("pretrain_eval_mod", str(ROOT / "scripts" / "pretrain.py"))
pre = importlib.util.module_from_spec(_spec_pre)
_spec_pre.loader.exec_module(pre)
_spec_bt = importlib.util.spec_from_file_location("backtest_eval_mod", str(ROOT / "scripts" / "backtest.py"))
bt = importlib.util.module_from_spec(_spec_bt)
_spec_bt.loader.exec_module(bt)

from backend.backtest.engine import (  # noqa: E402
    run_backtest, directional_signal, regression_alpha_beta, BARS_PER_YEAR_5M,
)


def evaluate_symbol(model, sym, args, device):
    """Per-fold (net alpha, regression alpha, beta, #trades) over the OOS tail."""
    feats, close, high, low = bt.build_symbol_features(
        sym, args.start_year, args.start_month, args.skip_download)
    seq_len = pre.SEQ_LEN
    n = len(feats)
    if n <= seq_len + 10:
        return None
    starts = np.arange(0, n - seq_len)            # window k decides bar k+seq_len-1
    close_seq = close[starts + seq_len - 1]
    ppy = bt.BARS_PER_YEAR_STOCK if pre._is_stock_symbol(sym) else BARS_PER_YEAR_5M

    # Evaluate the FIXED checkpoint over `folds` consecutive windows of the last
    # `oos_frac` of the timeline → tests regime stability of one trained model.
    oos0 = int(len(starts) * (1.0 - args.oos_frac))
    edges = np.linspace(oos0, len(starts), args.folds + 1).astype(int)
    folds = []
    for k in range(args.folds):
        a, b = int(edges[k]), int(edges[k + 1])
        if b - a < 5:
            continue
        s_fold = starts[a:b]
        probs = bt.probs_for_starts(model, feats, s_fold, pre.SYMBOL_TO_ID[sym], device,
                                    primary_h=args.primary_horizon)
        sig = directional_signal(probs[:, 0], probs[:, 1], min_confidence=args.min_confidence,
                                 min_edge=args.min_edge, allow_short=not args.long_only)
        c = close_seq[a:b]
        res = run_backtest(c, sig, fee_bps=args.fee_bps, slippage_bps=args.slippage_bps,
                           allow_short=not args.long_only, bars_per_year=ppy)
        bar_ret = np.zeros(len(c)); bar_ret[1:] = c[1:] / np.where(c[:-1] != 0, c[:-1], np.nan) - 1.0
        bar_ret = np.nan_to_num(bar_ret)
        ra, rb = regression_alpha_beta(res.net_returns, bar_ret, bars_per_year=ppy)
        folds.append({
            "alpha": res.metrics.get("excess_sharpe", 0.0),   # excess Sharpe vs B&H
            "reg_alpha": ra, "beta": rb, "trades": int(res.metrics["num_trades"]),
        })
    return folds or None


def main():
    ap = argparse.ArgumentParser(description="Walk-forward OOS evaluation (the promotion gate)")
    ap.add_argument("--checkpoint", default="models/trading_lstm_latest.pt")
    ap.add_argument("--symbols", nargs="+", default=pre.SYMBOLS)
    ap.add_argument("--start-year", type=int, default=2024)
    ap.add_argument("--start-month", type=int, default=1)
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--oos-frac", type=float, default=0.4, help="tail fraction split into folds")
    ap.add_argument("--primary-horizon", type=int, default=1,
                    help=f"index into HORIZONS {pre.HORIZONS} to trade (1 = H+12 / 1h)")
    ap.add_argument("--fee-bps", type=float, default=10.0)
    ap.add_argument("--slippage-bps", type=float, default=5.0)
    ap.add_argument("--min-confidence", type=float, default=0.45)
    ap.add_argument("--min-edge", type=float, default=0.05)
    ap.add_argument("--long-only", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = bt.load_model(args.checkpoint, device)
    hz = pre.HORIZONS[args.primary_horizon]
    print(f"Loaded {args.checkpoint} on {device} | trading H+{hz} (~{hz*5}m) | "
          f"{args.folds} folds over last {args.oos_frac:.0%}\n")
    print(f"{'SYMBOL':10s} {'per-fold net alpha (excess Sharpe)':40s} {'medA':>6s} {'+f':>4s} "
          f"{'regα':>7s} {'beta':>6s}")
    print("-" * 96)

    all_fold_medians, all_betas, all_regalpha = [], [], []
    for sym in args.symbols:
        if sym not in pre.SYMBOL_TO_ID:
            continue
        try:
            folds = evaluate_symbol(model, sym, args, device)
        except Exception as e:
            print(f"{sym:10s} SKIPPED ({str(e)[:60]})")
            continue
        if not folds:
            print(f"{sym:10s} (insufficient data)")
            continue
        a = np.array([f["alpha"] for f in folds])
        med = float(np.median(a)); pos = int((a > 0).sum())
        reg_a = float(np.mean([f["reg_alpha"] for f in folds]))
        beta = float(np.mean([f["beta"] for f in folds]))
        cells = " ".join(f"{x:+5.2f}" for x in a)
        print(f"{sym:10s} {cells:40s} {med:+6.2f} {pos}/{len(a):>2d} {reg_a:+7.2f} {beta:+6.2f}")
        all_fold_medians.append(med); all_betas.append(beta); all_regalpha.append(reg_a)

    if all_fold_medians:
        med_all = float(np.median(all_fold_medians))
        pos_syms = sum(1 for m in all_fold_medians if m > 0)
        print("-" * 96)
        print(f"VERDICT: median symbol fold-alpha = {med_all:+.2f}  "
              f"({pos_syms}/{len(all_fold_medians)} symbols net-positive)  "
              f"| mean regα={np.mean(all_regalpha):+.2f}  mean beta={np.mean(all_betas):+.2f}")
        print("Promote only if alpha is POSITIVE across MOST symbols/folds AND regα>0 with |beta| "
              "small. High Sharpe with beta≈1 + regα≈0 = market beta, NOT skill — do not ship it.")


if __name__ == "__main__":
    main()
