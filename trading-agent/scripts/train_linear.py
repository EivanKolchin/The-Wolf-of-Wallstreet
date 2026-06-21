#!/usr/bin/env python
"""Crypto linear model — the build the evidence supports.

The comprehensive probe showed the crypto edge is LINEAR and thin: a logistic model beat the GBM,
MLP and LSTM (more capacity = more overfit), and it's tradeable only at HIGH conviction (low
turnover). Stocks had no technical edge at any model/horizon → crypto-only here.

This trains ONE pooled logistic model over all crypto symbols (a universal crypto-momentum model)
and validates it WALK-FORWARD across several expanding time folds — the test the probe skipped. A
single split's +0.5 alpha could be one-regime luck; if the edge holds across folds it's real.

  * pooled logistic on the sign of the H-bar forward return (the robust winner)
  * high-conviction signals only (|p-0.5| past `--conf`) → low turnover, costs don't dominate
  * expanding walk-forward: train on all history before each test block (purged), test on it
  * reports per-fold + aggregate ALPHA (excess Sharpe vs buy-and-hold, after costs)
  * saves the final model (coef/intercept/scaler) to models/crypto_linear.npz for deployment

    python scripts/train_linear.py --skip-download --horizon 12 --conf 0.65 --folds 4
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

_spec = importlib.util.spec_from_file_location("pretrain_lin_mod", str(ROOT / "scripts" / "pretrain.py"))
pre = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pre)

from backend.backtest.engine import run_backtest  # noqa: E402

CRYPTO = "BTCUSDT ETHUSDT SOLUSDT AAVEUSDT XLMUSDT XRPUSDT ADAUSDT DOGEUSDT RENDERUSDT NEARUSDT"


def _forward_returns(close, h):
    out = np.full(close.shape[0], np.nan)
    if close.shape[0] > h:
        out[:-h] = close[h:] / close[:-h] - 1.0
    return out


def load_symbol(sym, start_year, start_month, skip_download, horizon):
    combined, _, _, _ = pre.assemble_with_cache(sym, start_year, start_month, skip_download)
    dfs = pre.load_full_history(sym, start_year, start_month, skip_download)
    close = dfs["5m"]["close"].to_numpy(np.float64)
    n = min(len(combined), len(close))
    return combined[:n].astype(np.float64), close[:n], _forward_returns(close[:n], horizon)


def _fit(Xtr, ytr, max_train=150000):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    if len(Xtr) > max_train:
        s = len(Xtr) // max_train + 1
        Xtr, ytr = Xtr[::s], ytr[::s]
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=0.1, max_iter=400).fit(sc.transform(Xtr), ytr)
    return sc, clf


def _signal_alpha(close_te, p_up, conf):
    sig = np.where(p_up > conf, 1.0, np.where(p_up < 1.0 - conf, -1.0, 0.0))
    if not np.any(sig != 0.0):
        return np.nan
    return run_backtest(close_te, sig, fee_bps=10, slippage_bps=5).metrics.get("excess_sharpe", 0.0)


def main():
    ap = argparse.ArgumentParser(description="Walk-forward linear crypto model")
    ap.add_argument("--symbols", default=CRYPTO)
    ap.add_argument("--start-year", type=int, default=2022)
    ap.add_argument("--start-month", type=int, default=1)
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--horizon", type=int, default=12, help="label horizon in 5m bars (12 = 1h)")
    ap.add_argument("--conf", type=float, default=0.65, help="conviction threshold to take a trade")
    ap.add_argument("--folds", type=int, default=4, help="walk-forward test folds over the last 40%")
    args = ap.parse_args()

    syms = args.symbols.split()
    print(f"Linear crypto model | H+{args.horizon} | conf={args.conf} | {args.folds}-fold walk-forward")
    data = {}
    for sym in syms:
        try:
            data[sym] = load_symbol(sym, args.start_year, args.start_month, args.skip_download, args.horizon)
            print(f"  loaded {sym}")
        except Exception as e:
            print(f"  skip {sym}: {str(e)[:50]}")
    if not data:
        print("no data"); return

    # Expanding walk-forward: test windows tile the last 40% of each symbol's timeline; each fold
    # trains on everything before its test window (minus an embargo to purge label overlap).
    edges = np.linspace(0.60, 1.0, args.folds + 1)
    print(f"\n{'fold':>4s} {'test window':>14s} {'medAlpha':>9s} {'meanAlpha':>10s} {'+syms':>6s}")
    fold_meds = []
    for k in range(args.folds):
        a, b = edges[k], edges[k + 1]
        Xtr_parts, ytr_parts = [], []
        for X, close, fwd in data.values():
            n = len(X); te0 = int(a * n)
            tr_hi = max(1, te0 - args.horizon)
            m = np.isfinite(fwd[:tr_hi])
            Xtr_parts.append(X[:tr_hi][m]); ytr_parts.append((fwd[:tr_hi][m] > 0).astype(int))
        Xtr = np.vstack(Xtr_parts); ytr = np.concatenate(ytr_parts)
        if len(np.unique(ytr)) < 2:
            print(f"{k:>4d}  (degenerate train)"); continue
        sc, clf = _fit(Xtr, ytr)
        alphas = []
        for X, close, fwd in data.values():
            n = len(X); te0, te1 = int(a * n), int(b * n)
            p = clf.predict_proba(sc.transform(X[te0:te1]))[:, 1]
            al = _signal_alpha(close[te0:te1], p, args.conf)
            if np.isfinite(al):
                alphas.append(al)
        med = float(np.median(alphas)) if alphas else np.nan
        fold_meds.append(med)
        mean = float(np.mean(alphas)) if alphas else np.nan
        print(f"{k:>4d} {f'{a:.2f}-{b:.2f}':>14s} {med:>9.2f} {mean:>10.2f} {len(alphas):>6d}")

    finite = [m for m in fold_meds if np.isfinite(m)]
    verdict = float(np.median(finite)) if finite else float("nan")
    pos = sum(1 for m in finite if m > 0)
    print(f"\nWALK-FORWARD VERDICT: median fold alpha = {verdict:+.2f}   ({pos}/{len(finite)} folds positive)")
    print("Positive in MOST folds → robust, deployable edge. Only the last fold → regime luck.")

    # Train the FINAL deployable model on ALL data and save (coef/intercept/scaler).
    Xall = np.vstack([X[np.isfinite(fwd)] for X, _, fwd in data.values()])
    yall = np.concatenate([(fwd[np.isfinite(fwd)] > 0).astype(int) for _, _, fwd in data.values()])
    sc, clf = _fit(Xall, yall)
    out = ROOT / "models" / "crypto_linear.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, coef=clf.coef_.astype(np.float32), intercept=clf.intercept_.astype(np.float32),
             mean=sc.mean_.astype(np.float32), scale=sc.scale_.astype(np.float32),
             horizon=np.int64(args.horizon), conf=np.float64(args.conf),
             feature_version=str(pre.FEATURE_VERSION))
    print(f"\nSaved final model → {out}  (p_up = sigmoid(((x-mean)/scale) @ coef + intercept))")


if __name__ == "__main__":
    main()
