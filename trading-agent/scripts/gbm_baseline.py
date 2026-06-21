#!/usr/bin/env python
"""Final comprehensive model probe — pick the best model class per asset, per horizon, per conviction.

The turnover sweep proved the crypto edge is real but lives in the high-conviction / low-turnover
tail. This runs the WHOLE comparison in one shot so we can start building:

  * MODELS:   gbm (shallow boosting) · mlp (small regularised NN) · logistic (linear)
  * ASSETS:   crypto and stocks reported SEPARATELY (the best model may differ by asset class)
  * HORIZONS: 48 (4h) and 288 (~1d)  — incl. the daily-semis lead the audit hinted at for stocks
  * CONVICTION: swept all the way to 0.999 (each model trained ONCE; thresholds are then free)

For every (horizon × model × asset-class) it prints the conviction sweep: median/mean ALPHA
(excess Sharpe vs buy-and-hold, after costs) and avg trades. Read it as: where does alpha go
clearly positive at a non-trivial trade count? That (asset, model, horizon, conf) is the build.

    python scripts/gbm_baseline.py --skip-download
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

_spec = importlib.util.spec_from_file_location("pretrain_gbm_mod", str(ROOT / "scripts" / "pretrain.py"))
pre = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pre)

from backend.backtest.engine import run_backtest  # noqa: E402

CONFS = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99, 0.999]
CRYPTO = "BTCUSDT ETHUSDT SOLUSDT AAVEUSDT XLMUSDT XRPUSDT ADAUSDT DOGEUSDT RENDERUSDT NEARUSDT"
STOCKS = "SNDK AMD MU AXTI BE NVDA TSM SMCI"


def _forward_returns(close, h):
    out = np.full(close.shape[0], np.nan)
    if close.shape[0] > h:
        out[:-h] = close[h:] / close[:-h] - 1.0
    return out


def _purged_split(n, test_frac, embargo):
    cut = max(1, min(n - 1, int(n * (1.0 - test_frac))))
    return np.arange(0, max(1, cut - embargo)), np.arange(cut, n)


def _make_model(kind):
    if kind == "mlp":
        from sklearn.neural_network import MLPClassifier
        return MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-2, max_iter=60,
                             early_stopping=True, n_iter_no_change=6, random_state=0)
    if kind == "logistic":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(C=0.1, max_iter=300)
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_depth=3, max_iter=200, learning_rate=0.05,
                                          l2_regularization=1.0, random_state=0)


def load_symbol(sym, start_year, start_month, skip_download, horizon, test_frac=0.2):
    combined, _, _, _ = pre.assemble_with_cache(sym, start_year, start_month, skip_download)
    dfs = pre.load_full_history(sym, start_year, start_month, skip_download)
    close = dfs["5m"]["close"].to_numpy(np.float64)
    n = min(len(combined), len(close))
    X, close = combined[:n].astype(np.float64), close[:n]
    fwd = _forward_returns(close, horizon)
    tr, te = _purged_split(n, test_frac, horizon)
    return X, close, fwd, tr, te


def fit_predict(X, fwd, tr, te, kind, max_train=80000):
    y = (fwd > 0).astype(int)
    m = np.isfinite(fwd[tr])
    Xtr, ytr = X[tr][m], y[tr][m]
    if len(Xtr) < 500 or len(np.unique(ytr)) < 2:
        raise RuntimeError("insufficient")
    if len(Xtr) > max_train:
        s = len(Xtr) // max_train + 1
        Xtr, ytr = Xtr[::s], ytr[::s]
    Xte = X[te]
    if kind in ("mlp", "logistic"):
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler().fit(Xtr)
        Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    clf = _make_model(kind)
    clf.fit(Xtr, ytr)
    return clf.predict_proba(Xte)[:, 1]


def _alpha_at(close_te, p_up, conf):
    sig = np.where(p_up > conf, 1.0, np.where(p_up < 1.0 - conf, -1.0, 0.0))
    if not np.any(sig != 0.0):
        return np.nan, 0                                  # fully flat = no strategy
    m = run_backtest(close_te, sig, fee_bps=10, slippage_bps=5).metrics
    return m.get("excess_sharpe", 0.0), int(m["num_trades"])


def main():
    ap = argparse.ArgumentParser(description="Comprehensive model probe (model x asset x horizon x conf)")
    ap.add_argument("--symbols", default=f"{CRYPTO} {STOCKS}")
    ap.add_argument("--models", default="gbm mlp logistic")
    ap.add_argument("--horizons", default="48 288")
    ap.add_argument("--start-year", type=int, default=2022)
    ap.add_argument("--start-month", type=int, default=1)
    ap.add_argument("--skip-download", action="store_true")
    args = ap.parse_args()

    symbols = args.symbols.split()
    models = args.models.split()
    horizons = [int(h) for h in args.horizons.split()]
    print(f"Probe: models={models}  horizons={horizons}  conf→0.999  (alpha = excess Sharpe vs B&H)")

    for horizon in horizons:
        data = {}
        for sym in symbols:
            try:
                data[sym] = load_symbol(sym, args.start_year, args.start_month, args.skip_download, horizon)
            except Exception as e:
                print(f"  load skip {sym} @H+{horizon}: {str(e)[:40]}")
        for kind in models:
            preds = {}
            for sym, (X, close, fwd, tr, te) in data.items():
                try:
                    preds[sym] = (close[te], fit_predict(X, fwd, tr, te, kind))
                except Exception:
                    pass
            crypto = [s for s in preds if s.endswith("USDT")]
            stocks = [s for s in preds if not s.endswith("USDT")]
            for gname, gsyms in (("CRYPTO", crypto), ("STOCKS", stocks)):
                if not gsyms:
                    continue
                print(f"\n=== H+{horizon} | {kind.upper():8s} | {gname} ({len(gsyms)} syms) "
                      f"{'='*max(0, 24-len(gname))}")
                print(f"{'conf':>6s} {'medAlpha':>9s} {'meanAlpha':>10s} {'avgTrades':>10s}")
                for conf in CONFS:
                    rows = [_alpha_at(preds[s][0], preds[s][1], conf) for s in gsyms]
                    a = np.array([r[0] for r in rows], float)
                    t = np.array([r[1] for r in rows], float)
                    if np.all(np.isnan(a)):
                        print(f"{conf:>6.3f} {'flat':>9s} {'flat':>10s} {0:>10.0f}")
                    else:
                        print(f"{conf:>6.3f} {np.nanmedian(a):>9.2f} {np.nanmean(a):>10.2f} {np.mean(t):>10.0f}")

    print("\nBUILD = the (asset, model, horizon, conf) where alpha is clearly >0 at a NON-trivial trade")
    print("count. Crypto's best model may differ from stocks'. Then I build that as the real model.")


if __name__ == "__main__":
    main()
