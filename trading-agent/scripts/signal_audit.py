#!/usr/bin/env python
"""Signal audit — the FAST feedback loop (minutes, not a 40-min train).

Three architectures tied at zero edge, so before we add features / commit to retrains we
MEASURE whether the inputs carry any predictive signal at each horizon. For a few symbols
(from the SAME cached parquet + the SAME assemble_feature_matrix as training), with a purged
train/test split, this reports:

  • per-feature rank-IC  (Spearman corr of each feature vs forward return)  → which features
    carry signal and which are dead/constant (exposes the ~37 stubbed-for-crypto columns);
  • a cheap gradient-boosting baseline's TEST AUC per horizon, for BOTH label definitions
    (sign-of-forward-return and the vol-neutral triple-barrier long-vs-short) → the honest
    "is there ANY learnable edge" number. AUC≈0.50 = none; >0.55 after a purge = real signal.

This is measurement only — it trains no NN and writes no checkpoint. Run it as a Colab cell
BEFORE the experiments to decide which Phase-B features/labels are worth a retrain.

    python scripts/signal_audit.py --start-year 2022 --skip-download \
        --symbols "BTCUSDT ETHUSDT NVDA" --horizons "12 48 288 864"
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

_spec = importlib.util.spec_from_file_location("pretrain_audit_mod", str(ROOT / "scripts" / "pretrain.py"))
pre = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pre)

from signals import feature_spec as fs  # readable feature names for the IC report


# ───────────────────────────── pure helpers (unit-testable) ──────────────────────────────
def forward_returns(close: np.ndarray, h: int) -> np.ndarray:
    """close[i+h]/close[i]-1, with the last h bars set to NaN (no future bar)."""
    close = np.asarray(close, dtype=np.float64)
    out = np.full(close.shape[0], np.nan)
    if close.shape[0] > h:
        out[:-h] = close[h:] / close[:-h] - 1.0
    return out


def _rankdata(x: np.ndarray) -> np.ndarray:
    """Average ranks (ties → mean rank), no scipy dependency."""
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(x.shape[0], dtype=np.float64)
    ranks[order] = np.arange(1, x.shape[0] + 1)
    # average ties
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    sums = np.zeros(counts.shape[0]); np.add.at(sums, inv, ranks)
    return (sums / counts)[inv]


def rank_ic(feature: np.ndarray, fwd: np.ndarray) -> float:
    """Spearman rank correlation between a feature column and forward return (valid rows only)."""
    f = np.asarray(feature, dtype=np.float64)
    y = np.asarray(fwd, dtype=np.float64)
    m = np.isfinite(f) & np.isfinite(y)
    if m.sum() < 50 or np.nanstd(f[m]) < 1e-12:
        return 0.0
    rf, ry = _rankdata(f[m]), _rankdata(y[m])
    rf -= rf.mean(); ry -= ry.mean()
    denom = np.sqrt((rf * rf).sum() * (ry * ry).sum())
    return float((rf * ry).sum() / denom) if denom > 0 else 0.0


def candidate_features(high, low, close, volume) -> np.ndarray:
    """Phase-B PROTOTYPE features — measured in-audit BEFORE any spec bump or retrain. The audit
    showed the surviving edge is trend/momentum at 1h-4h, so these strengthen that: multi-scale
    momentum, trend vs moving-average, position-in-range (swing levels), and a vol regime. All
    causal (backward rolling / pct_change). If they lift the baseline AUC, they're worth building
    for real in feature_spec v2.4; if not, we learned it for the price of a 10-min audit."""
    high = np.asarray(high, float); low = np.asarray(low, float)
    close = np.asarray(close, float); volume = np.asarray(volume, float)
    c = pd.Series(close)
    cols = []
    # multi-scale momentum (log return over w bars): 4h, 1d, 3d, 1w, ~1mo (the user's multi-scale)
    for w in (48, 288, 864, 2016, 8640):
        cols.append(np.nan_to_num(np.log(close / np.roll(close, w)), posinf=0.0, neginf=0.0) * (np.arange(len(close)) >= w))
    # trend: (close - SMA_w) / close  at 1d, 5d
    for w in (288, 1440):
        sma = c.rolling(w, min_periods=w // 2).mean().to_numpy()
        cols.append(np.nan_to_num((close - sma) / (close + 1e-9)))
    # position-in-range vs rolling swing high/low (0=low, 1=high) at 1d, 5d
    for w in (288, 1440):
        hi = pd.Series(high).rolling(w, min_periods=w // 2).max().to_numpy()
        lo = pd.Series(low).rolling(w, min_periods=w // 2).min().to_numpy()
        cols.append(np.nan_to_num((close - lo) / (hi - lo + 1e-9)))
    # vol regime: z-score of short ATR vs its own long window (>0 = unusually volatile now)
    tr = np.abs(np.diff(close, prepend=close[:1])) / (close + 1e-9)
    atr = pd.Series(tr).rolling(96, min_periods=20).mean()
    z = (atr - atr.rolling(1000, min_periods=100).mean()) / (atr.rolling(1000, min_periods=100).std() + 1e-9)
    cols.append(np.nan_to_num(z.to_numpy()))
    return np.column_stack(cols).astype(np.float32)


_BTC_CTX_CACHE: dict = {}


def btc_context(start_year, start_month, skip_download) -> pd.DataFrame:
    """BTC cross-asset context (the one genuinely-NEW information axis vs an alt's own technicals):
    BTC's recent multi-scale return + trend, to test whether BTC leads the alts at 1h-4h. Cached
    once per run. Causal (backward returns / rolling)."""
    key = (start_year, start_month)
    if key in _BTC_CTX_CACHE:
        return _BTC_CTX_CACHE[key]
    dfs = pre.load_full_history("BTCUSDT", start_year, start_month, skip_download)
    df = dfs["5m"]
    c = df["close"].to_numpy(np.float64)
    ctx = pd.DataFrame({"timestamp": df["timestamp"].to_numpy()})
    for w in (12, 48, 288):
        ctx[f"btc_r{w}"] = np.nan_to_num(np.log(c / np.roll(c, w)), posinf=0.0, neginf=0.0) * (np.arange(len(c)) >= w)
    sma = pd.Series(c).rolling(288, min_periods=144).mean().to_numpy()
    ctx["btc_trend"] = np.nan_to_num((c - sma) / (c + 1e-9))
    _BTC_CTX_CACHE[key] = ctx
    return ctx


def purged_split(n: int, test_frac: float, embargo: int):
    """Chronological train/test indices with an embargo gap so the train labels (horizon h)
    can't overlap the test window."""
    cut = max(1, min(n - 1, int(n * (1.0 - test_frac))))
    tr_hi = max(1, cut - embargo)
    return np.arange(0, tr_hi), np.arange(cut, n)


# ─────────────────────── Phase 1: perp funding candidate (gated, network) ────────────────
def _funding_candidate(sym: str, df5: pd.DataFrame, n: int):
    """(n, 4) causal perp-funding features aligned to df5's 5m bars, or None.

    Gated behind AUDIT_FUNDING=1 (it hits the Binance futures API). Funding is the one
    genuinely-NEW information axis vs an alt's own technicals — this lets the audit
    MEASURE its incremental AUC/IC before any feature_spec bump or retrain."""
    from backend.data.derivatives_feed import fetch_funding_history, funding_features
    ts = df5["timestamp"].values.astype("datetime64[ms]").astype("int64")[:n]
    fh = fetch_funding_history(sym, start_ms=int(ts[0]), end_ms=int(ts[-1]))
    if fh.empty:
        return None
    return funding_features(fh["timestamp"].to_numpy(), fh["funding_rate"].to_numpy(), ts)


# ───────────────────────────── audit per symbol/horizon ─────────────────────────────────
def _baseline_auc(X, y, tr, te, max_train=40000):
    """HistGradientBoosting test AUC on a binary label; subsamples train for speed. Returns
    None if sklearn is absent or a class is empty."""
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.metrics import roc_auc_score
    except Exception:
        return None
    mtr = np.isfinite(y[tr]); mte = np.isfinite(y[te])
    Xtr, ytr = X[tr][mtr], y[tr][mtr]; Xte, yte = X[te][mte], y[te][mte]
    if len(Xtr) > max_train:                          # stride-subsample (decorrelate + speed)
        s = len(Xtr) // max_train + 1
        Xtr, ytr = Xtr[::s], ytr[::s]
    if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
        return None
    clf = HistGradientBoostingClassifier(max_depth=3, max_iter=120, learning_rate=0.05,
                                         l2_regularization=1.0, random_state=0)
    clf.fit(Xtr, ytr)
    p = clf.predict_proba(Xte)[:, 1]
    return float(roc_auc_score(yte, p))


def audit_symbol(sym, start_year, start_month, skip_download, horizons, test_frac=0.2,
                 with_candidates=True):
    combined, _, _, _ = pre.assemble_with_cache(sym, start_year, start_month, skip_download)
    # Raw OHLCV for forward returns + the candidate features (cheap parquet read).
    dfs = pre.load_full_history(sym, start_year, start_month, skip_download)
    df5 = dfs["5m"]
    high = df5["high"].to_numpy(np.float64); low = df5["low"].to_numpy(np.float64)
    close = df5["close"].to_numpy(np.float64); volume = df5["volume"].to_numpy(np.float64)
    n = min(len(combined), len(close))
    combined, high, low, close, volume = combined[:n], high[:n], low[:n], close[:n], volume[:n]
    vol = pre._rolling_vol(close)                      # per-bar vol for the neutral band

    # Phase-B candidates appended to the base matrix → does the AUC rise? (measure before build)
    augmented = combined
    cand_labels: list = []
    if with_candidates:
        cand = candidate_features(high, low, close, volume)[:n]
        cand_labels = ["mom_48", "mom_288", "mom_864", "mom_2016", "mom_8640",
                       "trend_288", "trend_1440", "range_288", "range_1440", "vol_regime_z"]
        # Cross-asset: for crypto ALTS, append BTC's recent context (new info, not a momentum
        # restatement). BTC itself / stocks get no BTC context (it'd be self/irrelevant).
        if sym.endswith("USDT") and sym != "BTCUSDT":
            try:
                bctx = btc_context(start_year, start_month, skip_download)
                merged = pd.merge_asof(
                    pd.DataFrame({"timestamp": df5["timestamp"].to_numpy()[:n]}),
                    bctx.sort_values("timestamp"), on="timestamp", direction="backward")
                bcols = np.nan_to_num(merged[["btc_r12", "btc_r48", "btc_r288", "btc_trend"]]
                                      .to_numpy(np.float32))
                cand = np.column_stack([cand, bcols])
                cand_labels += ["btc_r12", "btc_r48", "btc_r288", "btc_trend"]
            except Exception as e:
                print(f"  (btc-context skipped for {sym}: {str(e)[:60]})")
        # Phase 1: perp funding (the genuinely-new positioning axis). Gated — network.
        if os.environ.get("AUDIT_FUNDING", "0") in ("1", "true", "True") and sym.endswith("USDT"):
            try:
                fc = _funding_candidate(sym, df5, n)
                if fc is not None:
                    cand = np.column_stack([cand, fc])
                    cand_labels += ["fund_level", "fund_change", "fund_zscore", "fund_carry"]
                    print(f"  (+funding features for {sym})")
            except Exception as e:
                print(f"  (funding skipped for {sym}: {str(e)[:60]})")
        augmented = np.column_stack([combined, cand])

    base_F = combined.shape[1]
    res = {"symbol": sym, "horizons": {}, "ic": {}, "cand_ic": {}, "cand_labels": cand_labels}
    for h in horizons:
        fwd = forward_returns(close, h)
        band = 0.5 * vol * np.sqrt(h)                  # vol-scaled neutral band
        tr, te = purged_split(n, test_frac, embargo=h)
        # label 1: sign of forward return (all rows)
        y_sign = np.where(np.isfinite(fwd), (fwd > 0).astype(float), np.nan)
        auc_sign = _baseline_auc(combined, y_sign, tr, te)
        auc_aug = _baseline_auc(augmented, y_sign, tr, te) if with_candidates else None
        # label 2: vol-neutral barrier (long vs short; neutrals dropped)
        y_bar = np.full(n, np.nan)
        y_bar[fwd > band] = 1.0; y_bar[fwd < -band] = 0.0
        auc_bar = _baseline_auc(combined, y_bar, tr, te)
        res["horizons"][h] = {"auc_sign": auc_sign, "auc_barrier": auc_bar, "auc_aug": auc_aug,
                              "traded_frac": float(np.isfinite(y_bar).mean())}
        # per-feature IC at this horizon (test region only → honest)
        res["ic"][h] = np.array([rank_ic(combined[te, j], fwd[te]) for j in range(base_F)])
        # standalone IC of each appended candidate (so funding's own signal is explicit,
        # not bundled into the +cand AUC)
        if cand_labels:
            res["cand_ic"][h] = np.array([rank_ic(augmented[te, base_F + k], fwd[te])
                                          for k in range(len(cand_labels))])
    return res


def main():
    ap = argparse.ArgumentParser(description="Measure feature/label signal before retraining")
    ap.add_argument("--symbols", default="BTCUSDT ETHUSDT NVDA")
    ap.add_argument("--start-year", type=int, default=2022)
    ap.add_argument("--start-month", type=int, default=1)
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--horizons", default="12 48 288 864")
    args = ap.parse_args()

    symbols = args.symbols.split()
    horizons = [int(x) for x in args.horizons.split()]
    results = []
    for sym in symbols:
        try:
            results.append(audit_symbol(sym, args.start_year, args.start_month,
                                        args.skip_download, horizons))
            print(f"audited {sym}")
        except Exception as e:
            print(f"skip {sym}: {str(e)[:80]}")

    if not results:
        print("no symbols audited"); return

    print("\n================  BASELINE TEST AUC — base vs +Phase-B-candidates  ================")
    print(f"{'symbol':10s} " + "  ".join(f"H+{h:<5d}(base/+cand)" for h in horizons))
    for r in results:
        cells = []
        for h in horizons:
            d = r["horizons"][h]
            s = "n/a" if d["auc_sign"] is None else f"{d['auc_sign']:.3f}"
            a = "n/a" if d.get("auc_aug") is None else f"{d['auc_aug']:.3f}"
            cells.append(f"{s}/{a}")
        print(f"{r['symbol']:10s} " + "  ".join(f"{c:>16s}" for c in cells))
    print("base = current 90 features; +cand = momentum/level/vol candidates (+ BTC cross-asset context for ALTS).")
    print("momentum/levels already shown neutral, so any ALT lift here = the BTC context. AUC>0.55 = real edge.")

    # aggregate |IC| across symbols+horizons → rank features, flag dead ones
    F = results[0]["ic"][horizons[0]].shape[0]
    absic = np.zeros(F)
    for r in results:
        for h in horizons:
            absic += np.abs(r["ic"][h])
    absic /= (len(results) * len(horizons))
    order = np.argsort(absic)[::-1]
    print("\n================  TOP FEATURES by mean |rank-IC|  ================")
    for j in order[:20]:
        print(f"  {j:3d} {fs.feature_name(j):22s} |IC|={absic[j]:.4f}")
    dead_idx = [int(j) for j in range(F) if absic[j] < 0.003]
    print(f"\n================  DEAD features (|IC|<0.003): {len(dead_idx)}/{F}  ================")
    print("  " + ", ".join(f"{j}:{fs.feature_name(j)}" for j in dead_idx))
    keep_idx = [int(j) for j in range(F) if absic[j] >= 0.003]
    print(f"\nKEEP ({len(keep_idx)}): " + ", ".join(fs.feature_name(j) for j in keep_idx))

    # Standalone candidate / funding IC (aggregated by label across symbols+horizons), so
    # funding's own signal is explicit rather than buried in the bundled +cand AUC.
    from collections import defaultdict
    cand_by_label: dict = defaultdict(list)
    for r in results:
        labels = r.get("cand_labels") or []
        for h in horizons:
            arr = r.get("cand_ic", {}).get(h)
            if arr is None:
                continue
            for k, lab in enumerate(labels):
                cand_by_label[lab].append(abs(float(arr[k])))
    if cand_by_label:
        print("\n================  CANDIDATE / FUNDING mean |rank-IC|  ================")
        for lab, vals in sorted(cand_by_label.items(), key=lambda kv: -np.mean(kv[1])):
            print(f"  {lab:16s} |IC|={np.mean(vals):.4f}  (n={len(vals)})")
        print("A candidate (incl. funding) earns a feature_spec slot only if its |IC| rivals the "
              "live top features (≈0.02+). Below the top base features ⇒ not worth a retrain.")


if __name__ == "__main__":
    main()
