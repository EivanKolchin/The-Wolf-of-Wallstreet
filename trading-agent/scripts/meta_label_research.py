#!/usr/bin/env python
"""Meta-labeling walk-forward A/B: does sizing the managed-beta book by P(target-before-stop) beat
the unsized binary book?

Pipeline (López de Prado meta-labeling):
  1. PRIMARY signal = managed-beta trend filter (long when close > 200-EMA, else flat) — binary, causal.
  2. LABEL = triple barrier on each long bar (target +k·σ before stop −m·σ within H bars → 1, else 0).
  3. FEATURES = causal regime/vol/trend state at the bar (incl. Garman-Klass vol + panel correlation).
  4. META-MODEL = shallow gradient boosting → p = P(target first). Trained POOLED across assets on a
     trailing 252-day window, predicted on the next 21-day OOS block, rolled forward (purged: labels
     whose horizon would overlap the test block are dropped from training).
  5. SIZE = max(0, 2p−1). Book return = mean over assets of size · signal · next-bar return.

Reports OOS: meta-model AUC, hit-rate (sized-in vs all), and the headline — Sharpe / vol / maxDD of
the UNSIZED vs SIZED book, plus a GROSS-MATCHED sized variant (so we measure allocation skill, not just
the de-leveraging that any p<1 sizing causes). Equities (5-ETF, 2010-) and the 7-asset crypto book.
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from backend.strategies.meta_label import (TripleBarrierParams, triple_barrier_labels,
                                           meta_features, size_from_proba, MetaLabeler)
from backend.signals.macro_regime import avg_pairwise_correlation

EQ = str(ROOT / "training_data" / "equity_daily")
RAW = str(ROOT / "training_data" / "raw")
BPY = 252


# ───────────────────────── data ─────────────────────────
def _load_etf(sym):
    f = sorted(glob.glob(f"{EQ}/{sym}_*_1d.parquet"))[0]
    df = pd.read_parquet(f)[["timestamp", "open", "high", "low", "close"]].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None).dt.normalize()
    return df.drop_duplicates("timestamp").set_index("timestamp")

def _load_crypto(sym):
    f = sorted(glob.glob(f"{RAW}/{sym}_1h_*.parquet"))[0]
    raw = pd.read_parquet(f)[["timestamp", "open", "high", "low", "close"]].copy()
    raw["timestamp"] = pd.to_datetime(raw["timestamp"]).dt.tz_localize(None).dt.normalize()
    g = raw.groupby("timestamp")
    return pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(),
                         "low": g["low"].min(), "close": g["close"].last()})

def build_book(symbols):
    frames = {}
    for s, kind in symbols:
        frames[s] = _load_etf(s) if kind == "etf" else _load_crypto(s)
    idx = None
    for df in frames.values():
        idx = df.index if idx is None else idx.intersection(df.index)
    return {s: df.reindex(idx).dropna() for s, df in frames.items()}, idx


# ───────────────────────── primary signal + per-asset arrays ─────────────────────────
def primary_signal(close, trend_ema=200):
    """Managed-beta: long when close>EMA200, else flat. Signal SHIFTED 1 bar → applied to t→t+1."""
    ema = pd.Series(close).ewm(span=trend_ema, adjust=False).mean().to_numpy()
    raw = (close > ema).astype(float)
    return np.concatenate([[0.0], raw[:-1]])           # causal: decision uses bar t-1's close


def assemble(book, tb: TripleBarrierParams, trend_ema=200):
    """Per-asset: signal, triple-barrier label, meta-features, next-bar return. Shared panel-corr regime."""
    syms = list(book)
    closes = np.column_stack([book[s]["close"].to_numpy(float) for s in syms])
    rets = np.vstack([np.diff(np.log(closes[:, j] + 1e-12), prepend=0.0) for j in range(len(syms))]).T
    panel_corr = avg_pairwise_correlation(rets, window=63)             # cross-asset fragility, causal
    data = {}
    for j, s in enumerate(syms):
        df = book[s]
        c = df["close"].to_numpy(float)
        sig = primary_signal(c, trend_ema)
        y = triple_barrier_labels(df["high"].to_numpy(float), df["low"].to_numpy(float), c, sig, tb)
        X, names = meta_features(df, trend_ema=trend_ema, panel_regime=panel_corr)
        fwd = np.concatenate([c[1:] / c[:-1] - 1.0, [np.nan]])          # t→t+1 simple return
        data[s] = dict(signal=sig, label=y, X=X, fwd=fwd, feat_names=names)
    return data, syms


# ───────────────────────── walk-forward A/B ─────────────────────────
def walk_forward(data, syms, *, lookback=252, step=21, horizon=21):
    """Roll a pooled meta-model. Train on trailing `lookback` bars (purging the last `horizon` bars
    so their forward labels can't leak into the OOS block), predict the next `step` bars per asset."""
    from sklearn.metrics import roc_auc_score
    n = len(data[syms[0]]["signal"])
    unsized, sized, regime_sized = [], [], []   # per-bar book returns, OOS only
    p_all, y_all = [], []
    ptr_all, ytr_all = [], []                   # in-sample (train) preds → train AUC (bug vs no-signal)
    t = lookback
    while t + step <= n:
        tr_lo, tr_hi = t - lookback, t - horizon          # purge horizon bars at the train tail
        Xtr, ytr = [], []
        for s in syms:
            d = data[s]
            sl = slice(tr_lo, tr_hi)
            m = (d["signal"][sl] > 0) & np.isfinite(d["label"][sl])
            if m.any():
                Xtr.append(d["X"][sl][m]); ytr.append(d["label"][sl][m])
        if not Xtr:
            t += step; continue
        Xtr = np.vstack(Xtr); ytr = np.concatenate(ytr)
        model = MetaLabeler().fit(Xtr, ytr)
        if model.model is not None:                       # track in-sample fit (every 5th fold for speed)
            ptr_all.append(model.predict_proba(Xtr)); ytr_all.append(ytr)
        # OOS block — batch ALL (bar, asset) rows through one predict_proba call (fast)
        hi = min(t + step, n)
        rows, meta = [], []                                   # meta = (local_bar_idx, asset, fwd, label)
        for bi, k in enumerate(range(t, hi)):
            for s in syms:
                d = data[s]
                if (d["signal"][k] > 0) and np.isfinite(d["fwd"][k]):
                    rows.append(d["X"][k]); meta.append((bi, s, d["fwd"][k], d["label"][k]))
        if rows:
            p = model.predict_proba(np.vstack(rows))
            size = size_from_proba(p)
            nb = hi - t
            u_acc = np.zeros(nb); s_acc = np.zeros(nb); cnt = np.zeros(nb)
            for (bi, s, fwd, lab), pj, szj in zip(meta, p, size):
                u_acc[bi] += fwd; s_acc[bi] += szj * fwd; cnt[bi] += 1
                if np.isfinite(lab):
                    p_all.append(float(pj)); y_all.append(float(lab))
            live = cnt > 0
            unsized.extend((u_acc[live] / cnt[live]).tolist())
            sized.extend((s_acc[live] / cnt[live]).tolist())
        t += step
    unsized = np.array(unsized); sized = np.array(sized)
    # gross-match: rescale sized so its mean |exposure| equals unsized's (=1) → isolates allocation skill
    scale = (np.abs(unsized).mean() / (np.abs(sized).mean() + 1e-12)) if sized.size else 1.0
    sized_gm = sized * scale
    auc = roc_auc_score(y_all, p_all) if len(set(y_all)) > 1 else float("nan")
    if ptr_all:
        ptr = np.concatenate(ptr_all); ytr = np.concatenate(ytr_all)
        train_auc = roc_auc_score(ytr, ptr) if len(set(ytr.tolist())) > 1 else float("nan")
    else:
        train_auc = float("nan")
    return dict(unsized=unsized, sized=sized, sized_gm=sized_gm, auc=auc, train_auc=train_auc,
                base_rate=float(np.mean(y_all)) if y_all else float("nan"),
                gross_scale=scale, n_pred=len(y_all))


def _stats(r):
    r = r[np.isfinite(r)]; sd = r.std()
    eq = np.cumprod(1 + r); peak = np.maximum.accumulate(eq); dd = ((eq - peak) / peak).min()
    return dict(sharpe=(r.mean()/sd*np.sqrt(BPY)) if sd > 1e-12 else float("nan"),
                ann_vol=sd*np.sqrt(BPY), ann_ret=r.mean()*BPY, maxdd=float(dd))


def report(name, symbols, tb):
    book, idx = build_book(symbols)
    data, syms = assemble(book, tb)
    res = walk_forward(data, syms, horizon=tb.horizon)
    print(f"\n================  META-LABELING A/B — {name}  ================")
    print(f"  {len(idx)} days {idx[0].date()}→{idx[-1].date()} | {len(syms)} assets | "
          f"barrier: +{tb.target_sigma}σ/−{tb.stop_sigma}σ/{tb.horizon}d")
    print(f"  meta-model AUC: OOS={res['auc']:.3f}  train(in-sample)={res['train_auc']:.3f}   "
          f"(base rate P(target)={res['base_rate']:.3f}, n_pred={res['n_pred']})   "
          f"gross-match scale={res['gross_scale']:.2f}")
    print(f"     [train>>0.5 but OOS≈0.5 ⇒ overfit / no generalizable signal; train≈0.5 ⇒ features can't "
          f"separate the label even in-sample]")
    print(f"\n  {'book':16s}  {'Sharpe':>8s}  {'ann vol':>8s}  {'ann ret':>8s}  {'maxDD':>8s}")
    for label, r in [("unsized (binary)", res["unsized"]), ("meta-sized", res["sized"]),
                     ("meta-sized (gross-m)", res["sized_gm"])]:
        s = _stats(r)
        print(f"  {label:16s}  {s['sharpe']:>+8.3f}  {s['ann_vol']:>8.1%}  {s['ann_ret']:>+8.1%}  {s['maxdd']:>8.1%}")
    su, ss = _stats(res["unsized"])["sharpe"], _stats(res["sized_gm"])["sharpe"]
    verdict = "LIFTS" if ss > su + 0.05 else "HURTS" if ss < su - 0.05 else "NEUTRAL"
    print(f"  => gross-matched meta-sizing {verdict} Sharpe vs unsized ({ss:+.3f} vs {su:+.3f}). "
          f"AUC>0.52 + LIFTS = real edge.")
    return res


if __name__ == "__main__":
    etf5 = [("SPY", "etf"), ("QQQ", "etf"), ("TQQQ", "etf"), ("TLT", "etf"), ("GLD", "etf")]
    seven = etf5 + [("BTCUSDT", "crypto"), ("ETHUSDT", "crypto")]
    # Symmetric barrier (1σ/1σ) → base rate ≈ 0.5 so the 2p−1 sizing map isn't degenerate; this is
    # the fair test of whether sizing helps. Asymmetric (1.5σ/1σ) also shown for completeness.
    print("\n##############  SYMMETRIC barrier (1σ/1σ, base rate ≈ 0.5 — fair sizing test)  ##############")
    tb_sym = TripleBarrierParams(horizon=21, target_sigma=1.0, stop_sigma=1.0, vol_window=20)
    report("5-ETF managed book", etf5, tb_sym)
    report("7-asset managed book", seven, tb_sym)
    print("\n##############  ASYMMETRIC barrier (1.5σ/1σ)  ##############")
    tb_asym = TripleBarrierParams(horizon=21, target_sigma=1.5, stop_sigma=1.0, vol_window=20)
    report("5-ETF managed book", etf5, tb_asym)
    report("7-asset managed book", seven, tb_asym)
