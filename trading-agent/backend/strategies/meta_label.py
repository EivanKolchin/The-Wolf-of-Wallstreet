"""Meta-labeling (López de Prado, AFML Ch.3) — the ML layer that actually fits our problem.

The whole alpha hunt proved the primary engine can't predict DIRECTION (~0.53 AUC). Meta-labeling
sidesteps that: keep the primary signal binary (long / flat), then train a SECONDARY model to
predict P(this trade hits its profit target before its stop) and use that probability to SIZE the
trade. ML's job becomes confidence/sizing — a far better-posed problem than direction — and the
mechanism is sound even with zero directional alpha, because volatility and regime ARE persistent:
the triple-barrier outcome depends on the vol/trend state, which the meta-labeler can learn.

This module is pure + testable: the triple-barrier labeler, the causal meta-feature builder, the
sizing map, and a thin classifier wrapper. The walk-forward A/B lives in scripts/meta_label_research.py.

EMPIRICAL VERDICT (2026-06-29, on the managed-beta DAILY trend primary): meta-labeling did NOT lift
the book. Train(in-sample) AUC ≈ 0.90 but OOS AUC ≈ 0.50 across symmetric/asymmetric barriers and
both universes — classic overfit / no generalizable signal; sizing on that noise HURT Sharpe
(0.93→0.69 on the 5-ETF book). The model CAN fit in-sample (so this is not a bug) — the triple-barrier
outcome of a trend-following trade is simply not OOS-predictable from vol/regime/momentum here. This
kept as validated scaffolding for A/B on a NOISIER primary (e.g. the NN 5m signals, meta-labeling's
canonical use case), which would need NN inference history to test. Don't wire sizing live off this.

Causality contract:
  * FEATURES at bar t use only information through t (trailing windows, current trend/vol state).
  * LABELS are forward-looking by construction (that is the thing we learn to predict); the
    walk-forward harness purges/embargoes so a training label can't peek into the test window.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class TripleBarrierParams:
    horizon: int = 21          # vertical (time) barrier, in bars
    target_sigma: float = 1.5  # upper barrier = +target_sigma * entry-vol (in return space)
    stop_sigma: float = 1.0    # lower barrier = -stop_sigma * entry-vol
    vol_window: int = 20       # trailing window for the entry-vol that scales the barriers


# ───────────────────────────── volatility estimators (causal) ─────────────────────────────
def realized_vol(close: np.ndarray, window: int = 20) -> np.ndarray:
    """Trailing per-bar return volatility (std of log returns over `window`). Known at bar t."""
    c = np.asarray(close, dtype=np.float64)
    r = np.diff(np.log(c + 1e-12), prepend=np.log(c[:1] + 1e-12))
    return pd.Series(r).rolling(window, min_periods=max(5, window // 2)).std().to_numpy()


def garman_klass_vol(o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray,
                     window: int = 20) -> np.ndarray:
    """Range-based (Garman-Klass) per-bar vol — more efficient than close-to-close per bar. Causal."""
    o = np.asarray(o, float); h = np.asarray(h, float); l = np.asarray(l, float); c = np.asarray(c, float)
    hl = np.log(np.maximum(h, 1e-12) / np.maximum(l, 1e-12))
    co = np.log(np.maximum(c, 1e-12) / np.maximum(o, 1e-12))
    gkv = 0.5 * hl * hl - (2.0 * np.log(2.0) - 1.0) * co * co
    return np.sqrt(pd.Series(gkv).rolling(window, min_periods=max(5, window // 2)).mean()
                   .clip(lower=0).to_numpy())


# ───────────────────────────── the triple-barrier label ─────────────────────────────
def triple_barrier_labels(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                          signal: np.ndarray, params: Optional[TripleBarrierParams] = None,
                          ) -> np.ndarray:
    """For every bar t where `signal`>0 (primary says LONG), look forward up to `horizon` bars:
       label 1 if the upper barrier (+target_sigma·σ_t) is touched BEFORE the lower (-stop_sigma·σ_t),
       else 0 (lower-first, or neither touched by the time barrier). NaN where there is no signal or
       not enough future bars. Touches use forward HIGH/LOW (the realistic intrabar reach).

    σ_t is the trailing entry-vol at t (known at t) — barriers are vol-scaled so the label is
    comparable across regimes. The look-forward is intentional: this is the prediction TARGET."""
    p = params or TripleBarrierParams()
    high = np.asarray(high, float); low = np.asarray(low, float); close = np.asarray(close, float)
    signal = np.asarray(signal, float)
    n = close.shape[0]
    sigma = realized_vol(close, p.vol_window)
    out = np.full(n, np.nan)
    for t in range(n):
        if not (signal[t] > 0) or not np.isfinite(sigma[t]) or sigma[t] <= 0:
            continue
        if t + 1 >= n:
            continue
        entry = close[t]
        up = entry * (1.0 + p.target_sigma * sigma[t])
        dn = entry * (1.0 - p.stop_sigma * sigma[t])
        end = min(t + p.horizon, n - 1)
        label = 0.0                                  # time-barrier / stop-first default
        for k in range(t + 1, end + 1):
            hit_up = high[k] >= up
            hit_dn = low[k] <= dn
            if hit_up and hit_dn:                    # both in one bar → conservative: assume stop first
                label = 0.0; break
            if hit_up:
                label = 1.0; break
            if hit_dn:
                label = 0.0; break
        out[t] = label
    return out


# ───────────────────────────── causal meta-features ─────────────────────────────
def meta_features(df: pd.DataFrame, *, trend_ema: int = 200, vol_window: int = 20,
                  panel_regime: Optional[np.ndarray] = None,
                  funding: Optional[np.ndarray] = None) -> Tuple[np.ndarray, List[str]]:
    """Build the meta-labeler's feature matrix — all known at the close of bar t. Describes the
    STATE when the primary fires: vol level (CtC + Garman-Klass), trend strength, momentum, drawdown,
    position-in-range, and (optional) a panel regime scalar + per-asset funding. NaN-safe."""
    o = df["open"].to_numpy(float); h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float); c = df["close"].to_numpy(float)
    cs = pd.Series(c)
    ema = cs.ewm(span=trend_ema, adjust=False).mean().to_numpy()
    rv = realized_vol(c, vol_window)
    gk = garman_klass_vol(o, h, l, c, vol_window)
    roll_hi = cs.rolling(252, min_periods=60).max().to_numpy()
    roll_lo = cs.rolling(252, min_periods=60).min().to_numpy()
    peak = cs.rolling(60, min_periods=20).max().to_numpy()
    feats = {
        "rv": rv,
        "gk_vol": gk,
        "gk_over_rv": np.where(rv > 1e-9, gk / (rv + 1e-9), 1.0),     # range vs close-to-close
        "trend_strength": np.nan_to_num(c / (ema + 1e-9) - 1.0),     # how far above/below trend
        "mom_20": np.nan_to_num(np.log(c / np.roll(c, 20)) * (np.arange(len(c)) >= 20)),
        "mom_60": np.nan_to_num(np.log(c / np.roll(c, 60)) * (np.arange(len(c)) >= 60)),
        "drawdown": np.nan_to_num((c - peak) / (peak + 1e-9)),       # ≤0; depth below 60d peak
        "pos_in_range": np.nan_to_num((c - roll_lo) / (roll_hi - roll_lo + 1e-9)),
    }
    if panel_regime is not None:
        feats["panel_corr"] = np.asarray(panel_regime, float)[:len(c)]
    if funding is not None:
        feats["funding"] = np.asarray(funding, float)[:len(c)]
    labels = list(feats.keys())
    X = np.column_stack([np.nan_to_num(feats[k], nan=0.0, posinf=0.0, neginf=0.0) for k in labels])
    return X.astype(np.float64), labels


# ───────────────────────────── sizing map ─────────────────────────────
def size_from_proba(p: np.ndarray) -> np.ndarray:
    """López de Prado sizing: size = max(0, 2p - 1). p=0.5 → 0 (no edge), p=1 → 1 (full)."""
    return np.clip(2.0 * np.asarray(p, dtype=np.float64) - 1.0, 0.0, 1.0)


# ───────────────────────────── classifier wrapper ─────────────────────────────
class MetaLabeler:
    """Shallow gradient-boosting classifier → P(target before stop). We use a CLASSIFIER (not the
    quantile-regression QuantileGBM, which models the return cone) because the meta-label is binary.
    HistGradientBoosting: fast, scale-invariant, regularized — right for this SNR and data scale."""

    def __init__(self, max_depth: int = 3, max_iter: int = 150, learning_rate: float = 0.05,
                 l2_regularization: float = 1.0, random_state: int = 0):
        self._kw = dict(max_depth=max_depth, max_iter=max_iter, learning_rate=learning_rate,
                        l2_regularization=l2_regularization, random_state=random_state)
        self.model = None
        self._const: Optional[float] = None    # fallback when a fold has one class only

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MetaLabeler":
        from sklearn.ensemble import HistGradientBoostingClassifier
        X = np.asarray(X, float); y = np.asarray(y, float)
        m = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
        X, y = X[m], y[m]
        if len(np.unique(y)) < 2:                # degenerate fold → predict the base rate
            self.model = None
            self._const = float(y.mean()) if len(y) else 0.5
            return self
        self.model = HistGradientBoostingClassifier(**self._kw).fit(X, y)
        self._const = None
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, float)
        if self.model is None:
            return np.full(X.shape[0], 0.5 if self._const is None else self._const)
        Xs = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        return self.model.predict_proba(Xs)[:, 1]
