"""Regression target for the Phase 2 hybrid: the VOL-NORMALIZED forward return.

The old model predicted a 3-class direction (long/short/hold), then sizing/thresholds were
bolted on afterwards. Predicting ``r_h / (sigma * sqrt(h))`` instead gives a continuous,
volatility-comparable signal where:
  • the magnitude is a natural conviction → it drives μ̂/σ̂² (Kelly) sizing directly;
  • a cost-aware no-trade band falls out (don't trade when |edge| < cost);
  • it's comparable across calm and volatile regimes / symbols (the √h·σ denominator).

Pure numpy — importable by the offline trainer, the audit, and any model. Causal: σ uses a
trailing window only, so the only forward-looking term is the (correctly tail-masked) return.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def rolling_bar_vol(close: np.ndarray, window: int = 20) -> np.ndarray:
    """Causal per-bar volatility: rolling std of 1-bar fractional returns over a trailing
    ``window``, with a robust median fill for the warm-up so it is always > 0."""
    close = np.asarray(close, dtype=np.float64)
    n = close.shape[0]
    r1 = np.zeros(n, dtype=np.float64)
    r1[1:] = np.diff(close) / (close[:-1] + 1e-12)
    sig = pd.Series(r1).rolling(window, min_periods=5).std().bfill().to_numpy()
    finite = sig[np.isfinite(sig) & (sig > 0)]
    med = float(np.median(finite)) if finite.size else 0.005
    med = med if med > 0 else 0.005
    return np.where(np.isfinite(sig) & (sig > 0), sig, med)


def vol_normalized_forward_return(close: np.ndarray, h: int,
                                  vol_window: int = 20) -> np.ndarray:
    """(N,) target = ``(close[i+h]/close[i] - 1) / (sigma_i * sqrt(h))``.

    ``sigma_i`` is the trailing per-bar vol (causal). The last ``h`` entries are NaN (no
    full forward window) — callers mask them exactly like the triple-barrier labels.
    """
    close = np.asarray(close, dtype=np.float64)
    n = close.shape[0]
    fwd = np.full(n, np.nan, dtype=np.float64)
    if n > h:
        fwd[:-h] = close[h:] / close[:-h] - 1.0
    sig = rolling_bar_vol(close, vol_window)
    return (fwd / (sig * np.sqrt(max(h, 1)) + 1e-12)).astype(np.float32)
