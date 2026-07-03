"""Macro risk filter — a top-down de-risking overlay for the managed-beta book.

Vol-targeting reacts to REALISED vol (trailing); this adds two forward-/cross-sectional signals it
misses: (1) the VIX — the market's IMPLIED vol, which spikes before realised vol fully prints; and
(2) cross-asset CORRELATION spikes — in a crisis everything sells off together, so diversification
(the book's whole risk model) silently fails exactly when you need it. Both map to an exposure
SCALAR in [floor, 1] that the book applies on top of trend-filter + vol-target + de-gear, as a final
gross cut in risk-off states.

Deliberately simple and robust over a fragile latent-state model: at this signal-to-noise a 2-state
Markov/HMM regime tends to overfit and, in practice, just rediscovers "high vol" — which the VIX and
correlation already measure directly and interpretably. But if you DO fit a regime model, feed its
P(risk-off) straight into ``regime_scalar_from_proba`` — the overlay interface is identical.

Everything is strictly causal: each scalar at bar t is computed from information through t-1
(``shift`` defaults to 1 bar), so it can only de-risk on what was already observable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _shift(a: np.ndarray, k: int, fill: float = 1.0) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    if k <= 0:
        return a
    out = np.full_like(a, fill)
    out[k:] = a[:-k]
    return out


def _band_scalar(x: np.ndarray, lo: float, hi: float, floor: float) -> np.ndarray:
    """Linear de-risk ramp: 1.0 where x≤lo (calm), ``floor`` where x≥hi (stressed), linear between.
    NaN x → 1.0 (no signal = don't de-risk)."""
    x = np.asarray(x, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        frac = np.clip((x - lo) / max(hi - lo, 1e-9), 0.0, 1.0)
    scal = 1.0 - frac * (1.0 - floor)
    return np.where(np.isfinite(x), scal, 1.0)


def vix_risk_scalar(vix: np.ndarray, lo: float = 18.0, hi: float = 32.0,
                    floor: float = 0.3, shift: int = 1) -> np.ndarray:
    """Exposure scalar from the VIX level: full risk-on at/below ``lo``, cut to ``floor`` at/above
    ``hi``. Causal (uses VIX through t-1)."""
    return _shift(_band_scalar(vix, lo, hi, floor), shift)


def avg_pairwise_correlation(returns: np.ndarray, window: int = 63) -> np.ndarray:
    """Trailing average pairwise correlation across the book's assets (a fragility gauge). Returns
    an (n,) array; NaN until the window fills. Pairwise so it ignores not-yet-listed (NaN) names."""
    R = np.asarray(returns, dtype=np.float64)
    n, S = R.shape
    if S < 2:
        return np.full(n, np.nan)
    df = pd.DataFrame(R)
    cols = []
    for i in range(S):
        for j in range(i + 1, S):
            cols.append(df[i].rolling(window, min_periods=max(10, window // 2)).corr(df[j]))
    return pd.concat(cols, axis=1).mean(axis=1, skipna=True).to_numpy()


def correlation_risk_scalar(returns: np.ndarray, window: int = 63, lo: float = 0.3,
                            hi: float = 0.65, floor: float = 0.5, shift: int = 1) -> np.ndarray:
    """Exposure scalar from cross-asset correlation: full at/below ``lo``, cut to ``floor`` once
    average pairwise correlation spikes to/above ``hi`` (diversification breaking down). Causal."""
    avg = avg_pairwise_correlation(returns, window)
    return _shift(_band_scalar(avg, lo, hi, floor), shift)


def regime_scalar_from_proba(p_risk_off: np.ndarray, floor: float = 0.3,
                             shift: int = 1) -> np.ndarray:
    """Plug-in for an EXTERNAL regime model (e.g. a Markov/HMM P(risk-off) series): linearly maps
    p∈[0,1] → scalar 1.0 (p=0) … floor (p=1). Same overlay contract as the VIX/correlation scalars,
    so a fitted regime model can be A/B'd against them with zero other changes. Causal."""
    p = np.clip(np.nan_to_num(np.asarray(p_risk_off, dtype=np.float64), nan=0.0), 0.0, 1.0)
    return _shift(1.0 - p * (1.0 - floor), shift)


def combined_macro_scalar(vix: np.ndarray = None, returns: np.ndarray = None, *,
                          vix_kw: dict = None, corr_kw: dict = None,
                          floor: float = 0.25) -> np.ndarray:
    """Combine available macro signals into one exposure scalar = product of the VIX and correlation
    scalars (each absent → treated as 1.0), clipped to [floor, 1]. Product = either stress source
    de-risks; both stressed compounds the cut. ``returns`` provides the length when vix is None."""
    n = len(vix) if vix is not None else (len(returns) if returns is not None else 0)
    if n == 0:
        raise ValueError("combined_macro_scalar: provide vix and/or returns")
    s = np.ones(n)
    if vix is not None:
        s = s * vix_risk_scalar(np.asarray(vix, dtype=np.float64), **(vix_kw or {}))
    if returns is not None:
        s = s * correlation_risk_scalar(np.asarray(returns, dtype=np.float64), **(corr_kw or {}))
    return np.clip(s, floor, 1.0)
