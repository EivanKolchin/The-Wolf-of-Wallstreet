"""Live target-weights for the managed-beta book — the bridge from the validated backtest to a
running agent. Given recent daily bars per asset, it returns the CURRENT target portfolio weights
(signed fraction of capital per asset) by reusing exactly the pieces the backtest validated:

  per-asset trend exposure (ManagedBeta, cash below the 200-EMA)  ×
  book volatility-target leverage (causal, from the combined book's trailing vol)  ×
  macro de-risk scalar (VIX + cross-asset correlation, optional)  ×
  per-asset weight (equal among active, or inverse-vol risk-parity)

Strictly causal: everything is read at the latest available bar, so the weights are exactly what
the backtest would hold today. The agent reconciles real holdings toward these weights.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from backend.backtest.portfolio import align_panel, vol_target_scale, _bar_returns
from backend.strategies.managed_beta import ManagedBeta, ManagedBetaParams


def _managed_net(pos: np.ndarray, ret: np.ndarray, cost: float) -> np.ndarray:
    m = min(len(pos), len(ret))
    pos, ret = pos[:m], ret[:m]
    gross = np.zeros(m); gross[1:] = pos[:-1] * ret[1:]
    turn = np.zeros(m); turn[0] = abs(pos[0]); turn[1:] = np.abs(pos[1:] - pos[:-1])
    return gross - turn * cost


def target_weights(bars: Dict[str, pd.DataFrame], params: Optional[ManagedBetaParams] = None, *,
                   target_vol: float = 0.15, max_leverage: float = 2.0, bars_per_year: float = 252,
                   alloc: str = "equal", vol_window: int = 63, cost_bps: float = 5.0,
                   vix: Optional[np.ndarray] = None, macro_floor: float = 0.25,
                   vix_lo: float = 18.0, vix_hi: float = 32.0, corr_hi: float = 0.65,
                   ) -> Dict[str, float]:
    """{symbol -> signed target weight} for the book as of the latest bar. Σ|w| = current gross.
    Needs ≥2 timestamp-aligned assets with enough history for the trend + vol windows; assets too
    short / not-yet-trending get weight 0. ``vix`` (if given) must align to the aligned calendar."""
    usable = {s: d for s, d in bars.items() if d is not None and len(d) > 5}
    if len(usable) < 2:
        return {s: 0.0 for s in bars}
    aligned = align_panel(usable, how="inner")
    syms = list(aligned)
    n = len(aligned[syms[0]])
    params = params or ManagedBetaParams()
    pos = ManagedBeta(params).generate_positions(aligned)        # per-asset exposure path
    cost = cost_bps / 1e4

    rets = {s: _bar_returns(aligned[s]["close"].to_numpy()) for s in syms}
    per_asset = np.vstack([_managed_net(pos[s], rets[s], cost) for s in syms])   # (S, n)

    # per-asset weight at the latest bar: equal among active, or inverse trailing-vol (risk-parity)
    roll_vol = {s: pd.Series(rets[s]).rolling(vol_window, min_periods=max(10, vol_window // 4)).std()
                .to_numpy() for s in syms}
    active = np.array([1.0 if (np.isfinite(roll_vol[s][-1]) and roll_vol[s][-1] > 1e-9) else 0.0
                       for s in syms])
    if alloc == "risk_parity":
        base = np.array([(1.0 / roll_vol[s][-1]) if active[i] else 0.0 for i, s in enumerate(syms)])
    else:
        base = active.copy()
    w_asset = base / base.sum() if base.sum() > 0 else np.zeros(len(syms))

    # book vol-target leverage (causal) from the equal-weight combined book, latest value
    combined = per_asset.mean(axis=0)
    lev = vol_target_scale(combined, target_vol, bars_per_year, window=vol_window,
                           max_leverage=max_leverage)
    lev_now = float(lev[-1]) if lev.size else 0.0

    macro_now = 1.0
    if vix is not None or len(syms) >= 2:
        from backend.signals.macro_regime import combined_macro_scalar
        rp = np.column_stack([rets[s] for s in syms])
        v = np.asarray(vix, dtype=np.float64)[-n:] if vix is not None else None
        scal = combined_macro_scalar(vix=v, returns=rp, floor=macro_floor,
                                     vix_kw={"lo": vix_lo, "hi": vix_hi}, corr_kw={"hi": corr_hi})
        macro_now = float(scal[-1]) if scal.size else 1.0

    expo_now = np.array([float(pos[s][-1]) for s in syms])
    weights = w_asset * expo_now * lev_now * macro_now
    out = {s: 0.0 for s in bars}
    for i, s in enumerate(syms):
        out[s] = float(weights[i])
    return out
