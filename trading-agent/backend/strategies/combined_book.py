"""Live target-weights for the TWO-SLEEVE book = managed-beta  +  4h regime-gated TS-momentum.

This is the bridge from the validated diversification result (scripts/combined_book_research.py:
managed-beta ⊕ TS-momentum, correlation +0.10, combined Sharpe 0.65→0.80, maxDD −27%→−19.5% over
2022-26) to a running paper book. Each sleeve is built EXACTLY as its backtest validated it, then
the two are merged by capital fraction:

    managed_beta sleeve   → managed_book.target_weights(daily bars)          [ETFs + crypto-daily]
    TS-momentum sleeve    → ts_momentum_target_weights(4h crypto bars)       [crypto perps]
    combined[sym] = w_managed · mb[sym]  +  w_ts · ts[sym]   (summed where a symbol is in both)

Both sub-books are internally vol-targeted, so 50/50 capital ≈ equal risk (the validated split).
Strictly causal: every value is read at the latest bar. Nothing here executes — it returns target
weights for the StrategyAgent's paper reconciliation loop (which still refuses live routing).
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from backend.backtest.portfolio import align_panel, vol_target_scale, _bar_returns
from backend.risk.funding_cap import FundingCapConfig, funding_scale
from backend.strategies.ts_momentum import TSMomentumBreakout, TSMomentumParams
from backend.strategies.managed_book import target_weights as managed_target_weights
from backend.strategies.managed_beta import ManagedBetaParams


def ts_momentum_target_weights(bars: Dict[str, pd.DataFrame],
                               params: Optional[TSMomentumParams] = None, *,
                               target_vol: float = 0.12, max_leverage: float = 2.0,
                               bars_per_year: float = 6 * 365, vol_window: int = 60,
                               alloc: str = "inverse_vol", cost_bps: float = 15.0,
                               funding_annual: Optional[Dict[str, float]] = None,
                               funding_cfg: Optional[FundingCapConfig] = None,
                               ) -> Dict[str, float]:
    """{symbol -> signed target weight} for the TS-momentum sleeve as of the latest 4h bar.

    Per-symbol exposure ∈ {−1,0,+1} from the validated TSMomentumBreakout (4h, ADX-gated, L/S),
    weighted across symbols (equal among active, or inverse trailing-vol), then scaled by the
    book-level vol-target leverage — the same overlay the portfolio backtester applies.

    ``funding_annual``: optional {symbol -> SMOOTHED annualized funding rate} (feed the
    FundingCapTracker's EMA, never the raw last print). Each symbol on the PAYING side of
    funding is scaled down along the continuous funding_scale ramp — sign-aware, so a short
    that receives crowded-long funding is never capped."""
    params = params or TSMomentumParams(entry_channel=48, exit_channel=24, atr_mult=3.0,
                                        ema_trend=50, adx_min=25, allow_short=True)
    usable = {s: d for s, d in bars.items() if d is not None and len(d) > max(params.ema_trend, 60)}
    if len(usable) < 2:
        return {s: 0.0 for s in bars}
    strat = TSMomentumBreakout(params)
    pos = strat.generate_positions(usable)                       # {sym: path in {-1,0,1}}
    syms = list(usable)
    cost = cost_bps / 1e4

    rets, per_asset = {}, []
    for s in syms:
        c = usable[s]["close"].to_numpy(np.float64)
        r = _bar_returns(c)
        p = np.asarray(pos[s], dtype=np.float64)
        m = min(len(p), len(r)); p, r = p[:m], r[:m]
        gross = np.zeros(m); gross[1:] = p[:-1] * r[1:]
        turn = np.zeros(m); turn[0] = abs(p[0]); turn[1:] = np.abs(p[1:] - p[:-1])
        rets[s] = r
        per_asset.append(gross - turn * cost)
    L = min(len(x) for x in per_asset)
    R = np.vstack([x[-L:] for x in per_asset])                   # (S, L)

    roll_vol = {s: pd.Series(rets[s]).rolling(vol_window, min_periods=max(10, vol_window // 4)).std()
                .to_numpy() for s in syms}
    active = np.array([1.0 if (np.isfinite(roll_vol[s][-1]) and roll_vol[s][-1] > 1e-9) else 0.0
                       for s in syms])
    if alloc == "inverse_vol":
        base = np.array([(1.0 / roll_vol[s][-1]) if active[i] else 0.0 for i, s in enumerate(syms)])
    else:
        base = active.copy()
    w_asset = base / base.sum() if base.sum() > 0 else np.zeros(len(syms))

    combined = R.mean(axis=0)
    lev = vol_target_scale(combined, target_vol, bars_per_year, window=vol_window,
                           max_leverage=max_leverage)
    lev_now = float(lev[-1]) if lev.size else 0.0
    expo_now = np.array([float(pos[s][-1]) for s in syms])

    weights = w_asset * expo_now * lev_now
    out = {s: 0.0 for s in bars}
    fcfg = funding_cfg or FundingCapConfig()
    for i, s in enumerate(syms):
        w = float(weights[i])
        if funding_annual and s in funding_annual and w != 0.0:
            carry_cost = float(np.sign(w)) * float(funding_annual[s])   # + = we pay
            w *= funding_scale(carry_cost, fcfg.lo_annual, fcfg.hi_annual, fcfg.floor)
        out[s] = w
    return out


def combined_target_weights(managed_bars: Dict[str, pd.DataFrame],
                            ts_bars: Dict[str, pd.DataFrame], *,
                            w_managed: float = 0.5, w_ts: float = 0.5,
                            managed_params: Optional[ManagedBetaParams] = None,
                            ts_params: Optional[TSMomentumParams] = None,
                            managed_kwargs: Optional[dict] = None,
                            ts_kwargs: Optional[dict] = None) -> Dict[str, float]:
    """Merge the two validated sleeves into one book of target weights, scaled by capital fraction.

    ``managed_bars``: daily OHLCV for the managed-beta universe (ETFs + crypto-daily + optional VIX).
    ``ts_bars``:      4h OHLCV for the TS-momentum crypto-perp universe.
    Symbols present in both sleeves (e.g. a crypto name) have their weights SUMMED — the net book
    exposure is what gets reconciled. Returns {symbol -> signed target weight}; Σ|w| = gross."""
    mb = managed_target_weights(managed_bars, managed_params, **(managed_kwargs or {})) if managed_bars else {}
    ts = ts_momentum_target_weights(ts_bars, ts_params, **(ts_kwargs or {})) if ts_bars else {}
    out: Dict[str, float] = {}
    for s, w in mb.items():
        out[s] = out.get(s, 0.0) + w_managed * float(w)
    for s, w in ts.items():
        out[s] = out.get(s, 0.0) + w_ts * float(w)
    return out
