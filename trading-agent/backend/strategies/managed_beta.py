"""Managed beta — the honest deliverable after the alpha hunt.

The whole investigation's verdict: at retail scale on liquid assets, market-neutral ALPHA is ~0
everywhere; the thing that consistently added risk-adjusted value was the RISK machinery. So instead
of pretending to pick winners, this harvests the long-run risk premium of an asset (equity / crypto /
leveraged ETP) but DELIVERS it better than buy-and-hold: a long-biased exposure that steps out of
sustained downtrends (the canonical Faber 10-month / 200-day trend filter — trends are where crashes
happen), to be combined with portfolio-layer vol-targeting + drawdown de-gearing.

This is not alpha and it is not sold as alpha: against the market the beta is ~1. The claim is
narrow and testable — HIGHER Sharpe and SHALLOWER max drawdown than holding the same asset, i.e.
positive regression alpha *versus buy-and-hold of that asset*. Documented, robust, and exactly what
trend-following / tactical-allocation books monetise.

Emits a long-biased exposure in [down_exposure, up_exposure] (default {0, 1}: invested in an
uptrend, in cash below trend). ``down_exposure`` < 0 turns it into a two-sided trend follower
(crisis-alpha short leg); the portfolio layer sizes/vol-targets the whole book. Strictly causal:
the trend filter at bar t uses the EMA through t (known at the close), applied to the t→t+1 return.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

from backend.features.pipeline import ta, _safe
from backend.strategies.base import Strategy, StrategySpec


@dataclass
class ManagedBetaParams:
    trend_ema: int = 200          # long-term trend filter (~10 months on daily — Faber)
    fast_ema: int = 0             # optional faster confirmation (0 = off); long needs price>BOTH
    up_exposure: float = 1.0      # exposure when in the uptrend
    down_exposure: float = 0.0    # exposure below trend (0 = to cash; <0 = trend-follow short)
    band: float = 0.0             # dead-band as a fraction of price to cut whipsaw (0 = off)


class ManagedBeta(Strategy):
    def __init__(self, params: Optional[ManagedBetaParams] = None,
                 spec: Optional[StrategySpec] = None):
        super().__init__(spec or StrategySpec("managed_beta", "mixed", "1d", market_neutral=False))
        self.p = params or ManagedBetaParams()

    def generate_positions(self, data: Dict[str, pd.DataFrame]) -> Dict[str, np.ndarray]:
        return {sym: self._positions_one(df) for sym, df in data.items()}

    def _positions_one(self, df: pd.DataFrame) -> np.ndarray:
        p = self.p
        close = df["close"].to_numpy(np.float64)
        n = close.shape[0]
        warmup = max(p.trend_ema, p.fast_ema) + 2
        if n < warmup:
            return np.zeros(n)
        cs = pd.Series(close)
        ema = _safe(ta.ema(cs, length=p.trend_ema)).astype(np.float64)
        fast = _safe(ta.ema(cs, length=p.fast_ema)).astype(np.float64) if p.fast_ema > 0 else None
        up_thr = ema * (1.0 + p.band)
        dn_thr = ema * (1.0 - p.band)

        pos = np.zeros(n)
        cur = 0.0
        for t in range(n):
            if not np.isfinite(ema[t]):
                pos[t] = cur
                continue
            px = close[t]
            long_ok = px > up_thr[t] and (fast is None or px > fast[t])
            if long_ok:
                cur = p.up_exposure
            elif px < dn_thr[t]:                       # decisively below trend → step out / short
                cur = p.down_exposure
            # else: inside the dead-band → hold current exposure (whipsaw guard)
            pos[t] = cur
        return self._safe_positions(pos, n)
