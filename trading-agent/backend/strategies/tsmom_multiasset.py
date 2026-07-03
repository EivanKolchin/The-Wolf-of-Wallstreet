"""Daily multi-asset long-short time-series momentum (Moskowitz–Ooi–Pedersen style).

The third sleeve the 2-sleeve book needs: managed-beta is long-biased (harvests the
risk premium, bleeds in sustained bears) and the 4h crypto TS-momentum is crypto-only.
Classic daily TSMOM on the SAME broad ETF+crypto universe is the most replicated
diversifier in the literature — long assets with positive trailing 3/6/12-month
returns, short those with negative, each scaled inversely to its own volatility. It
is symmetric (crisis alpha: it gets short in sustained bears — exactly the regime
where the long-biased sleeve parks in cash and earns nothing) and its documented
correlation to long-only beta is near zero over full cycles.

Signal per asset (strictly causal, close-of-bar t):
    sig_t   = mean_k( sign(close_t / close_{t-k} - 1) )    for k in lookbacks
    scale_t = clip(target_asset_vol / trailing_ann_vol_t, 0, 1)
    pos_t   = sig_t * scale_t          (then a dead-band suppresses tiny rebalances)

The per-asset inverse-vol scaling is the MOP construction (their 40%/σ): every asset
contributes comparable risk, so BTC doesn't dominate TLT. The dead-band keeps daily
turnover (the tax that killed every intraday idea here) near zero — the position only
moves when the target drifts materially or the sign flips.

Portfolio sizing/vol-targeting stays at the portfolio layer, like every other sleeve.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from backend.strategies.base import Strategy, StrategySpec


@dataclass
class TSMOMParams:
    lookbacks: Tuple[int, ...] = (63, 126, 252)  # ~3/6/12 months in trading days
    vol_window: int = 63          # trailing window for the per-asset vol estimate
    target_asset_vol: float = 0.15  # per-asset annualised vol target (MOP-style scaling)
    ppy: float = 252.0            # periods/year for annualising the trailing vol
    dead_band: float = 0.10       # min |Δposition| to actually rebalance (turnover guard);
    #                               a sign flip always trades regardless of the band
    min_history_factor: float = 1.05  # need max(lookbacks)*factor bars before trading


class TSMOMMultiAsset(Strategy):
    def __init__(self, params: Optional[TSMOMParams] = None,
                 spec: Optional[StrategySpec] = None):
        super().__init__(spec or StrategySpec("tsmom_multiasset", "mixed", "1d",
                                              market_neutral=False))
        self.p = params or TSMOMParams()

    def generate_positions(self, data: Dict[str, pd.DataFrame]) -> Dict[str, np.ndarray]:
        return {sym: self._positions_one(df) for sym, df in data.items()}

    def _positions_one(self, df: pd.DataFrame) -> np.ndarray:
        p = self.p
        close = df["close"].to_numpy(np.float64)
        n = close.shape[0]
        warmup = int(max(max(p.lookbacks), p.vol_window) * p.min_history_factor)
        if n <= warmup:
            return np.zeros(n)

        cs = pd.Series(close)
        # mean of per-lookback trailing-return signs; NaN (pre-listing / warm-up) rows
        # contribute nothing (treated as 0 conviction for that lookback).
        signs = np.zeros(n, dtype=np.float64)
        valid = np.zeros(n, dtype=np.float64)
        for k in p.lookbacks:
            r = cs.pct_change(k).to_numpy()
            ok = np.isfinite(r)
            signs[ok] += np.sign(r[ok])
            valid += ok.astype(np.float64)
        sig = np.where(valid > 0, signs / np.maximum(valid, 1.0), 0.0)

        # trailing annualised vol at t (uses returns through t only)
        r1 = cs.pct_change().to_numpy()
        vol = (pd.Series(r1).rolling(p.vol_window, min_periods=max(10, p.vol_window // 4))
               .std().to_numpy()) * np.sqrt(p.ppy)
        with np.errstate(divide="ignore", invalid="ignore"):
            scale = np.where(np.isfinite(vol) & (vol > 1e-9),
                             np.clip(p.target_asset_vol / vol, 0.0, 1.0), 0.0)
        target = sig * scale
        target[:warmup] = 0.0

        # dead-band rebalancing: hold the current position until the target drifts by
        # more than the band or flips sign — the turnover guard.
        pos = np.zeros(n)
        cur = 0.0
        for t in range(n):
            tgt = target[t]
            flipped = (np.sign(tgt) != np.sign(cur)) and (tgt != 0.0 or cur != 0.0)
            if flipped or abs(tgt - cur) >= p.dead_band:
                cur = tgt
            pos[t] = cur
        return self._safe_positions(pos, n)


def tsmom_target_weights(bars: Dict[str, pd.DataFrame],
                         params: Optional[TSMOMParams] = None) -> Dict[str, float]:
    """{symbol -> latest signed target weight} for the live sleeve (pre portfolio
    vol-target): each asset's last-bar TSMOM position, equal-capital across the
    symbols that have enough history. Strictly causal — reads only the latest bar."""
    p = params or TSMOMParams()
    strat = TSMOMMultiAsset(p)
    usable = {s: d for s, d in bars.items()
              if d is not None and len(d) > max(p.lookbacks)}
    if not usable:
        return {s: 0.0 for s in bars}
    pos = strat.generate_positions(usable)
    k = float(len(usable))
    out = {s: 0.0 for s in bars}
    for s in usable:
        out[s] = float(pos[s][-1]) / k
    return out
