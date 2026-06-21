"""Intraday mean-reversion — REGIME-GATED to ranging markets.

Economic rationale: in a range (no trend), price over-extensions snap back — fade the
extreme. The catch, and the lesson from the stat-arb failure, is that mean-reversion is
*catastrophic in a trend* (you keep fading a move that keeps going). So this strategy ONLY
trades when the regime is RANGING (ADX below a threshold); in a trend it stands aside and
lets the momentum strategies work. That regime gate is the whole point.

Signal: z-score of close vs a rolling mean. Enter long when oversold (z < -entry), short when
overbought (z > +entry); exit as it reverts toward the mean (|z| < exit); hard-stop if it keeps
going (|z| > stop — a trend starting). Per-symbol, causal; the portfolio layer sizes it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

from backend.features.pipeline import ta, _safe
from backend.strategies.base import Strategy, StrategySpec


@dataclass
class MeanReversionParams:
    z_window: int = 48          # rolling window for the close z-score (≈2 days on 1h)
    entry_z: float = 2.0
    exit_z: float = 0.5         # exit once reverted within this of the mean
    stop_z: float = 4.0         # |z| beyond this ⇒ a trend is starting, bail
    adx_max: float = 20.0       # ONLY trade when ADX < this (ranging regime). 0 = no gate.
    allow_short: bool = True


class MeanReversion(Strategy):
    def __init__(self, params: Optional[MeanReversionParams] = None,
                 spec: Optional[StrategySpec] = None):
        super().__init__(spec or StrategySpec("mean_reversion", "crypto", "15m"))
        self.p = params or MeanReversionParams()

    def generate_positions(self, data: Dict[str, pd.DataFrame]) -> Dict[str, np.ndarray]:
        return {sym: self._positions_one(df) for sym, df in data.items()}

    def _positions_one(self, df: pd.DataFrame) -> np.ndarray:
        p = self.p
        close = df["close"].to_numpy(np.float64)
        n = close.shape[0]
        if n < max(p.z_window, 30) + 2:
            return np.zeros(n)
        cs = pd.Series(close)
        mu = cs.rolling(p.z_window, min_periods=p.z_window // 2).mean().to_numpy()
        sd = cs.rolling(p.z_window, min_periods=p.z_window // 2).std().to_numpy()
        z = (close - mu) / (sd + 1e-12)
        if p.adx_max > 0:
            adx_df = ta.adx(pd.Series(df["high"].to_numpy(np.float64)),
                            pd.Series(df["low"].to_numpy(np.float64)), cs, length=14)
            adx = _safe(adx_df.iloc[:, 0]).astype(np.float64) if adx_df is not None else np.zeros(n)
        else:
            adx = np.zeros(n)

        pos = np.zeros(n)
        d = 0
        for t in range(n):
            zt = z[t]
            if not np.isfinite(zt):
                pos[t] = float(d)
                continue
            ranging = (p.adx_max <= 0) or (np.isfinite(adx[t]) and adx[t] < p.adx_max)
            if d == 0:
                if ranging and zt < -p.entry_z:
                    d = 1                                   # oversold → fade up (long)
                elif ranging and p.allow_short and zt > p.entry_z:
                    d = -1                                  # overbought → fade down (short)
            elif d == 1:
                # exit: reverted to mean, kept falling past the stop, OR a trend emerged
                if zt > -p.exit_z or zt < -p.stop_z or not ranging:
                    d = 0
            elif d == -1:
                if zt < p.exit_z or zt > p.stop_z or not ranging:
                    d = 0
            pos[t] = float(d)
        return self._safe_positions(pos, n)
