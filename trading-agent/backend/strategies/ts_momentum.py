"""Crypto time-series momentum / breakout — the first validated edge (1h bars).

Economic rationale: trends in crypto persist (time-series momentum is one of the most robust,
most-replicated systematic edges across assets — Moskowitz–Ooi–Pedersen). We enter on a
Donchian channel breakout in the direction of the higher-timeframe trend, and protect the
trade with a Chandelier (ATR trailing) stop + a shorter opposite-channel exit (the classic
turtle exit). Position sizing/vol-targeting is handled at the portfolio layer, so the strategy
emits a clean directional path in {-1, 0, +1}.

All indicators are causal (prior-bar windows are shifted by 1) and computed with the pure
``ta`` shim from backend.features.pipeline (no talib dependency). The stop is path-dependent,
so it's applied in a small per-symbol loop — fine at 1h granularity. Parameters are exposed on
a dataclass so the ML meta-layer can later adapt them per regime.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

from backend.features.pipeline import ta, _safe
from backend.strategies.base import Strategy, StrategySpec


@dataclass
class TSMomentumParams:
    entry_channel: int = 48       # Donchian breakout lookback (≈2 days on 1h)
    exit_channel: int = 24        # opposite-channel (turtle) exit lookback
    atr_period: int = 14
    atr_mult: float = 3.0         # Chandelier trailing-stop width in ATRs (symmetric mode)
    ema_trend: int = 100          # higher-timeframe trend filter
    use_trend_filter: bool = True
    adx_min: float = 0.0          # require ADX ≥ this to enter (0 = off)
    allow_short: bool = True
    # ── CONVEX EXITS (asymmetric stops: cut losers fast, let winners run) ──────────────
    # Off by default = the validated symmetric Chandelier (atr_mult both ways). When ON, a trade
    # uses a TIGHT initial stop until it proves itself (moves `activation_atr` ATR in favour),
    # then switches to a WIDE trailing stop FLOORED at breakeven — so losers are cut small and
    # winners are allowed to run into the fat right tail. This deliberately shapes the return
    # distribution toward positive skew ("small red / big green"); it also raises turnover, so
    # A/B it on the backtest harness before trusting the exact params live.
    convex_exits: bool = False
    initial_atr_mult: float = 1.5     # tight stop from the favourable extreme, pre-activation
    trail_atr_mult: float = 4.0       # wide trailing stop once the trade is in profit
    activation_atr: float = 1.0       # ATR of favourable excursion needed to widen + lock breakeven

    def long_stop(self, entry: float, peak: float, atr: float) -> float:
        """Stop price for a long. Symmetric Chandelier unless convex_exits, in which case:
        tight (initial_atr_mult) until `peak` is activation_atr·ATR above entry, then a wide
        (trail_atr_mult) trail floored at breakeven so a winner can't revert to a loss."""
        if not self.convex_exits:
            return peak - self.atr_mult * atr
        if (peak - entry) >= self.activation_atr * atr:
            return max(entry, peak - self.trail_atr_mult * atr)
        return peak - self.initial_atr_mult * atr

    def short_stop(self, entry: float, trough: float, atr: float) -> float:
        """Mirror of long_stop for a short (``trough`` = lowest low since entry)."""
        if not self.convex_exits:
            return trough + self.atr_mult * atr
        if (entry - trough) >= self.activation_atr * atr:
            return min(entry, trough + self.trail_atr_mult * atr)
        return trough + self.initial_atr_mult * atr


class TSMomentumBreakout(Strategy):
    def __init__(self, params: Optional[TSMomentumParams] = None,
                 spec: Optional[StrategySpec] = None):
        super().__init__(spec or StrategySpec("ts_momentum_1h", "crypto", "1h"))
        self.p = params or TSMomentumParams()

    def generate_positions(self, data: Dict[str, pd.DataFrame]) -> Dict[str, np.ndarray]:
        return {sym: self._positions_one(df) for sym, df in data.items()}

    def _positions_one(self, df: pd.DataFrame) -> np.ndarray:
        p = self.p
        high = df["high"].to_numpy(np.float64)
        low = df["low"].to_numpy(np.float64)
        close = df["close"].to_numpy(np.float64)
        n = close.shape[0]
        warmup = max(p.entry_channel, p.ema_trend, p.atr_period) + 2
        if n < warmup:
            return np.zeros(n)

        hs, ls, cs = pd.Series(high), pd.Series(low), pd.Series(close)
        # Donchian channels over PRIOR bars only (shift(1) → strictly causal)
        upper = hs.rolling(p.entry_channel).max().shift(1).to_numpy()
        lower = ls.rolling(p.entry_channel).min().shift(1).to_numpy()
        exit_lo = ls.rolling(p.exit_channel).min().shift(1).to_numpy()
        exit_hi = hs.rolling(p.exit_channel).max().shift(1).to_numpy()
        atr = _safe(ta.atr(hs, ls, cs, length=p.atr_period)).astype(np.float64)
        ema = _safe(ta.ema(cs, length=p.ema_trend)).astype(np.float64)
        adx = None
        if p.adx_min > 0:
            adx_df = ta.adx(hs, ls, cs, length=14)
            adx = _safe(adx_df.iloc[:, 0]).astype(np.float64) if adx_df is not None else None

        pos = np.zeros(n)
        d = 0           # current direction
        peak = 0.0      # favourable extreme since entry (for the trailing stop)
        entry_px = 0.0  # entry price (for the convex activation + breakeven floor)
        for t in range(n):
            px = close[t]
            if not np.isfinite(upper[t]) or atr[t] <= 0:
                pos[t] = float(d)
                continue
            trend_long = (not p.use_trend_filter) or px > ema[t]
            trend_short = (not p.use_trend_filter) or px < ema[t]
            adx_ok = adx is None or (np.isfinite(adx[t]) and adx[t] >= p.adx_min)
            if d == 0:
                if px > upper[t] and trend_long and adx_ok:
                    d, peak, entry_px = 1, high[t], px
                elif p.allow_short and px < lower[t] and trend_short and adx_ok:
                    d, peak, entry_px = -1, low[t], px
            elif d == 1:
                peak = max(peak, high[t])
                if px < p.long_stop(entry_px, peak, atr[t]) or px < exit_lo[t]:   # stop or turtle exit
                    d = 0
            elif d == -1:
                peak = min(peak, low[t])
                if px > p.short_stop(entry_px, peak, atr[t]) or px > exit_hi[t]:
                    d = 0
            pos[t] = float(d)
        return self._safe_positions(pos, n)
