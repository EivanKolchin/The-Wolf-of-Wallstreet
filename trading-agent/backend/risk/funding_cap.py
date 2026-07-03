"""Funding-aware leverage scaling for the perp sleeve — smoothed, sign-aware, whipsaw-proof.

Leverage on perps is financed by the funding rate: a levered position that PAYS funding
gives back the very CAGR the leverage was meant to buy. But funding is noisy — the
instantaneous (premium-index) rate can flip from strongly positive to strongly negative
within hours — so a naive "cap leverage the moment funding crosses X" rule would
aggressively de-lever and re-lever the book on every flip (whipsaw = pure turnover cost).

Anti-whipsaw design, two layers, both causal:
  1. **EMA smoothing** of the funding prints (default half-life ≈ 3 days of 8h prints):
     the cap responds to the *sustained* carry cost, not the last print.
  2. **A continuous ramp instead of a threshold**: the scale falls LINEARLY from 1.0
     (at ``lo_annual``) to ``floor`` (at ``hi_annual``). There is no discrete switch to
     oscillate around — small changes in smoothed funding produce small changes in
     leverage. (A two-threshold hysteresis state machine is unnecessary once the input
     is smoothed and the output is continuous.)

Sign-awareness: funding only *costs* when you are on the paying side — long positions
pay positive funding, shorts pay negative funding. ``carry cost = sign(position) ×
funding``. A short that RECEIVES the crowded longs' funding is never capped; that carry
is income, exactly what the funding-carry literature harvests.

Pure numpy + a tiny stateful tracker for live use. Unit-tested offline.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

# Binance USD-M funding prints every 8h → 3/day.
PRINTS_PER_YEAR = 3.0 * 365.0


@dataclass
class FundingCapConfig:
    ema_halflife_prints: float = 9.0    # ≈ 3 days of 8h prints — the smoothing layer
    lo_annual: float = 0.10             # smoothed pay-side carry below this → scale 1.0
    hi_annual: float = 0.40             # at/above this → scale = floor (linear in between)
    floor: float = 0.25                 # never scale below this (position can still de-risk
    #                                     via the trend/vol layers; funding alone won't flatten)
    prints_per_year: float = PRINTS_PER_YEAR


def ema_funding(prints: np.ndarray, halflife_prints: float) -> np.ndarray:
    """Causal EMA over a series of funding prints. out[i] uses prints[0..i] only."""
    prints = np.asarray(prints, dtype=np.float64).ravel()
    n = prints.size
    out = np.zeros(n)
    if n == 0:
        return out
    alpha = 1.0 - 0.5 ** (1.0 / max(float(halflife_prints), 1e-9))
    acc = prints[0]
    out[0] = acc
    for i in range(1, n):
        acc = alpha * prints[i] + (1.0 - alpha) * acc
        out[i] = acc
    return out


def funding_scale(carry_cost_annual, lo: float = 0.10, hi: float = 0.40,
                  floor: float = 0.25):
    """Leverage multiplier in [floor, 1] from the SMOOTHED annualized pay-side carry cost.

    cost ≤ lo → 1.0; cost ≥ hi → floor; linear ramp in between. Negative cost
    (we RECEIVE funding) → 1.0. Vectorized; scalar in → scalar out."""
    c = np.asarray(carry_cost_annual, dtype=np.float64)
    hi = max(float(hi), float(lo) + 1e-9)
    frac = np.clip((c - lo) / (hi - lo), 0.0, 1.0)
    out = 1.0 - frac * (1.0 - float(floor))
    return float(out) if out.ndim == 0 else out


@dataclass
class FundingCapTracker:
    """Live per-symbol tracker: feed funding prints (or the current premium-index rate at
    each rebalance), read a leverage scale for the CURRENT position sign.

    ``update`` is idempotent per print — feed whatever cadence you have; the EMA half-life
    is expressed in prints, so 8h prints give the intended ~3-day memory."""
    cfg: FundingCapConfig = field(default_factory=FundingCapConfig)
    _ema: Dict[str, float] = field(default_factory=dict)

    def update(self, symbol: str, funding_print: float) -> float:
        alpha = 1.0 - 0.5 ** (1.0 / max(self.cfg.ema_halflife_prints, 1e-9))
        prev = self._ema.get(symbol)
        cur = float(funding_print) if prev is None else alpha * float(funding_print) + (1 - alpha) * prev
        self._ema[symbol] = cur
        return cur

    def ema_annual(self, symbol: str) -> Optional[float]:
        e = self._ema.get(symbol)
        return None if e is None else e * self.cfg.prints_per_year

    def scale(self, symbol: str, position_sign: float) -> float:
        """Multiplier in [floor, 1] for a position of the given sign. Unknown symbol → 1.0
        (no data = no cap; the vol/risk layers still govern)."""
        ann = self.ema_annual(symbol)
        if ann is None or position_sign == 0:
            return 1.0
        cost = float(np.sign(position_sign)) * ann        # + = we pay, − = we receive
        return funding_scale(cost, self.cfg.lo_annual, self.cfg.hi_annual, self.cfg.floor)


def funding_scale_series(funding_prints: np.ndarray, position_signs: np.ndarray,
                         cfg: Optional[FundingCapConfig] = None) -> np.ndarray:
    """Backtest helper: per-print leverage scales for a position-sign path aligned to the
    prints. Causal (EMA through print i scales print i's holding)."""
    cfg = cfg or FundingCapConfig()
    ema = ema_funding(funding_prints, cfg.ema_halflife_prints) * cfg.prints_per_year
    cost = np.sign(np.asarray(position_signs, dtype=np.float64)) * ema
    return funding_scale(cost, cfg.lo_annual, cfg.hi_annual, cfg.floor)
