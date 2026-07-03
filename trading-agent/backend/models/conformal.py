"""Online recalibration for the model overlay — adapt EVERY bar without touching weights.

The safe answer to "markets shift faster than retraining schedules" is NOT online gradient
descent on the network (this project's online-AWR experiment already showed per-trade weight
updates are noise at our signal-to-noise); it is online adaptation of the two scalars that
sit BETWEEN the frozen model and the money:

  1. **OnlineConformal** — adaptive conformal inference (Gibbs & Candès 2021 style): a
     multiplicative width factor on the predicted uncertainty interval, updated each bar
     from realized coverage. Interval too narrow lately (misses > α) → widen → the
     edge/uncertainty gate gets more cautious. Provably tracks target coverage under
     distribution shift, no gradients, one scalar of state.
  2. **ConvictionShrink** — an EWMA of realized sign-agreement between the model's edge
     and the outcome. Hit-rate at/below coin-flip → conviction shrinks toward 0 → the
     overlay weight collapses toward its NEUTRAL midpoint (≈ the constant-scaling control,
     which is Sharpe-preserving) — the model can only hurt the book while it is actually
     demonstrating skill. Also the antidote to the DMN's saturated ±1 head.

Both are pure, stateful-scalar, unit-tested, and symmetric: they loosen back up when the
model starts working again (at the EWMA/step timescale, not instantly)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import math


@dataclass
class OnlineConformal:
    """Per-symbol multiplicative interval-width factor tracking target coverage.

    ``update(symbol, edge, unc, realized)``: a *miss* is a realized outcome outside
    ``edge ± width_factor·unc/2``. width ×= exp(gamma·(miss − alpha)) — misses widen,
    covered bars narrow slowly, and the long-run miss rate is driven toward ``alpha``."""
    alpha: float = 0.2            # target miss rate (p10–p90 interval → 20%)
    gamma: float = 0.05           # adaptation step (per bar)
    lo: float = 0.25              # clamp: never trust the interval more than 4× ...
    hi: float = 4.0               # ... nor less than 1/4 of its nominal width
    _w: Dict[str, float] = field(default_factory=dict)

    def width_factor(self, symbol: str) -> float:
        return self._w.get(symbol, 1.0)

    def effective_unc(self, symbol: str, unc: float) -> float:
        return max(1e-9, float(unc)) * self.width_factor(symbol)

    def update(self, symbol: str, edge: float, unc: float, realized: float) -> float:
        w = self._w.get(symbol, 1.0)
        half = max(1e-9, float(unc)) * w / 2.0
        miss = 1.0 if abs(float(realized) - float(edge)) > half else 0.0
        w = w * math.exp(self.gamma * (miss - self.alpha))
        w = min(self.hi, max(self.lo, w))
        self._w[symbol] = w
        return w


@dataclass
class ConvictionShrink:
    """Per-symbol conviction multiplier from the EWMA hit-rate of sign(edge) vs sign(realized).

    shrink = clip((p̂ − 0.5)/(p_ref − 0.5), 0, 1): full conviction at hit-rate ≥ ``p_ref``
    (default 55%), zero at coin-flip. Initialised AT ``p_ref`` — the model enters with the
    trust its OOS validation earned and must keep earning it; a losing streak pulls the
    overlay toward neutral within ~``halflife`` decisions."""
    p_ref: float = 0.55
    halflife: float = 40.0        # decisions (≈ a week of 4h bars per symbol)
    _p: Dict[str, float] = field(default_factory=dict)

    def _alpha(self) -> float:
        return 1.0 - 0.5 ** (1.0 / max(self.halflife, 1e-9))

    def hit_rate(self, symbol: str) -> float:
        return self._p.get(symbol, self.p_ref)

    def shrink(self, symbol: str) -> float:
        p = self.hit_rate(symbol)
        return max(0.0, min(1.0, (p - 0.5) / max(self.p_ref - 0.5, 1e-9)))

    def update(self, symbol: str, edge: float, realized: float) -> float:
        if edge == 0.0 or realized == 0.0:
            return self.shrink(symbol)          # no information in a flat call/outcome
        hit = 1.0 if (edge > 0) == (realized > 0) else 0.0
        a = self._alpha()
        self._p[symbol] = a * hit + (1.0 - a) * self.hit_rate(symbol)
        return self.shrink(symbol)
