"""Cross-sectional momentum — dollar-neutral crypto (β≈0).

Economic rationale: relative strength persists. Rather than bet on market direction (which a
regime flip kills — the trap that broke every directional model here), we rank the universe by
trailing momentum and go LONG the strongest k / SHORT the weakest k in EQUAL DOLLARS. The book
is dollar-neutral by construction, so it profits from *dispersion* (winners beating losers) and
carries ≈no market beta — the Sharpe is pure alpha and far more regime-stable than direction.

This is a CROSS-SECTIONAL strategy: it ranks across symbols at each rebalance, so it requires
timestamp-aligned input (use backend.backtest.portfolio.align_panel). Positions are emitted as
per-symbol paths (+1/k for longs, -1/k for shorts, 0 otherwise), held between rebalances; the
portfolio layer vol-targets the whole book.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

from backend.strategies.base import Strategy, StrategySpec


@dataclass
class XSectionalMomentumParams:
    lookback: int = 168       # momentum window (≈1 week on 1h)
    hold: int = 24            # rebalance every N bars (≈daily on 1h)
    k: int = 0                # names per side; 0 ⇒ derive from `quantile`
    quantile: float = 0.30    # top/bottom fraction when k == 0
    kind: str = "momentum"    # "momentum" (long winners) | "reversal" (long losers)


class CrossSectionalMomentum(Strategy):
    def __init__(self, params: Optional[XSectionalMomentumParams] = None,
                 spec: Optional[StrategySpec] = None):
        super().__init__(spec or StrategySpec("xs_momentum", "crypto", "1h", market_neutral=True))
        self.p = params or XSectionalMomentumParams()

    def generate_positions(self, data: Dict[str, pd.DataFrame]) -> Dict[str, np.ndarray]:
        syms = list(data)
        if len(syms) < 4:                                   # need breadth for a long-short book
            return {s: np.zeros(len(data[s])) for s in syms}
        lengths = {len(data[s]) for s in syms}
        if len(lengths) != 1:
            raise ValueError("CrossSectionalMomentum needs timestamp-aligned, equal-length data "
                             "(call backend.backtest.portfolio.align_panel first)")
        n = lengths.pop()
        S = len(syms)
        closes = np.column_stack([data[s]["close"].to_numpy(np.float64) for s in syms])  # (n, S)
        logc = np.log(np.maximum(closes, 1e-12))
        lb = self.p.lookback
        signal = np.full((n, S), np.nan)
        if n > lb:
            signal[lb:] = logc[lb:] - logc[:-lb]            # trailing log return (causal)
        if self.p.kind == "reversal":
            signal = -signal

        k = self.p.k or max(1, int(round(self.p.quantile * S)))
        k = max(1, min(k, S // 2))
        W = np.zeros((n, S))
        cur = np.zeros(S)
        rebal = set(range(lb, n, self.p.hold))
        for t in range(n):
            if t in rebal:
                row = signal[t]
                valid = np.isfinite(row)
                if valid.sum() >= 2 * k:
                    order = np.argsort(np.where(valid, row, -np.inf))  # ascending: weakest→strongest
                    longs, shorts = order[-k:], order[:k]
                    w = np.zeros(S)
                    w[longs] = 1.0 / k                       # equal-dollar long/short ⇒ Σw = 0
                    w[shorts] = -1.0 / k
                    cur = w
                else:
                    cur = np.zeros(S)
            W[t] = cur
        return {s: self._safe_positions(W[:, j], n) for j, s in enumerate(syms)}
