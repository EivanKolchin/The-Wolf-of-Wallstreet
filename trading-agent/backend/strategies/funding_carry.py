"""Funding carry — delta-neutral perp funding capture (the uncorrelated diversifier).

Economic rationale: perpetual-futures funding is a recurring cash flow paid between longs and
shorts every ~8h. By holding the RECEIVING side of a DELTA-NEUTRAL book (e.g. short perp +
long spot when funding is positive), you harvest that funding with ~no price exposure. The PnL
is the funding stream, not price direction — so it is **uncorrelated to the momentum strategies**,
which is exactly what a momentum-heavy book needs for diversification (and it earns in flat
markets where trend/momentum make nothing).

This strategy's PnL is funding accrual, not ``position × price-return``, so it implements
``generate_returns`` directly. It expects each symbol's frame to carry a ``funding_rate`` column
(the 8h rate, forward-filled to each bar); the harness attaches it from ``derivatives_feed`` and
sets ``periods_per_bar`` from the timeframe. With no funding column it produces no return (inert).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

from backend.strategies.base import Strategy, StrategySpec


@dataclass
class FundingCarryParams:
    entry_rate: float = 0.0001    # only harvest when |8h funding| exceeds this (~1bp/8h)
    fee_bps: float = 4.0          # round-trip cost charged when the receiving side flips
    periods_per_bar: float = 1.0  # funding periods (8h) elapsed per bar (harness sets by timeframe)


class FundingCarry(Strategy):
    def __init__(self, params: Optional[FundingCarryParams] = None,
                 spec: Optional[StrategySpec] = None):
        super().__init__(spec or StrategySpec("funding_carry", "crypto", "1h", market_neutral=True))
        self.p = params or FundingCarryParams()

    def generate_positions(self, data: Dict[str, pd.DataFrame]) -> Dict[str, np.ndarray]:
        # Delta-neutral: no net PRICE position. PnL comes from generate_returns (funding).
        return {s: np.zeros(len(df)) for s, df in data.items()}

    def generate_returns(self, data: Dict[str, pd.DataFrame]):
        p = self.p
        series = []
        for sym, df in data.items():
            if "funding_rate" not in df.columns:
                continue
            f = df["funding_rate"].to_numpy(np.float64)
            n = f.shape[0]
            # Receiving side: short the perp when funding>0 (longs pay shorts), long when <0.
            side = np.where(np.abs(f) > p.entry_rate, -np.sign(f), 0.0)
            net = np.zeros(n)
            # Holding the receiving side into bar t earns |funding| accrued over that bar
            # (prorated by periods_per_bar); charge cost when the side flips (turnover).
            recv = -side[:-1] * f[:-1] * p.periods_per_bar          # = |f| when on the receiving side
            turn = np.abs(np.diff(side, prepend=0.0))[1:]
            net[1:] = recv - turn * (p.fee_bps / 1e4)
            series.append(net)
        if not series:
            return None
        L = min(len(x) for x in series)
        return np.mean(np.vstack([x[-L:] for x in series]), axis=0)   # equal-weight across symbols
