"""Strategy framework — the contract every systematic strategy implements.

The book is a PORTFOLIO of strategies (time-series momentum, cross-sectional momentum,
statistical arbitrage, mean-reversion, carry). Each strategy is a self-contained,
economically-motivated signal generator that the portfolio layer sizes and risk-manages.

The backtest-facing contract is ``generate_positions``: a VECTORISED, strictly-CAUSAL
per-bar target position per symbol in [-1, 1] (sign = side, magnitude = pre-risk
conviction). The position at bar ``t`` may use information only up to and including bar
``t``; the backtester applies it to the return from ``t -> t+1`` (so there is never any
look-ahead). Vectorising the whole series keeps walk-forward evaluation fast.

``Signal`` is the event/live view of the same intent (one decision for one symbol), used by
the live orchestrator and the ML meta-labeler; ``positions_to_signals`` bridges the two.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd


@dataclass
class Signal:
    """A strategy's trade intent for one symbol at one decision time."""
    symbol: str
    side: int                       # +1 long, -1 short, 0 flat
    strength: float = 1.0           # conviction in [0, 1] (pre risk-scaling)
    stop: float = 0.0               # stop distance as a fraction of price (0 = none)
    target: float = 0.0             # take-profit distance as a fraction of price (0 = none)
    horizon: int = 0                # expected holding in bars (0 = open-ended)
    strategy: str = ""
    meta: dict = field(default_factory=dict)


@dataclass
class StrategySpec:
    name: str
    asset_class: str                # "crypto" | "stock"
    timeframe: str                  # "5m" | "15m" | "1h" | "4h"
    market_neutral: bool = False    # True ⇒ dollar-neutral by construction (β≈0)


class Strategy(ABC):
    """Base class for a systematic strategy."""

    def __init__(self, spec: StrategySpec):
        self.spec = spec

    @property
    def name(self) -> str:
        return self.spec.name

    @abstractmethod
    def generate_positions(self, data: Dict[str, pd.DataFrame]) -> Dict[str, np.ndarray]:
        """``data``: {symbol -> OHLCV DataFrame (open/high/low/close/volume[, timestamp])}.

        Returns {symbol -> position array} aligned 1:1 to each symbol's bars, values in
        [-1, 1]. Must be causal: ``position[t]`` uses only data[:t+1]."""
        raise NotImplementedError

    # ----- shared helpers for subclasses (all causal) -----
    @staticmethod
    def _safe_positions(pos: np.ndarray, n: int) -> np.ndarray:
        """Clip to [-1, 1], replace non-finite with 0, and length-check to n bars."""
        pos = np.nan_to_num(np.asarray(pos, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        pos = np.clip(pos, -1.0, 1.0)
        if pos.shape[0] != n:
            raise ValueError(f"{n} bars expected, position array has {pos.shape[0]}")
        return pos


def positions_to_signals(strategy_name: str, data: Dict[str, pd.DataFrame],
                         positions: Dict[str, np.ndarray]) -> List[Signal]:
    """Bridge: the LAST bar's target position per symbol → a live ``Signal`` list (sign =
    side, |position| = strength). Used by the live orchestrator / meta-labeler."""
    out: List[Signal] = []
    for sym, pos in positions.items():
        if pos is None or len(pos) == 0:
            continue
        p = float(pos[-1])
        out.append(Signal(symbol=sym, side=int(np.sign(p)), strength=abs(p),
                          strategy=strategy_name))
    return out
