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


def dollar_neutral_weights(signal: np.ndarray, hold: int, start: int, k: int) -> np.ndarray:
    """(n, S) dollar-neutral position path from a (n, S) cross-sectional SIGNAL: each rebalance
    (every ``hold`` bars from ``start``) longs the top-k / shorts the bottom-k by signal in equal
    dollars (Σ position = 0), held between rebalances. NaN signals (e.g. not-yet-listed names) are
    skipped, so it tolerates an outer-aligned panel. Shared by all cross-sectional factors.

    Positions are emitted at FULL per-name conviction (±1, the base-class [-1, 1] contract), NOT
    pre-divided by k. The capital-normalisation (1/S equal capital per name) is the backtester's
    job — ``strategy_net_returns`` averages per-symbol PnL across the universe. Pre-dividing by k
    here as well would double-normalise the book down by ~k× (a near-zero-vol book that defeats
    vol-targeting + drawdown de-gearing); emitting ±1 yields a gross-(2·quantile) dollar-neutral
    book at a real, risk-manageable scale."""
    n, S = signal.shape
    W = np.zeros((n, S))
    cur = np.zeros(S)
    rebal = set(range(start, n, hold))
    for t in range(n):
        if t in rebal:
            row = signal[t]
            valid = np.isfinite(row)
            if valid.sum() >= 2 * k:
                order = np.argsort(np.where(valid, row, -np.inf))   # ascending: weakest→strongest
                longs, shorts = order[-k:], order[:k]
                w = np.zeros(S)
                w[longs] = 1.0
                w[shorts] = -1.0
                cur = w
            else:
                cur = np.zeros(S)
        W[t] = cur
    return W


@dataclass
class XSectionalMomentumParams:
    lookback: int = 168       # momentum window (≈1 week on 1h)
    skip: int = 0             # SKIP the most recent `skip` bars (12-1 momentum skips ~1mo to
    #                           avoid short-term-reversal contamination; 0 = standard momentum)
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
        lb, sk = self.p.lookback, max(0, int(self.p.skip))
        signal = np.full((n, S), np.nan)
        if n > lb and lb > sk:
            if sk > 0:
                # 12-1 momentum: return from t-lookback to t-skip (excludes the last `skip` bars).
                signal[lb:] = logc[lb - sk: n - sk] - logc[:n - lb]
            else:
                signal[lb:] = logc[lb:] - logc[:-lb]        # trailing log return (causal)
        if self.p.kind == "reversal":
            signal = -signal

        k = self.p.k or max(1, int(round(self.p.quantile * S)))
        k = max(1, min(k, S // 2))
        W = dollar_neutral_weights(signal, self.p.hold, start=lb, k=k)
        return {s: self._safe_positions(W[:, j], n) for j, s in enumerate(syms)}


@dataclass
class ScoreParams:
    hold: int = 21            # rebalance cadence in bars (≈monthly on daily)
    warmup: int = 252         # bars to skip at the start (signal not yet meaningful)
    k: int = 0
    quantile: float = 0.2
    higher_is_better: bool = True   # True ⇒ long the HIGH-score names (value/quality); False flips


class CrossSectionalScore(Strategy):
    """Dollar-neutral long/short book from a PRECOMPUTED cross-sectional score panel — the generic
    factor engine for fundamentals (value = high earnings/book yield, quality = high gross
    profitability), where the signal is computed outside the price series. Longs the top-k / shorts
    the bottom-k by score in equal dollars (Σ=0, β≈0 by construction), rebalanced every ``hold``.

    ``signal`` is a (n, S) array aligned to ``symbols`` order and to the SAME bar calendar the
    backtester uses; NaNs (not-yet-public fundamentals, unlisted names) are skipped each rebalance,
    so it tolerates an outer-aligned panel. The signal MUST be point-in-time (no look-ahead)."""

    def __init__(self, signal: np.ndarray, symbols: list, name: str = "factor",
                 params: Optional["ScoreParams"] = None, spec: Optional[StrategySpec] = None):
        super().__init__(spec or StrategySpec(name, "stock", "1d", market_neutral=True))
        self.p = params or ScoreParams()
        self.signal = np.asarray(signal, dtype=np.float64)
        self.symbols = list(symbols)
        if self.signal.shape[1] != len(self.symbols):
            raise ValueError("signal columns must match symbols")

    def generate_positions(self, data: Dict[str, pd.DataFrame]) -> Dict[str, np.ndarray]:
        syms = list(data)
        n = len(data[syms[0]])
        if syms != self.symbols:
            raise ValueError("CrossSectionalScore: data symbol order must match the signal's")
        if len({len(data[s]) for s in syms}) != 1 or self.signal.shape[0] != n:
            raise ValueError("CrossSectionalScore needs aligned, equal-length data matching signal rows")
        S = len(syms)
        sig = self.signal if self.p.higher_is_better else -self.signal
        sig = sig.copy()
        sig[:self.p.warmup] = np.nan                      # respect warm-up (no early rebalances)
        k = self.p.k or max(1, int(round(self.p.quantile * S)))
        k = max(1, min(k, S // 2))
        W = dollar_neutral_weights(sig, self.p.hold, start=self.p.warmup, k=k)
        return {s: self._safe_positions(W[:, j], n) for j, s in enumerate(syms)}


@dataclass
class LowVolParams:
    vol_window: int = 60      # trailing realized-vol window (≈3 months on daily)
    hold: int = 21
    k: int = 0
    quantile: float = 0.2


class LowVolatility(Strategy):
    """Low-volatility anomaly — long the LOWEST-trailing-vol names / short the highest, equal
    dollars (β≈0). Low-vol stocks historically earn similar returns at much lower risk; the
    dollar-neutral version harvests that as relative-value. Price-only, classic, and LOW
    correlation to momentum — the kind of partner that lifts a combined book's Sharpe."""

    def __init__(self, params: Optional["LowVolParams"] = None, spec: Optional[StrategySpec] = None):
        super().__init__(spec or StrategySpec("low_vol", "stock", "1d", market_neutral=True))
        self.p = params or LowVolParams()

    def generate_positions(self, data: Dict[str, pd.DataFrame]) -> Dict[str, np.ndarray]:
        syms = list(data)
        if len(syms) < 4:
            return {s: np.zeros(len(data[s])) for s in syms}
        if len({len(data[s]) for s in syms}) != 1:
            raise ValueError("LowVolatility needs timestamp-aligned, equal-length data (align_panel)")
        n = len(data[syms[0]]); S = len(syms)
        closes = np.column_stack([data[s]["close"].to_numpy(np.float64) for s in syms])
        rets = np.full((n, S), np.nan)
        rets[1:] = np.log(np.maximum(closes[1:], 1e-12)) - np.log(np.maximum(closes[:-1], 1e-12))
        w = self.p.vol_window
        vol = pd.DataFrame(rets).rolling(w, min_periods=w // 2).std().to_numpy()   # trailing vol (causal)
        signal = -vol                                                              # long LOW vol
        k = self.p.k or max(1, int(round(self.p.quantile * S)))
        k = max(1, min(k, S // 2))
        W = dollar_neutral_weights(signal, self.p.hold, start=w, k=k)
        return {s: self._safe_positions(W[:, j], n) for j, s in enumerate(syms)}
