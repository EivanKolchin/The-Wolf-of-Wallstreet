"""Phase 10 — correlation / lead-lag math (standalone foundation).

Provides:
  * `correlation_matrix_from_prices`: Pearson correlation across multiple assets' return series.
  * `lead_lag(a, b, max_lag)`: cross-correlation at integer lags. Returns the lag that
    maximises |corr(a, shift(b, lag))| + the correlation value (positive lag = a leads b).
  * `CorrelationMatrix`: rolling helper that computes a fresh snapshot and persists
    each one APPEND-ONLY (never overwrites past entries), so structural shifts in
    cross-asset relationships are preserved.

The full RelationalLayer (per-asset encoders + cross-asset attention) lands when the
model architecture is reworked (Phase 10 model). These helpers are the math
foundation: usable now for diagnostics, dual-ETP-hedge confidence, and the
eventual relational layer's correlation/lead-lag features.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

try:
    from backend.memory.database import CorrelationSnapshot
    HAS_DB = True
except Exception:
    HAS_DB = False


def _returns(prices) -> np.ndarray:
    p = np.asarray(prices, dtype=float)
    if p.size < 2:
        return np.zeros(0, dtype=float)
    base = np.where(p[:-1] == 0, np.nan, p[:-1])
    r = np.diff(p) / base
    r = r[np.isfinite(r)]
    return r


def correlation_matrix_from_prices(price_map: dict[str, list]) -> tuple[list[str], np.ndarray]:
    """Pearson correlation of returns across assets.
    Returns `(symbols_in_order, NxN matrix)`."""
    symbols = sorted(price_map.keys())
    if not symbols:
        return symbols, np.zeros((0, 0))
    series = [_returns(price_map[s]) for s in symbols]
    n = min((len(s) for s in series), default=0)
    if n < 5:
        # Not enough data — return an identity matrix as a sane neutral default.
        return symbols, np.eye(len(symbols))
    mat = np.vstack([s[-n:] for s in series])
    return symbols, np.corrcoef(mat)


def lead_lag(a, b, max_lag: int = 5) -> dict:
    """Cross-correlation of two return series at integer lags.

    Returns `{"lag": L, "corr": c, "abs_corr": |c|}` where the chosen `L` maximises
    `|corr(a, shift(b, L))|`. Convention: positive lag means *a leads b* (today's `a`
    correlates with `b` L bars later); negative lag means b leads a.
    """
    ra = _returns(a)
    rb = _returns(b)
    n = min(len(ra), len(rb))
    if n < 10:
        return {"lag": 0, "corr": 0.0, "abs_corr": 0.0}
    ra = ra[-n:]
    rb = rb[-n:]

    best_lag = 0
    best_corr = 0.0
    for L in range(-int(max_lag), int(max_lag) + 1):
        if L == 0:
            x, y = ra, rb
        elif L > 0:
            x, y = ra[:-L], rb[L:]       # a's value compared to b L-bars later -> a leads
        else:                            # L < 0: b leads a
            x, y = ra[-L:], rb[:L]
        if len(x) < 5 or np.std(x) <= 0 or np.std(y) <= 0:
            continue
        c = float(np.corrcoef(x, y)[0, 1])
        if abs(c) > abs(best_corr):
            best_corr = c
            best_lag = L
    return {"lag": int(best_lag), "corr": float(best_corr), "abs_corr": float(abs(best_corr))}


class CorrelationMatrix:
    """Append-only correlation memory across the asset universe.

    `update(price_map)` computes a fresh snapshot and appends a `CorrelationSnapshot`
    row — never overwrites past entries, so the structural-shift history is preserved.
    """

    def __init__(self, db_session_factory=None):
        self.db_session_factory = db_session_factory
        self.symbols: list[str] = []
        self.matrix: np.ndarray = np.zeros((0, 0))
        self.version: int = 0

    def compute(self, price_map: dict[str, list]) -> tuple[list[str], np.ndarray]:
        self.symbols, self.matrix = correlation_matrix_from_prices(price_map)
        self.version += 1
        return self.symbols, self.matrix

    async def update_and_persist(self, price_map: dict[str, list]) -> bool:
        self.compute(price_map)
        return await self.persist()

    async def persist(self) -> bool:
        if not self.db_session_factory or not HAS_DB or self.matrix.size == 0:
            return False
        try:
            payload = {"symbols": list(self.symbols), "matrix": self.matrix.tolist()}
            async with self.db_session_factory() as session:
                snap = CorrelationSnapshot(version=int(self.version), matrix=payload)
                session.add(snap)
                await session.commit()
            return True
        except Exception:
            return False

    def to_dict(self) -> dict:
        return {"version": self.version, "symbols": list(self.symbols), "matrix": self.matrix.tolist()}

    def pair(self, a: str, b: str) -> Optional[float]:
        """Return the correlation between two symbols in the latest snapshot, or None."""
        if a not in self.symbols or b not in self.symbols or self.matrix.size == 0:
            return None
        ia, ib = self.symbols.index(a), self.symbols.index(b)
        return float(self.matrix[ia, ib])
