"""Statistical arbitrage — market-neutral cointegrated pairs (stocks; 15m–1h).

Economic rationale: there is no reliable *single-name intraday* edge in these stocks (the audit
confirmed it), but RELATIVE mispricings between economically-linked names (e.g. semis) revert.
We trade the mean-reverting SPREAD of a cointegrated pair, hedged so the common factor (and the
market) cancels — the book is β≈0 and the Sharpe is pure relative-value alpha.

Two cleanly-separated pieces (so pair selection can't peek at the future):
  • ``find_cointegrated_pairs`` — Engle-Granger cointegration test on a TRAINING PREFIX only,
    returning the significant pairs to trade. (Selection look-ahead is the classic stat-arb
    backtest bug; restricting it to the prefix avoids it.)
  • ``StatArbPairs`` — the trading rule: a causal rolling-OLS hedge ratio, z-scored spread,
    enter at |z|>entry, exit at |z|<exit, hard-stop at |z|>stop (cointegration break). Positions
    are accumulated per symbol (a name can appear in several pairs); the portfolio layer
    vol-targets the book.

Requires timestamp-aligned input (use backend.backtest.portfolio.align_panel).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from backend.strategies.base import Strategy, StrategySpec

try:
    from statsmodels.tsa.stattools import coint
    HAS_STATSMODELS = True
except Exception:  # pragma: no cover
    HAS_STATSMODELS = False


def ou_half_life(spread: np.ndarray) -> float:
    """Ornstein-Uhlenbeck mean-reversion half-life (bars): regress Δspread on lagged spread,
    half-life = -ln(2)/β for β<0, else inf (not mean-reverting). Filters slow/non-reverting pairs."""
    s = np.asarray(spread, dtype=np.float64)
    s = s[np.isfinite(s)]
    if s.size < 30:
        return float("inf")
    lag, ds = s[:-1], np.diff(s)
    beta = np.polyfit(lag - lag.mean(), ds, 1)[0]
    return float(-np.log(2.0) / beta) if beta < -1e-9 else float("inf")


def find_cointegrated_pairs(data: Dict[str, pd.DataFrame], train_frac: float = 0.5,
                            pmax: float = 0.05, max_pairs: int = 10,
                            use_log: bool = True) -> List[Tuple[str, str]]:
    """Engle-Granger cointegrated pairs, tested on the FIRST ``train_frac`` of the (aligned)
    history only → the selection never sees the trading period. Returns up to ``max_pairs``
    (A, B) sorted by significance. Falls back to highest-correlation pairs without statsmodels."""
    syms = list(data)
    n = min(len(data[s]) for s in syms)
    cut = max(50, int(n * train_frac))
    series = {}
    for s in syms:
        c = data[s]["close"].to_numpy(np.float64)[:cut]
        series[s] = np.log(np.maximum(c, 1e-12)) if use_log else c
    scored: List[Tuple[float, str, str]] = []
    for i in range(len(syms)):
        for j in range(i + 1, len(syms)):
            a, b = series[syms[i]], series[syms[j]]
            if HAS_STATSMODELS:
                try:
                    pval = float(coint(a, b)[1])
                except Exception:
                    continue
                if pval <= pmax:
                    scored.append((pval, syms[i], syms[j]))
            else:
                corr = float(np.corrcoef(a, b)[0, 1])
                if abs(corr) >= 0.8:
                    scored.append((1.0 - abs(corr), syms[i], syms[j]))   # lower = better
    scored.sort(key=lambda x: x[0])
    return [(a, b) for _, a, b in scored[:max_pairs]]


@dataclass
class StatArbParams:
    hedge_window: int = 168       # rolling-OLS hedge-ratio window
    z_window: int = 168           # rolling z-score window for the spread
    entry_z: float = 2.0
    exit_z: float = 0.5
    stop_z: float = 3.5           # |z| beyond this => assume cointegration broke, get out
    use_log: bool = True
    # Rolling mean-reversion gate (defends against cointegration breakdown — the #1 stat-arb
    # killer): only ENTER when the spread is currently mean-reverting, measured by a trailing
    # Ornstein-Uhlenbeck half-life. A pair whose spread has started trending (half-life blows
    # up) is skipped, so we never keep fading a relationship that has structurally broken.
    recheck_window: int = 480     # trailing window for the rolling half-life estimate
    max_half_life: float = 240.0  # only trade if the spread reverts within this many bars (0 = gate off)


class StatArbPairs(Strategy):
    def __init__(self, pairs: List[Tuple[str, str]], params: Optional[StatArbParams] = None,
                 spec: Optional[StrategySpec] = None):
        super().__init__(spec or StrategySpec("stat_arb", "stock", "15m", market_neutral=True))
        self.pairs = list(pairs)
        self.p = params or StatArbParams()

    def _pair_weights(self, a_px: np.ndarray, b_px: np.ndarray):
        """Causal per-bar (wA, wB) for one pair (gross-normalised to 1)."""
        p = self.p
        a = np.log(np.maximum(a_px, 1e-12)) if p.use_log else a_px.astype(np.float64)
        b = np.log(np.maximum(b_px, 1e-12)) if p.use_log else b_px.astype(np.float64)
        sa, sb = pd.Series(a), pd.Series(b)
        cov = sa.rolling(p.hedge_window, min_periods=p.hedge_window // 2).cov(sb)
        var = sb.rolling(p.hedge_window, min_periods=p.hedge_window // 2).var()
        beta = (cov / var.replace(0.0, np.nan)).to_numpy()
        spread = a - beta * b
        ss = pd.Series(spread)
        mu = ss.rolling(p.z_window, min_periods=p.z_window // 2).mean().to_numpy()
        sd = ss.rolling(p.z_window, min_periods=p.z_window // 2).std().to_numpy()
        z = (spread - mu) / (sd + 1e-12)
        n = len(a)

        # Rolling OU half-life gate: regress Δspread on lagged spread over a trailing window;
        # half-life = -ln2/slope for slope<0 (mean-reverting), else inf. Only ENTER when the
        # spread is currently reverting fast enough — skips pairs whose cointegration has broken.
        if p.max_half_life and p.max_half_life > 0:
            ds = pd.Series(np.diff(spread, prepend=spread[:1]))
            slag = pd.Series(np.concatenate([[spread[0]], spread[:-1]]))
            w = p.recheck_window
            cov_hl = slag.rolling(w, min_periods=w // 2).cov(ds).to_numpy()
            var_hl = slag.rolling(w, min_periods=w // 2).var().to_numpy()
            with np.errstate(divide="ignore", invalid="ignore"):
                slope = cov_hl / var_hl
                hl = np.where(slope < -1e-9, -np.log(2.0) / slope, np.inf)
            revert_ok = np.isfinite(hl) & (hl > 0) & (hl < p.max_half_life)
        else:
            revert_ok = np.ones(n, dtype=bool)

        wA = np.zeros(n); wB = np.zeros(n)
        d = 0
        for t in range(n):
            zt, bt = z[t], beta[t]
            if not (np.isfinite(zt) and np.isfinite(bt)):
                d = 0
                continue
            if d == 0:
                if revert_ok[t] and zt > p.entry_z:
                    d = -1                      # spread rich -> short spread (short A, long B)
                elif revert_ok[t] and zt < -p.entry_z:
                    d = 1                       # spread cheap -> long spread (long A, short B)
            elif abs(zt) < p.exit_z or abs(zt) > p.stop_z:
                d = 0
            if d != 0:
                g = 1.0 + abs(bt)               # gross-normalise the pair to ~1
                wA[t] = d / g
                wB[t] = -d * bt / g
        return wA, wB

    def generate_positions(self, data: Dict[str, pd.DataFrame]) -> Dict[str, np.ndarray]:
        syms = list(data)
        if not self.pairs:
            return {s: np.zeros(len(data[s])) for s in syms}
        lengths = {len(data[s]) for s in syms}
        if len(lengths) != 1:
            raise ValueError("StatArbPairs needs timestamp-aligned, equal-length data "
                             "(call backend.backtest.portfolio.align_panel first)")
        n = lengths.pop()
        acc = {s: np.zeros(n) for s in syms}
        for a_sym, b_sym in self.pairs:
            if a_sym not in data or b_sym not in data:
                continue
            wA, wB = self._pair_weights(data[a_sym]["close"].to_numpy(np.float64),
                                        data[b_sym]["close"].to_numpy(np.float64))
            acc[a_sym] += wA
            acc[b_sym] += wB
        return {s: self._safe_positions(acc[s], n) for s in syms}
