"""Entity graph for cross-asset news propagation — curated edges, CONTINUOUSLY re-validated.

The naive design (a static map like MSTR→BTC) goes stale: market narratives rotate — a
stock that traded as a crypto proxy one year trades as an AI proxy the next. So every
curated edge here carries only a PRIOR weight; the LIVE weight is the prior scaled by the
edge's realized rolling return-correlation, recomputed on every refresh:

    live_weight = prior × ramp(corr)     ramp: 0 below ``min_corr``, → 1 at ``full_corr``

An edge whose correlation dies decays to zero automatically — the graph cannot keep
trading a dead narrative, and a strengthening relationship re-earns its weight without
code changes. Edges with missing/short data are likewise zero (stale data = no edge).

Usage: ``refresh(bars)`` once per rebalance with whatever daily/4h bars the agent already
fetched; ``related_sources(dst)`` lists the source assets that currently matter for a
ticker; ``derive(impact, src, dst)`` clones a source-asset NewsImpact into a weighted
impact for the destination (magnitude/confidence scaled by the live edge weight), which
the existing news overlay consumes unchanged (duck-typed fields)."""
from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Edge:
    src: str        # source ASSET key ("BTC", "ETH", "NVDA", …)
    dst: str        # destination tradable symbol ("MSTR", "TSM", …)
    kind: str       # "proxy" | "supply_chain" | "sector"
    prior: float    # hand-curated prior weight in (0, 1]


# Curated priors — the STARTING point, not the truth; live weights are correlation-gated.
CURATED_EDGES: List[Edge] = [
    # crypto → equity proxies
    Edge("BTC", "MSTR", "proxy", 0.9),          # levered BTC treasury
    Edge("BTC", "COIN", "proxy", 0.7),          # crypto-volume beta
    Edge("ETH", "COIN", "proxy", 0.5),
    Edge("BTC", "ETHUSDT", "sector", 0.5),      # BTC systemic news moves the complex
    # AI-semis supply chain / sector
    Edge("NVDA", "TSM", "supply_chain", 0.6),
    Edge("NVDA", "SMCI", "supply_chain", 0.6),
    Edge("NVDA", "AMD", "sector", 0.5),
    Edge("AMD", "TSM", "supply_chain", 0.4),
    Edge("MU", "SNDK", "sector", 0.5),          # memory/storage complex
]


def source_symbols(src: str) -> List[str]:
    """Candidate market-data / news-provider keys for a source asset."""
    s = src.upper()
    return [s, f"{s}USDT", f"{s}-USD"]


def _daily_returns(df: pd.DataFrame) -> Optional[pd.Series]:
    if df is None or "close" not in getattr(df, "columns", []):
        return None
    d = df.copy()
    if "timestamp" in d.columns:
        d = d.set_index(pd.to_datetime(d["timestamp"]))
    c = d["close"].astype(float)
    if len(c) < 3:
        return None
    daily = c.resample("1D").last().dropna() if isinstance(d.index, pd.DatetimeIndex) else c
    return daily.pct_change().dropna()


def edge_correlation(src_df: pd.DataFrame, dst_df: pd.DataFrame,
                     window: int = 90) -> Optional[float]:
    """Trailing ``window``-day return correlation between two bar frames (None if unusable)."""
    rs, rd = _daily_returns(src_df), _daily_returns(dst_df)
    if rs is None or rd is None:
        return None
    j = pd.concat([rs, rd], axis=1, join="inner").dropna().tail(window)
    if len(j) < max(20, window // 4):
        return None
    c = float(j.corr().iloc[0, 1])
    return c if np.isfinite(c) else None


@dataclass
class EntityGraph:
    edges: List[Edge] = field(default_factory=lambda: list(CURATED_EDGES))
    window: int = 90          # trailing days for the validation correlation
    min_corr: float = 0.25    # below this the edge is DEAD (weight 0)
    full_corr: float = 0.60   # at/above this the edge earns its full prior
    _weights: Dict[Tuple[str, str], float] = field(default_factory=dict)

    def _ramp(self, corr: Optional[float]) -> float:
        if corr is None or corr < self.min_corr:
            return 0.0
        span = max(self.full_corr - self.min_corr, 1e-9)
        return float(min(1.0, (corr - self.min_corr) / span))

    def refresh(self, bars: Dict[str, pd.DataFrame]) -> Dict[Tuple[str, str], float]:
        """Re-validate every edge against the bars the agent already holds. Edges whose
        source or destination has no usable data decay to 0 (stale = dead)."""
        weights: Dict[Tuple[str, str], float] = {}
        for e in self.edges:
            src_df = next((bars[k] for k in source_symbols(e.src) if k in bars), None)
            dst_df = bars.get(e.dst)
            if dst_df is None:
                dst_df = next((bars[k] for k in source_symbols(e.dst) if k in bars), None)
            corr = edge_correlation(src_df, dst_df, self.window) \
                if (src_df is not None and dst_df is not None) else None
            weights[(e.src, e.dst)] = e.prior * self._ramp(corr)
        self._weights = weights
        return dict(weights)

    def weight(self, src: str, dst: str) -> float:
        return float(self._weights.get((src.upper(), dst.upper()), 0.0))

    def related_sources(self, dst_symbol: str) -> List[Tuple[str, float]]:
        """[(source_asset, live_weight)] with weight > 0 for a destination ticker."""
        out = []
        for e in self.edges:
            if e.dst.upper() == str(dst_symbol).upper():
                w = self._weights.get((e.src, e.dst), 0.0)
                if w > 0:
                    out.append((e.src, float(w)))
        return out

    @staticmethod
    def derive(impact, src: str, dst: str, weight: float):
        """Clone a source-asset NewsImpact into a WEIGHTED impact for ``dst``. Duck-typed:
        copies the fields backend/risk/news_overlay.py reads, scaling magnitude and
        confidence by the live edge weight so a dead narrative contributes nothing."""
        w = float(max(0.0, min(1.0, weight)))
        return SimpleNamespace(
            severity=getattr(impact, "severity", "NEUTRAL"),
            direction=getattr(impact, "direction", ""),
            confidence=float(getattr(impact, "confidence", 0.0) or 0.0) * w,
            trust_score=getattr(impact, "trust_score", 0.0),
            magnitude_pct_low=float(getattr(impact, "magnitude_pct_low", 0.0) or 0.0) * w,
            magnitude_pct_high=float(getattr(impact, "magnitude_pct_high", 0.0) or 0.0) * w,
            t_max_minutes=getattr(impact, "t_max_minutes", 0),
            created_at=getattr(impact, "created_at", None),
            asset=dst,
            symbol_relevance={dst: w},
            via_edge=f"{src}->{dst}(w={w:.2f})",
        )
