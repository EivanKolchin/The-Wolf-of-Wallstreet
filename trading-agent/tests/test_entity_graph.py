"""Entity graph — correlation-validated edge weights, narrative-shift decay, news propagation."""
import asyncio
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from backend.signals.entity_graph import (  # noqa: E402
    Edge, EntityGraph, edge_correlation, source_symbols,
)


def _bars(ret: np.ndarray, start="2024-01-01") -> pd.DataFrame:
    close = 100.0 * np.cumprod(1.0 + ret)
    return pd.DataFrame({"timestamp": pd.date_range(start, periods=len(close), freq="D"),
                         "open": close, "high": close, "low": close, "close": close,
                         "volume": np.ones_like(close)})


def _corr_pair(n=200, rho=0.8, seed=0):
    rng = np.random.default_rng(seed)
    a = 0.02 * rng.standard_normal(n)
    b = rho * a + np.sqrt(max(0.0, 1 - rho ** 2)) * 0.02 * rng.standard_normal(n)
    return a, b


def test_edge_weight_tracks_correlation():
    a, b = _corr_pair(rho=0.85, seed=1)
    g = EntityGraph(edges=[Edge("BTC", "MSTR", "proxy", 0.9)])
    w = g.refresh({"BTCUSDT": _bars(a), "MSTR": _bars(b)})
    assert w[("BTC", "MSTR")] > 0.5                     # strong corr → near-full prior

    a2, b2 = _corr_pair(rho=0.0, seed=2)                # narrative dead
    w2 = g.refresh({"BTCUSDT": _bars(a2), "MSTR": _bars(b2)})
    assert w2[("BTC", "MSTR")] == 0.0                   # edge decayed to zero
    assert g.related_sources("MSTR") == []              # and no longer propagates


def test_missing_or_short_data_kills_edge():
    g = EntityGraph(edges=[Edge("BTC", "MSTR", "proxy", 0.9)])
    a, _ = _corr_pair()
    assert g.refresh({"BTCUSDT": _bars(a)})[("BTC", "MSTR")] == 0.0     # dst missing
    short = _bars(np.zeros(5))
    assert g.refresh({"BTCUSDT": short, "MSTR": short})[("BTC", "MSTR")] == 0.0


def test_source_symbol_resolution():
    assert "BTCUSDT" in source_symbols("BTC") and "BTC-USD" in source_symbols("BTC")
    a, b = _corr_pair(rho=0.9, seed=3)
    g = EntityGraph(edges=[Edge("BTC", "MSTR", "proxy", 0.8)])
    w = g.refresh({"BTC-USD": _bars(a), "MSTR": _bars(b)})   # yfinance-style key works too
    assert w[("BTC", "MSTR")] > 0.4


def test_derive_scales_magnitude_and_confidence():
    from types import SimpleNamespace
    imp = SimpleNamespace(severity="SEVERE", direction="bearish", confidence=0.9,
                          trust_score=0.8, magnitude_pct_low=5.0, magnitude_pct_high=10.0,
                          t_max_minutes=120, created_at=None)
    d = EntityGraph.derive(imp, "BTC", "MSTR", 0.5)
    assert d.severity == "SEVERE" and d.direction == "bearish"          # class preserved
    assert d.confidence == 0.45 and d.magnitude_pct_high == 5.0         # scaled by weight
    assert d.asset == "MSTR" and d.symbol_relevance == {"MSTR": 0.5}
    assert "BTC->MSTR" in d.via_edge


def test_view_exposes_nodes_and_live_vs_dead_edges():
    a, b = _corr_pair(rho=0.9, seed=5)
    g = EntityGraph(edges=[Edge("BTC", "MSTR", "proxy", 0.9),
                           Edge("NVDA", "TSM", "supply_chain", 0.6)])
    g.refresh({"BTCUSDT": _bars(a), "MSTR": _bars(b)})       # NVDA/TSM have no data → dead
    v = g.view()
    ids = {n["id"] for n in v["nodes"]}
    assert {"BTC", "MSTR", "NVDA", "TSM"} <= ids
    by_key = {(e["src"], e["dst"]): e for e in v["edges"]}
    assert by_key[("BTC", "MSTR")]["weight"] > 0             # live, corr-validated
    assert by_key[("NVDA", "TSM")]["weight"] == 0.0          # dead (no data)
    assert by_key[("NVDA", "TSM")]["prior"] == 0.6           # prior still visible
    kinds = {n["id"]: n["kind"] for n in v["nodes"]}
    assert kinds["BTC"] == "crypto" and kinds["TSM"] == "equity"


def test_agent_propagates_source_news_to_proxy():
    """A BTC impact reaches MSTR through the agent's news path, weighted by the live edge."""
    from backend.agents.strategy_agent import StrategyAgent
    from types import SimpleNamespace

    a, b = _corr_pair(rho=0.9, seed=4)
    graph = EntityGraph(edges=[Edge("BTC", "MSTR", "proxy", 0.9)])
    graph.refresh({"BTCUSDT": _bars(a), "MSTR": _bars(b)})

    btc_impact = SimpleNamespace(severity="SEVERE", direction="bearish", confidence=0.9,
                                 trust_score=0.9, magnitude_pct_low=6.0,
                                 magnitude_pct_high=12.0, t_max_minutes=180, created_at=None)

    def provider(symbol):
        return [btc_impact] if symbol == "BTCUSDT" else []

    agent = StrategyAgent(universe=["MSTR"], bar_provider=None,
                          news_provider=provider, entity_graph=graph)
    loop = asyncio.get_event_loop_policy().new_event_loop()
    impacts = loop.run_until_complete(agent._news_impacts("MSTR"))
    assert len(impacts) == 1                              # MSTR has no own news; 1 propagated
    assert impacts[0].severity == "SEVERE"
    assert 0 < impacts[0].magnitude_pct_high < 12.0       # scaled down by the edge weight
    # and a symbol with no edges gets nothing
    assert loop.run_until_complete(agent._news_impacts("SPY")) == []
