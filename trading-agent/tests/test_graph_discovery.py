"""LLM edge discovery — parsing, whitelist/prior guards, persistence, and the
propose→validator-disposes safety property (a discovered edge earns weight only by correlating)."""
import asyncio
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.agents.graph_discovery import (  # noqa: E402
    GraphDiscoveryAgent, _parse_edges, MAX_DISCOVERED_PRIOR, DEFAULT_DST_WHITELIST,
)
from backend.signals.entity_graph import Edge, EntityGraph  # noqa: E402


class FakeLLM:
    def __init__(self, response):
        self.response = response
        self.calls = 0

    async def generate_text(self, prompt, tier="haiku", max_tokens=300, json_mode=True):
        self.calls += 1
        return self.response


def _run(coro):
    return asyncio.get_event_loop_policy().new_event_loop().run_until_complete(coro)


def test_parse_valid_and_filters_bad():
    raw = '''Here you go:
    [{"src":"BTC","dst":"MSTR","kind":"proxy","confidence":0.9,"why":"treasury"},
     {"src":"BTC","dst":"NOTREAL","kind":"proxy","confidence":0.9},
     {"src":"X","dst":"X","kind":"sector","confidence":0.5},
     {"src":"NVDA","dst":"TSM","kind":"supply_chain","confidence":0.6}]'''
    edges = _parse_edges(raw, DEFAULT_DST_WHITELIST)
    keys = {(e.src, e.dst) for e in edges}
    assert ("BTC", "MSTR") in keys and ("NVDA", "TSM") in keys
    assert ("BTC", "NOTREAL") not in keys          # off-whitelist dst dropped
    assert not any(e.src == e.dst for e in edges)   # self-edge dropped


def test_prior_capped():
    raw = '[{"src":"BTC","dst":"MSTR","kind":"proxy","confidence":1.0}]'
    e = _parse_edges(raw, DEFAULT_DST_WHITELIST)[0]
    assert e.prior <= MAX_DISCOVERED_PRIOR + 1e-9   # discovered edges never start "curated-strong"


def test_no_llm_is_noop(tmp_path):
    agent = GraphDiscoveryAgent(llm_service=None, store_path=str(tmp_path / "e.json"))
    assert _run(agent.discover(["BTC ETF approved"])) == []


def test_malformed_llm_response(tmp_path):
    agent = GraphDiscoveryAgent(FakeLLM("sorry, no JSON here"), store_path=str(tmp_path / "e.json"))
    assert _run(agent.discover(["headline"])) == []


def test_discover_persists_and_dedupes(tmp_path):
    store = tmp_path / "e.json"
    llm = FakeLLM('[{"src":"BTC","dst":"COIN","kind":"proxy","confidence":0.8}]')
    agent = GraphDiscoveryAgent(llm, store_path=str(store))
    _run(agent.discover(["Coinbase volumes track bitcoin"]))
    assert store.exists()
    # same edge again → dedupe (refresh), not stack
    _run(agent.discover(["again"]))
    rows = agent._load()
    assert len([e for e in rows if (e.src, e.dst) == ("BTC", "COIN")]) == 1


def test_apply_to_graph_adds_as_prior_only():
    """The discovered edge is added, but its LIVE weight is still 0 until it correlates —
    the validator, not the LLM, decides whether it ever trades."""
    llm = FakeLLM('[{"src":"BTC","dst":"COIN","kind":"proxy","confidence":0.9}]')
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        agent = GraphDiscoveryAgent(llm, store_path=str(Path(td) / "e.json"))
        _run(agent.discover(["headline"]))
        g = EntityGraph(edges=[])                   # empty graph
        added = agent.apply_to(g)
        assert added == 1 and any(e.dst == "COIN" for e in g.edges)

        # DECORRELATED data → the discovered edge stays at weight 0 (LLM cannot force a trade)
        rng = np.random.default_rng(0)
        def bars(seed):
            r = rng.standard_normal(200) * 0.02
            c = 100 * np.cumprod(1 + r)
            return pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=200, freq="D"),
                                 "open": c, "high": c, "low": c, "close": c, "volume": np.ones(200)})
        g.refresh({"BTCUSDT": bars(1), "COIN": bars(2)})   # independent series
        assert g.weight("BTC", "COIN") == 0.0

        # CORRELATED data → the same discovered edge now earns weight
        a = rng.standard_normal(200) * 0.02
        ca = 100 * np.cumprod(1 + a)
        cb = 100 * np.cumprod(1 + 0.9 * a + 0.1 * rng.standard_normal(200) * 0.02)
        def df(c):
            return pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=200, freq="D"),
                                 "open": c, "high": c, "low": c, "close": c, "volume": np.ones(200)})
        g.refresh({"BTCUSDT": df(ca), "COIN": df(cb)})
        assert g.weight("BTC", "COIN") > 0.0
