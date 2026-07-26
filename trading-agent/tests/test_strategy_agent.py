"""StrategyAgent tests: reconciliation toward target weights, the small-delta skip, idempotence
(no churn once at target), the risk-halt block, and the hard live-mode safety guard."""
import asyncio
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

import pytest  # noqa: E402

from backend.agents.strategy_agent import StrategyAgent, PaperBook  # noqa: E402
from backend.strategies.managed_beta import ManagedBetaParams  # noqa: E402


def _series(close, start="2015-01-01"):
    c = np.asarray(close, float)
    ts = pd.date_range(start, periods=len(c), freq="1D")
    return pd.DataFrame({"timestamp": ts, "open": c, "high": c + 1, "low": c - 1, "close": c,
                         "volume": np.ones(len(c))})


class FakeBars:
    def __init__(self, data):
        self.data = data

    def get_daily(self, sym):
        return self.data.get(sym)            # sync; agent handles sync or coroutine


def _bars(n=300):
    rng = np.random.default_rng(1)
    up = 100 + np.cumsum(np.abs(rng.standard_normal(n)) * 0.3 + 0.2)
    up2 = 100 + np.cumsum(np.abs(rng.standard_normal(n)) * 0.3 + 0.25)
    dn = np.maximum(200 - np.cumsum(np.abs(rng.standard_normal(n)) * 0.3 + 0.2), 1)
    return {"A": _series(up), "B": _series(up2), "C": _series(dn)}


def _agent(**kw):
    return StrategyAgent(universe=["A", "B", "C"], bar_provider=FakeBars(_bars()),
                         portfolio=PaperBook(equity=100_000.0),
                         params=ManagedBetaParams(trend_ema=50), **kw)


def test_rebalance_moves_toward_targets_then_is_idempotent():
    agent = _agent()
    plan = asyncio.run(agent.rebalance_once())
    longs = {o["symbol"] for o in plan["orders"] if o["target_weight"] > 0}
    assert "A" in longs and "B" in longs           # uptrend names get bought toward target
    assert "C" in plan["skipped"] or all(o["symbol"] != "C" for o in plan["orders"])  # downtrend flat
    # after applying, a second rebalance on identical bars needs no further orders
    plan2 = asyncio.run(agent.rebalance_once())
    assert plan2["orders"] == []


def test_small_delta_is_skipped():
    agent = _agent(min_rebalance_delta=0.99)        # huge threshold → nothing is worth trading
    plan = asyncio.run(agent.rebalance_once())
    assert plan["orders"] == [] and set(plan["skipped"]) == {"A", "B", "C"}


def test_risk_halt_blocks_trading():
    class HaltedRM:
        is_halted = True
    agent = _agent(risk_manager=HaltedRM())
    plan = asyncio.run(agent.rebalance_once())
    assert plan["blocked"] == "risk_halted" and plan["orders"] == []


def test_live_mode_requires_explicit_router():
    with pytest.raises(RuntimeError):
        StrategyAgent(universe=["A", "B"], bar_provider=FakeBars(_bars()), paper=False)


def test_insufficient_data_does_not_trade():
    agent = StrategyAgent(universe=["A"], bar_provider=FakeBars({"A": _series(np.linspace(100, 110, 300))}),
                          portfolio=PaperBook(), params=ManagedBetaParams(trend_ema=50))
    plan = asyncio.run(agent.rebalance_once())
    assert plan["blocked"] == "insufficient_data" and plan["orders"] == []


# ── news overlay + 2-sleeve wiring ──
from datetime import datetime, timezone
from dataclasses import dataclass


@dataclass
class _Impact:
    severity: str = "SEVERE"
    asset: str = "A"
    direction: str = "down"
    magnitude_pct_low: float = 10.0
    magnitude_pct_high: float = 20.0
    confidence: float = 0.85
    t_min_minutes: int = 5
    t_max_minutes: int = 120
    trust_score: float = 0.9
    created_at: str = datetime.now(timezone.utc).isoformat()
    symbol_relevance: dict | None = None


def test_news_overlay_flattens_asset_against_severe_news():
    """A is an uptrend that would be bought; severe bearish news on A must flatten/halt it."""
    class News:
        def impacts_for(self, sym):
            return [_Impact(asset=sym)] if sym == "A" else []
    agent = _agent(news_provider=News())
    plan = asyncio.run(agent.rebalance_once())
    a_target = plan["targets"].get("A", 0.0)
    # the managed target for A may be >0, but after the overlay no order should OPEN a long in A
    a_orders = [o for o in plan["orders"] if o["symbol"] == "A"]
    assert all(o["target_weight"] == 0.0 for o in a_orders)
    assert "A" in plan["news"]


def test_news_overlay_records_halt_window():
    class News:
        def impacts_for(self, sym):
            return [_Impact(asset=sym)] if sym == "A" else []
    agent = _agent(news_provider=News())
    asyncio.run(agent.rebalance_once())
    assert "A" in agent._asset_halts and agent._asset_halts["A"] > 0


def test_no_news_provider_is_unchanged():
    """Without a news provider the agent behaves exactly as before (managed-only)."""
    plan = asyncio.run(_agent().rebalance_once())
    assert plan["news"] == {} and plan["blocked"] is None


def test_rebalance_once_never_touches_redis(monkeypatch):
    """REGRESSION (2026-07-09): `rebalance_once()` used to end with `_publish_portfolio()`.
    `get_redis()` reaches the REAL server when the owner's backend is up, so running this very
    test file published a fake A/B/C book over the live `strategy:paperbook:state` and wrecked it
    (equity 98,995 -> 15,698). Publishing now lives in `run()`; this pins that invariant."""
    import backend.memory.redis_client as rc

    async def _boom(*a, **k):
        raise AssertionError("rebalance_once() must not perform Redis I/O")

    # _publish_portfolio imports get_redis INSIDE the function, so patching the module attr works.
    monkeypatch.setattr(rc, "get_redis", _boom)
    plan = asyncio.run(_agent().rebalance_once())
    assert plan["blocked"] is None and plan["orders"]


def _seed_state_and_restore(agent, state: dict):
    """Put `state` in (fake) redis, run the restore, return whatever state survived."""
    import json
    from backend.memory.redis_client import get_redis

    async def _run():
        r = await get_redis()
        await r.delete("paper:reset_requested")
        await r.set("strategy:paperbook:state", json.dumps(state))
        await agent._restore_paper_book()
        return await r.get("strategy:paperbook:state")

    return asyncio.run(_run())


def test_restore_paper_book_rejects_foreign_symbols():
    """REGRESSION: a persisted book holding symbols outside this agent's universe was written by
    something else (a test sharing the Redis, another config). Adopting it silently imports that
    book's equity — exactly how the live book inherited a wrecked 15,698 from a unit-test book."""
    agent = _agent()                                   # universe A/B/C, seeded at 100k
    remaining = _seed_state_and_restore(agent, {
        "equity": 15_698.61, "initial_equity": 100_000.0,
        "positions": {"AMD": 0.03, "NVDA": 0.02},      # neither is in universe ["A","B","C"]
        "last_prices": {"AMD": 555.0},
    })
    assert agent.portfolio.equity == 100_000.0         # refused to adopt the wrecked equity
    assert not agent.portfolio.positions
    assert remaining is None                           # and purged the poisoned state


def test_restore_paper_book_accepts_own_universe():
    """The guard must not break the normal resume-across-restart path."""
    agent = _agent()
    _seed_state_and_restore(agent, {
        "equity": 98_995.08, "initial_equity": 100_000.0,
        "positions": {"A": 0.30, "B": 0.20},           # both in universe
        "last_prices": {"A": 150.0, "B": 160.0},
    })
    assert agent.portfolio.equity == 98_995.08
    assert agent.portfolio.positions == {"A": 0.30, "B": 0.20}


def test_restore_paper_book_consumes_reset_flag():
    """`paper:reset_requested` must be CLEARED once honoured — with NN_AGENT_ENABLED=false nothing
    else clears it, so a sticky flag would wipe the book on every subsequent restart."""
    import json
    from backend.memory.redis_client import get_redis
    agent = _agent()

    async def _run():
        r = await get_redis()
        await r.set("paper:reset_requested", "true")
        await r.set("strategy:paperbook:state", json.dumps({"equity": 42.0, "positions": {}}))
        await agent._restore_paper_book()
        return await r.get("paper:reset_requested"), await r.get("strategy:paperbook:state")

    flag, state = asyncio.run(_run())
    assert agent.portfolio.equity == 100_000.0   # fresh book, not the 42.0 state
    assert flag is None and state is None        # flag consumed, state purged


def test_two_sleeve_combined_targets_include_crypto():
    """With a ts_bar_provider, the combined book adds the TS-momentum crypto sleeve."""
    n = 400
    rng = np.random.default_rng(2)
    trend = 100 * np.exp(np.cumsum(0.006 + rng.normal(0, 0.005, n)))

    def _ohlc4h(c):
        ts = pd.date_range("2023-01-01", periods=len(c), freq="4h")
        return pd.DataFrame({"timestamp": ts, "open": c, "high": c * 1.002, "low": c * 0.998,
                             "close": c, "volume": np.ones(len(c)) * 1000})

    class TsBars:
        def get_bars(self, sym):
            return _ohlc4h(trend)

    agent = StrategyAgent(universe=["A", "B", "C"], bar_provider=FakeBars(_bars()),
                          portfolio=PaperBook(equity=100_000.0), params=ManagedBetaParams(trend_ema=50),
                          ts_bar_provider=TsBars(), ts_universe=["BTCUSDT", "ETHUSDT"],
                          w_managed=0.5, w_ts=0.5)
    plan = asyncio.run(agent.rebalance_once())
    assert any(s in plan["targets"] for s in ("BTCUSDT", "ETHUSDT"))
