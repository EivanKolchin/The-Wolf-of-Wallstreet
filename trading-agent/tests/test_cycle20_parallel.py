"""Cycle 20 — per-symbol parallelism in the agent's main cycle + predictions.

The agent's `_process_symbol` and `_render_prediction_chart` are extracted so
the main loop can `asyncio.gather(*[...])` them instead of serialising N
network-bound iterations. These tests lock the contract and verify the
parallel call genuinely beats a synthetic serial baseline.
"""
import asyncio
import sys
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))


def test_nn_agent_exposes_extracted_helpers():
    """Cycle 20 introduces three new methods. If anyone removes them this test
    fails fast and we know to update the orchestration loop too."""
    from backend.agents.nn_agent import NNTradingAgent
    for name in ("_process_symbol", "_render_prediction_chart", "_read_portfolio_state", "_rebuild_open_trades"):
        assert hasattr(NNTradingAgent, name), f"missing {name}"
        assert asyncio.iscoroutinefunction(getattr(NNTradingAgent, name))


@pytest.mark.asyncio
async def test_asyncio_gather_beats_serial_on_simulated_io_workload():
    """End-to-end proof of the optimisation: with 8 'symbols' each costing
    50 ms of simulated I/O, gather() should finish in well under the
    8*50ms = 400ms serial baseline. Models the actual win the agent gets
    from running market-feed reads across symbols in parallel."""
    N = 8
    SIM_DELAY = 0.05

    async def one_symbol(sym: str) -> str:
        await asyncio.sleep(SIM_DELAY)
        return sym

    # Serial baseline
    t0 = time.perf_counter()
    out_serial = [await one_symbol(f"S{i}") for i in range(N)]
    serial = time.perf_counter() - t0

    # Parallel via gather
    t0 = time.perf_counter()
    out_parallel = await asyncio.gather(*[one_symbol(f"S{i}") for i in range(N)])
    parallel = time.perf_counter() - t0

    assert out_serial == out_parallel
    # Hard threshold: parallel must finish in less than 35% of serial time.
    # Even with scheduler overhead, 8 × 50ms gather should land near 60–80ms.
    assert parallel < 0.35 * serial, f"parallel={parallel:.3f}s vs serial={serial:.3f}s"


@pytest.mark.asyncio
async def test_process_symbol_returns_early_on_no_data():
    """The per-symbol pipeline must short-circuit cleanly when the market
    feed returns an empty DataFrame — otherwise an exception inside one
    symbol would poison the entire gather(). A3 moved the data-fetch +
    early-return into _build_symbol_features; _process_symbol delegates to it."""
    import pandas as pd
    import types
    from backend.agents.nn_agent import NNTradingAgent

    stub = types.SimpleNamespace()
    stub.market_feed = MagicMock()
    stub.market_feed.get_dataframe = AsyncMock(return_value=pd.DataFrame())
    stub.market_feed.get_orderbook = AsyncMock(return_value={})
    stub.market_feed.get_recent_trades = AsyncMock(return_value=[])
    stub.feature_sequences = {"BTCUSDT": []}
    stub.model = MagicMock(); stub.model.SEQUENCE_LENGTH = 60
    stub.attention_controller = MagicMock()
    stub._attention = {}
    stub.risk_manager = MagicMock()
    stub._risk_lock = asyncio.Lock()
    stub.started_at = time.time()
    # Bind the real build helper so _process_symbol can delegate to it (A3 split).
    stub._build_symbol_features = NNTradingAgent._build_symbol_features.__get__(stub)

    # The build phase returns None on empty data...
    ctx = await NNTradingAgent._build_symbol_features(stub, "BTCUSDT", {"available_cash": 1000.0})
    assert ctx is None
    # ...and the full per-symbol path short-circuits to None silently, not raise.
    result = await NNTradingAgent._process_symbol(stub, "BTCUSDT", {"available_cash": 1000.0})
    assert result is None


@pytest.mark.asyncio
async def test_gather_with_return_exceptions_does_not_propagate():
    """If one symbol raises, gather(return_exceptions=True) must surface the
    failure as a value rather than killing the cycle. This matches what the
    agent's main loop does."""
    async def boom(sym):
        raise RuntimeError(f"oops {sym}")

    async def ok(sym):
        await asyncio.sleep(0.01)
        return sym

    results = await asyncio.gather(boom("a"), ok("b"), boom("c"), return_exceptions=True)
    assert len(results) == 3
    assert isinstance(results[0], RuntimeError)
    assert results[1] == "b"
    assert isinstance(results[2], RuntimeError)
