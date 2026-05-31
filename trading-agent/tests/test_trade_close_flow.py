import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

from backend.agents.nn_agent import NNTradingAgent


class DummyExecutionEngine:
    def __init__(self):
        self._callback = None

    def set_trade_closed_callback(self, callback):
        self._callback = callback

    async def emit_closed(self, trade, pnl_pct: float):
        assert self._callback is not None
        await self._callback(trade, pnl_pct)


@pytest.mark.asyncio
async def test_trade_close_callback_updates_model_and_logs_outcome():
    model = MagicMock()
    model.SEQUENCE_LENGTH = 60
    model.online_update = MagicMock()
    model.check_and_rollback = MagicMock(return_value=False)

    execution_engine = DummyExecutionEngine()
    news_queue = SimpleNamespace(redis=AsyncMock(), get_nowait=AsyncMock(return_value=None))

    agent = NNTradingAgent(
        market_feed=MagicMock(),
        feature_builder=MagicMock(),
        regime_detector=MagicMock(),
        model=model,
        risk_manager=MagicMock(),
        execution_engine=execution_engine,
        news_queue=news_queue,
        severe_flag=SimpleNamespace(value=False),
        symbols=["BTCUSDT"],
    )

    agent.training_backbone = MagicMock()
    seq = np.ones((model.SEQUENCE_LENGTH, 62), dtype=np.float32)
    agent.open_trade_context["BTCUSDT"] = seq
    agent.open_trades["BTCUSDT"] = SimpleNamespace(direction=SimpleNamespace(value="long"))

    trade = SimpleNamespace(
        id="trade-1",
        symbol="BTCUSDT",
        direction=SimpleNamespace(value="long"),
    )
    await execution_engine.emit_closed(trade, pnl_pct=0.012)

    model.online_update.assert_called_once()
    agent.training_backbone.record_outcome.assert_called_once()
    assert "BTCUSDT" not in agent.open_trade_context
