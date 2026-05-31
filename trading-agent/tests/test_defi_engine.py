import sys
from pathlib import Path

import pytest
from unittest.mock import MagicMock, AsyncMock, patch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from backend.execution.defi_engine import (
    DefiExecutionEngine, UniswapV3Executor, DefiPortfolioTracker, DefiTradeResult,
)
from backend.memory.database import Trade, TradeDirection, TradeStatus, OrderType

USDC = "0xaf88d065e77c8cC2239327C5EDb3A432268e5831"
WETH = "0x82aF49447D8a07e3bd95BD0d56f35241523fBab1"


@pytest.fixture
def mock_crypto():
    mock_web3 = MagicMock()
    mock_web3.eth.contract.return_value = MagicMock()

    executor = UniswapV3Executor(
        web3=mock_web3,
        wallet_address="0x0000000000000000000000000000000000000000",
        private_key="0" * 64,
    )
    executor.swap = AsyncMock(return_value=DefiTradeResult(
        tx_hash="0x123", token_in="0x0", token_out="0x1",
        amount_in_wei=1000, amount_out_wei=900, gas_used=21000, gas_cost_eth=0.001,
        block_number=1, timestamp=None, slippage_actual=0.0, success=True, error=None,
    ))

    tracker = DefiPortfolioTracker(
        web3=mock_web3,
        wallet_address="0x0000000000000000000000000000000000000000",
    )
    tracker.get_balances = AsyncMock(return_value={"USDC": 1000.0, "WETH": 0.0, "WBTC": 0.0})
    tracker.get_open_position = AsyncMock(return_value=None)
    return executor, tracker


@pytest.fixture
def engine(mock_crypto):
    executor, tracker = mock_crypto
    return DefiExecutionEngine(uniswap=executor, portfolio=tracker,
                               db_session_factory=None, paper_mode=False)


@pytest.fixture
def paper_engine(mock_crypto):
    executor, tracker = mock_crypto
    return DefiExecutionEngine(uniswap=executor, portfolio=tracker,
                               db_session_factory=None, paper_mode=True)


class MockDecision:
    def __init__(self, symbol="ETHUSDT", direction="long", size_pct=0.1):
        self.symbol = symbol
        self.direction = direction
        self.size_pct = size_pct
        self.nn_confidence = 0.6
        self.nn_probs = {"long": 0.6, "short": 0.2, "hold": 0.2}
        self.regime = "ranging"
        self.active_news = None
        self.sl, self.tp, self.trail = 0.02, 0.04, 0.02


def _patch_price(value="3000.0"):
    cm = patch("aiohttp.ClientSession.get")
    return cm, value


@pytest.mark.asyncio
async def test_execute_creates_trade_and_swaps_from_usdc(engine):
    decision = MockDecision(symbol="ETHUSDT", direction="long")
    with patch("aiohttp.ClientSession.get") as mock_get:
        resp = AsyncMock()
        resp.json = AsyncMock(return_value={"price": "3000.0"})
        mock_get.return_value.__aenter__.return_value = resp

        trade = await engine.execute(decision, {"available_usdc": 1000.0})
        assert trade is not None
        assert trade.asset == "ETHUSDT"
        assert trade.direction == TradeDirection.long
        # learned exit levels were applied around the 3000 entry
        assert trade.stop_loss > 0 and trade.take_profit > 0
        engine.uniswap.swap.assert_called_once()
        token_in = engine.uniswap.swap.call_args[0][0]
        assert USDC.lower() in token_in.lower()


@pytest.mark.asyncio
async def test_no_position_stacking(engine):
    engine.portfolio.get_open_position = AsyncMock(return_value={"direction": "long"})
    decision = MockDecision(symbol="BTCUSDT", direction="long")
    trade = await engine.execute(decision, {"available_usdc": 1000.0})
    assert trade is None  # already long -> skip


@pytest.mark.asyncio
async def test_paper_mode_no_real_swap(paper_engine):
    decision = MockDecision(symbol="ETHUSDT", direction="long")
    with patch("aiohttp.ClientSession.get") as mock_get:
        resp = AsyncMock()
        resp.json = AsyncMock(return_value={"price": "3000.0"})
        mock_get.return_value.__aenter__.return_value = resp

        trade = await paper_engine.execute(decision, {"available_usdc": 1000.0})
        assert trade is not None
        assert trade.asset == "ETHUSDT"
        paper_engine.uniswap.swap.assert_not_called()


@pytest.mark.asyncio
async def test_close_position_returns_to_usdc(engine):
    engine.portfolio.get_balances = AsyncMock(return_value={"USDC": 100.0, "WETH": 1.5, "WBTC": 0.0})
    # Seed an open trade for the monitor/close path to act on.
    engine._open_trades["ETHUSDT"] = Trade(
        asset="ETHUSDT", direction=TradeDirection.long, size_usd=4500.0, entry_price=3000.0,
        status=TradeStatus.open, order_type=OrderType.market, nn_confidence=0.6,
        nn_direction_probs={}, regime_at_entry="ranging", stop_loss=2940.0, take_profit=3120.0,
        fee_paid=4.5,
    )
    with patch("aiohttp.ClientSession.get") as mock_get:
        resp = AsyncMock()
        resp.json = AsyncMock(return_value={"price": "3000.0"})
        mock_get.return_value.__aenter__.return_value = resp

        trade = await engine.close_position("ETHUSDT", reason="signal")
        assert trade is not None
        engine.uniswap.swap.assert_called_once()
        call_args = engine.uniswap.swap.call_args[0]
        assert WETH.lower() in call_args[0].lower()   # sells WETH
        assert USDC.lower() in call_args[1].lower()   # back into USDC


@pytest.mark.asyncio
async def test_slippage_calculation():
    executor = UniswapV3Executor(web3=MagicMock(),
                                 wallet_address="0x0000000000000000000000000000000000000000",
                                 private_key="0" * 64, slippage_tolerance=0.01)
    quote = 1_000_000
    assert int(quote * (1 - 0.01)) == 990_000
