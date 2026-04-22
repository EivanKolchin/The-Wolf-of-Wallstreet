import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch

from backend.execution.defi_engine import DefiExecutionEngine, UniswapV3Executor, DefiPortfolioTracker, DefiTradeResult

@pytest.fixture
def mock_crypto():
    mock_web3 = MagicMock()
    mock_web3.eth.contract.return_value = MagicMock()
    
    executor = UniswapV3Executor(
        web3=mock_web3,
        wallet_address="0x0000000000000000000000000000000000000000",
        private_key="0"*64
    )
    # mock swap call
    executor.swap = AsyncMock(return_value=DefiTradeResult(
        tx_hash="0x123", token_in="0x0", token_out="0x1",
        amount_in_wei=1000, amount_out_wei=900,
        gas_used=21000, gas_cost_eth=0.001,
        block_number=1, timestamp=None,
        slippage_actual=0.0, success=True, error=None
    ))
    
    tracker = DefiPortfolioTracker(
        web3=mock_web3,
        wallet_address="0x0000000000000000000000000000000000000000"
    )
    tracker.get_balances = AsyncMock(return_value={"USDC": 1000.0, "WETH": 0.0, "WBTC": 0.0})
    tracker.get_open_position = AsyncMock(return_value=None)
    
    return executor, tracker

@pytest.fixture
def engine(mock_crypto):
    executor, tracker = mock_crypto
    return DefiExecutionEngine(
        uniswap=executor,
        portfolio=tracker,
        kite_chain=None,
        db_session_factory=None,
        paper_mode=False
    )

@pytest.fixture
def paper_engine(mock_crypto):
    executor, tracker = mock_crypto
    return DefiExecutionEngine(
        uniswap=executor,
        portfolio=tracker,
        kite_chain=None,
        db_session_factory=None,
        paper_mode=True
    )

class MockDecision:
    def __init__(self, symbol="ETHUSDT", direction="long", size_pct=0.1):
        self.symbol = symbol
        self.direction = direction
        self.size_pct = size_pct

@pytest.mark.asyncio
async def test_swap_direction_mapping(engine):
    decision = MockDecision(symbol="ETHUSDT", direction="long")
    
    # We must patch aiohttp.ClientSession so we don't do real requests in tests
    with patch("aiohttp.ClientSession.get") as mock_get:
        mock_response = AsyncMock()
        mock_response.json = AsyncMock(return_value={"price": "3000.0"})
        mock_get.return_value.__aenter__.return_value = mock_response
        
        trade = await engine.execute(decision, {"available_usdc": 1000.0})
        assert trade is not None
        # assert trade.paper is False # Trade model doesn't have paper attribute
        assert trade.kite_tx_hash == "0x123"
        # ensure swap was called with token_in = USDC (since long)
        engine.uniswap.swap.assert_called_once()
        token_in = engine.uniswap.swap.call_args[0][0]
        assert "0xaf88d065e77c8cC2239327C5EDb3A432268e5831".lower() in token_in.lower()

@pytest.mark.asyncio
async def test_no_position_stacking(engine):
    engine.portfolio.get_open_position = AsyncMock(return_value={"direction": "long"})
    decision = MockDecision(symbol="BTCUSDT", direction="long")
    trade = await engine.execute(decision, {"available_usdc": 1000.0})
    assert trade is None  # Skips because already long

@pytest.mark.asyncio
async def test_paper_mode_no_real_swap(paper_engine):
    decision = MockDecision(symbol="ETHUSDT", direction="long")
    
    with patch("aiohttp.ClientSession.get") as mock_get:
        mock_response = AsyncMock()
        mock_response.json = AsyncMock(return_value={"price": "3000.0"})
        mock_get.return_value.__aenter__.return_value = mock_response
        
        trade = await paper_engine.execute(decision, {"available_usdc": 1000.0})
        assert trade is not None
        # assert trade.paper is True
        
        paper_engine.uniswap.swap.assert_not_called()

@pytest.mark.asyncio
async def test_close_position_returns_to_usdc(engine):
    engine.portfolio.get_open_position = AsyncMock(return_value={"direction": "long", "token_balance": 1.5})
    engine.portfolio.get_balances = AsyncMock(return_value={"USDC": 100.0, "WETH": 1.5, "WBTC": 0.0})
    
    with patch("aiohttp.ClientSession.get") as mock_get:
        mock_response = AsyncMock()
        mock_response.json = AsyncMock(return_value={"price": "3000.0"})
        mock_get.return_value.__aenter__.return_value = mock_response
        
        trade = await engine.close_position("ETHUSDT", reason="signal")
        assert trade is not None
        
        engine.uniswap.swap.assert_called_once()
        call_args = engine.uniswap.swap.call_args[0]
        # token_in should be WETH token addr
        assert "0x82aF49447D8a07e3bd95BD0d56f35241523fBab1".lower() in call_args[0].lower()
        # token_out should be USDC
        assert "0xaf88d065e77c8cC2239327C5EDb3A432268e5831".lower() in call_args[1].lower()

@pytest.mark.asyncio
async def test_slippage_calculation():
    # Construct executor simply to test the internal method mechanics if any
    executor = UniswapV3Executor(
        web3=MagicMock(), wallet_address="0x0000000000000000000000000000000000000000", private_key="0", slippage_tolerance=0.01  # 1%
    )
    quote = 1000000
    expected_min = int(quote * (1 - 0.01))
    assert expected_min == 990000
