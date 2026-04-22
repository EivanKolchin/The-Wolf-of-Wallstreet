import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
import structlog
from web3 import Web3
from web3.exceptions import ContractLogicError
from typing import Optional, Dict, Any
import aiohttp
from backend.memory.database import AgentEvent

logger = structlog.get_logger(__name__)

# Constants
ARBITRUM_RPC_URL = "https://arb1.arbitrum.io/rpc"
UNISWAP_V3_ROUTER = Web3.to_checksum_address("0xE592427A0AEce92De3Edee1F18E0157C05861564")
UNISWAP_V3_QUOTER = Web3.to_checksum_address("0xb27308f9F90D607463bb33eA1BeBb41C27CE5AB6")
USDC_ADDRESS = Web3.to_checksum_address("0xaf88d065e77c8cC2239327C5EDb3A432268e5831")
WETH_ADDRESS = Web3.to_checksum_address("0x82aF49447D8a07e3bd95BD0d56f35241523fBab1")
WBTC_ADDRESS = Web3.to_checksum_address("0x2f2a2543B76A4166549F7aaB2e75Bef0aefC5B0f")
POOL_FEE_TIER = 3000

ERC20_ABI = json.loads('''[
    {"constant":false,"inputs":[{"name":"_spender","type":"address"},{"name":"_value","type":"uint256"}],"name":"approve","outputs":[{"name":"","type":"bool"}],"payable":false,"stateMutability":"nonpayable","type":"function"},
    {"constant":true,"inputs":[{"name":"_owner","type":"address"}],"name":"balanceOf","outputs":[{"name":"balance","type":"uint256"}],"payable":false,"stateMutability":"view","type":"function"},
    {"constant":true,"inputs":[{"name":"_owner","type":"address"},{"name":"_spender","type":"address"}],"name":"allowance","outputs":[{"name":"","type":"uint256"}],"payable":false,"stateMutability":"view","type":"function"}
]''')

UNISWAP_V3_ROUTER_ABI = json.loads('''[
    {"inputs":[{"components":[{"internalType":"address","name":"tokenIn","type":"address"},{"internalType":"address","name":"tokenOut","type":"address"},{"internalType":"uint24","name":"fee","type":"uint24"},{"internalType":"address","name":"recipient","type":"address"},{"internalType":"uint256","name":"deadline","type":"uint256"},{"internalType":"uint256","name":"amountIn","type":"uint256"},{"internalType":"uint256","name":"amountOutMinimum","type":"uint256"},{"internalType":"uint160","name":"sqrtPriceLimitX96","type":"uint160"}],"internalType":"struct ISwapRouter.ExactInputSingleParams","name":"params","type":"tuple"}],"name":"exactInputSingle","outputs":[{"internalType":"uint256","name":"amountOut","type":"uint256"}],"stateMutability":"payable","type":"function"}
]''')

UNISWAP_V3_QUOTER_ABI = json.loads('''[
    {"inputs":[{"internalType":"address","name":"tokenIn","type":"address"},{"internalType":"address","name":"tokenOut","type":"address"},{"internalType":"uint24","name":"fee","type":"uint24"},{"internalType":"uint256","name":"amountIn","type":"uint256"},{"internalType":"uint160","name":"sqrtPriceLimitX96","type":"uint160"}],"name":"quoteExactInputSingle","outputs":[{"internalType":"uint256","name":"amountOut","type":"uint256"}],"stateMutability":"nonpayable","type":"function"}
]''')

@dataclass
class DefiTradeResult:
    tx_hash: str
    token_in: str
    token_out: str
    amount_in_wei: int
    amount_out_wei: int
    gas_used: int
    gas_cost_eth: float
    block_number: int
    timestamp: datetime
    slippage_actual: float
    success: bool
    error: Optional[str] = None

class UniswapV3Executor:
    def __init__(self, web3: Web3, wallet_address: str, private_key: str, slippage_tolerance: float = 0.005, deadline_seconds: int = 180):
        self.web3 = web3
        self.wallet_address = Web3.to_checksum_address(wallet_address)
        self.private_key = private_key
        self.slippage_tolerance = slippage_tolerance
        self.deadline_seconds = deadline_seconds
        
        self.router_contract = self.web3.eth.contract(address=UNISWAP_V3_ROUTER, abi=UNISWAP_V3_ROUTER_ABI)
        self.quoter_contract = self.web3.eth.contract(address=UNISWAP_V3_QUOTER, abi=UNISWAP_V3_QUOTER_ABI)

    async def get_quote(self, token_in: str, token_out: str, amount_in_wei: int) -> int:
        token_in = Web3.to_checksum_address(token_in)
        token_out = Web3.to_checksum_address(token_out)
        amount_out = self.quoter_contract.functions.quoteExactInputSingle(
            token_in, token_out, POOL_FEE_TIER, amount_in_wei, 0
        ).call()
        return amount_out

    async def approve_token(self, token_address: str, spender: str, amount_wei: int) -> Optional[str]:
        token_address = Web3.to_checksum_address(token_address)
        spender = Web3.to_checksum_address(spender)
        token_contract = self.web3.eth.contract(address=token_address, abi=ERC20_ABI)
        
        allowance = token_contract.functions.allowance(self.wallet_address, spender).call()
        if allowance >= amount_wei:
            return None
            
        tx = token_contract.functions.approve(spender, amount_wei).build_transaction({
            'from': self.wallet_address,
            'nonce': self.web3.eth.get_transaction_count(self.wallet_address),
        })
        
        signed_tx = self.web3.eth.account.sign_transaction(tx, private_key=self.private_key)
        tx_hash = self.web3.eth.send_raw_transaction(signed_tx.rawTransaction) # type: ignore
        self.web3.eth.wait_for_transaction_receipt(tx_hash, timeout=60)
        return tx_hash.hex()

    async def _get_token_decimals(self, token_address: str) -> int:
        if token_address == USDC_ADDRESS: return 6
        if token_address == WBTC_ADDRESS: return 8
        return 18

    async def swap(self, token_in: str, token_out: str, amount_in_usd: float, direction: str) -> DefiTradeResult:
        try:
            token_in = Web3.to_checksum_address(token_in)
            token_out = Web3.to_checksum_address(token_out)
            
            amount_in_wei = 0
            if token_in == USDC_ADDRESS:
                amount_in_wei = int(amount_in_usd * 1e6)
            else:
                # Need to convert USD to token amount
                # For WETH/WBTC, let's fetch current price from Binance
                symbol_map = {WETH_ADDRESS: "ETHUSDT", WBTC_ADDRESS: "BTCUSDT"}
                symbol = symbol_map.get(token_in)
                if not symbol:
                    raise ValueError(f"Unknown token in for USD conversion: {token_in}")
                
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"https://api.binance.us/api/v3/ticker/price?symbol={symbol}") as resp:
                        data = await resp.json()
                        price = float(data["price"])
                        
                amount_in_token = amount_in_usd / price
                decimals = await self._get_token_decimals(token_in)
                amount_in_wei = int(amount_in_token * (10 ** decimals))
            
            quote_out_wei = await self.get_quote(token_in, token_out, amount_in_wei)
            amount_out_min = int(quote_out_wei * (1 - self.slippage_tolerance))
            
            await self.approve_token(token_in, UNISWAP_V3_ROUTER, amount_in_wei)
            
            deadline = self.web3.eth.get_block('latest')['timestamp'] + self.deadline_seconds
            
            params = {
                'tokenIn': token_in,
                'tokenOut': token_out,
                'fee': POOL_FEE_TIER,
                'recipient': self.wallet_address,
                'deadline': deadline,
                'amountIn': amount_in_wei,
                'amountOutMinimum': amount_out_min,
                'sqrtPriceLimitX96': 0
            }
            
            tx = self.router_contract.functions.exactInputSingle(params).build_transaction({
                'from': self.wallet_address,
                'nonce': self.web3.eth.get_transaction_count(self.wallet_address),
                # value: 0
            })
            
            try:
                gas_estimate = self.web3.eth.estimate_gas(tx)
                tx['gas'] = int(gas_estimate * 1.2)
            except Exception as e:
                tx['gas'] = 250000
                
            signed_tx = self.web3.eth.account.sign_transaction(tx, private_key=self.private_key)
            tx_hash = self.web3.eth.send_raw_transaction(signed_tx.rawTransaction) # type: ignore
            
            receipt = self.web3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
            
            # Very simplified actual_out extraction (in reality you'd parse Event logs)
            # We'll just assume quote for return or min if failed
            actual_out_wei = quote_out_wei # Mocked
            gas_used = receipt['gasUsed']
            gas_cost_eth = float(self.web3.from_wei(gas_used * tx.get('gasPrice', self.web3.eth.gas_price), 'ether'))
            
            return DefiTradeResult(
                tx_hash=tx_hash.hex(),
                token_in=token_in,
                token_out=token_out,
                amount_in_wei=amount_in_wei,
                amount_out_wei=actual_out_wei,
                gas_used=gas_used,
                gas_cost_eth=gas_cost_eth,
                block_number=receipt['blockNumber'],
                timestamp=datetime.now(),
                slippage_actual=0.0,
                success=receipt['status'] == 1,
                error=None if receipt['status'] == 1 else "Transaction reverted"
            )
        except Exception as e:
            logger.error("swap_failed", error=str(e))
            return DefiTradeResult(
                tx_hash="", token_in=token_in, token_out=token_out,
                amount_in_wei=0, amount_out_wei=0, gas_used=0, gas_cost_eth=0.0,
                block_number=0, timestamp=datetime.now(), slippage_actual=0.0,
                success=False, error=str(e)
            )

class DefiPortfolioTracker:
    def __init__(self, web3: Web3, wallet_address: str, redis_client=None):
        self.web3 = web3
        self.wallet_address = Web3.to_checksum_address(wallet_address)
        self.redis_client = redis_client
        
        self.usdc = self.web3.eth.contract(address=USDC_ADDRESS, abi=ERC20_ABI)
        self.weth = self.web3.eth.contract(address=WETH_ADDRESS, abi=ERC20_ABI)
        self.wbtc = self.web3.eth.contract(address=WBTC_ADDRESS, abi=ERC20_ABI)

    async def get_balances(self) -> Dict[str, float]:
        usdc_bal = self.usdc.functions.balanceOf(self.wallet_address).call() / 1e6
        weth_bal = self.weth.functions.balanceOf(self.wallet_address).call() / 1e18
        wbtc_bal = self.wbtc.functions.balanceOf(self.wallet_address).call() / 1e8
        return {"USDC": usdc_bal, "WETH": weth_bal, "WBTC": wbtc_bal}

    async def get_portfolio_value_usd(self) -> float:
        balances = await self.get_balances()
        usd_value = balances["USDC"]
        
        async with aiohttp.ClientSession() as session:
            if balances["WETH"] > 0:
                async with session.get("https://api.binance.us/api/v3/ticker/price?symbol=ETHUSDT") as resp:
                    data = await resp.json()
                    usd_value += balances["WETH"] * float(data["price"])
            if balances["WBTC"] > 0:
                async with session.get("https://api.binance.us/api/v3/ticker/price?symbol=BTCUSDT") as resp:
                    data = await resp.json()
                    usd_value += balances["WBTC"] * float(data["price"])
        return usd_value

    async def get_open_position(self, asset: str) -> Optional[Dict[str, Any]]:
        balances = await self.get_balances()
        if asset == "ETHUSDT" and balances["WETH"] > 0.001:
            async with aiohttp.ClientSession() as session:
                async with session.get("https://api.binance.us/api/v3/ticker/price?symbol=ETHUSDT") as resp:
                    data = await resp.json()
                    price = float(data["price"])
            size_usd = balances["WETH"] * price
            
            entry_price = price
            if self.redis_client:
                cached = self.redis_client.get(f"entry_price:{asset}")
                if cached:
                    entry_price = float(cached)
            
            return {
                "asset": asset,
                "direction": "long",
                "size_usd": size_usd,
                "token_balance": balances["WETH"],
                "avg_entry_price": entry_price
            }
        elif asset == "BTCUSDT" and balances["WBTC"] > 0.0001:
            async with aiohttp.ClientSession() as session:
                async with session.get("https://api.binance.us/api/v3/ticker/price?symbol=BTCUSDT") as resp:
                    data = await resp.json()
                    price = float(data["price"])
            size_usd = balances["WBTC"] * price
            
            entry_price = price
            if self.redis_client:
                cached = self.redis_client.get(f"entry_price:{asset}")
                if cached:
                    entry_price = float(cached)
                    
            return {
                "asset": asset,
                "direction": "long",
                "size_usd": size_usd,
                "token_balance": balances["WBTC"],
                "avg_entry_price": entry_price
            }
        return None

SYMBOL_TO_TOKENS = {
  "BTCUSDT": {"base": WBTC_ADDRESS, "quote": USDC_ADDRESS, "base_decimals": 8},
  "ETHUSDT": {"base": WETH_ADDRESS, "quote": USDC_ADDRESS, "base_decimals": 18},
}

class DefiExecutionEngine:
    def __init__(self, uniswap: UniswapV3Executor, portfolio: DefiPortfolioTracker, kite_chain, db_session_factory, 
                 market_data_service=None, paper_mode: bool = True, risk_manager=None):
        self.uniswap = uniswap
        self.portfolio = portfolio
        self.kite_chain = kite_chain
        self.db_session_factory = db_session_factory
        self.paper_mode = paper_mode
        self.market_data_service = market_data_service
        self.redis_client = getattr(portfolio, "redis_client", None)
        self.risk_manager = risk_manager

    async def _get_live_state(self) -> Dict[str, Any]:
        if not self.redis_client:
            return {"unrealized_pnl": 0.0, "total_value_locked": 0.0, "positions": []}
        raw = await self.redis_client.get("portfolio:live_state")
        if not raw:
            return {"unrealized_pnl": 0.0, "total_value_locked": 0.0, "positions": []}
        try:
            return json.loads(raw)
        except Exception:
            return {"unrealized_pnl": 0.0, "total_value_locked": 0.0, "positions": []}

    async def _set_live_state(self, state: Dict[str, Any]) -> None:
        if self.redis_client:
            await self.redis_client.set("portfolio:live_state", json.dumps(state))

    async def execute(self, decision: Any, portfolio_state: Dict[str, Any]) -> Any:
        try:
            if getattr(decision, "direction", "hold") == "hold":
                return None
                
            symbol = decision.symbol
            direction = decision.direction
            size_pct = getattr(decision, "size_pct", 1.0)
            
            size_usd = size_pct * portfolio_state.get("available_usdc", 0.0)
            if size_usd <= 0 and direction == "long":
                return None

            tokens = SYMBOL_TO_TOKENS.get(symbol)
            if not tokens:
                return None
                
            token_in = USDC_ADDRESS if direction == "long" else tokens["base"]
            token_out = tokens["base"] if direction == "long" else USDC_ADDRESS

            current_pos = await self.portfolio.get_open_position(symbol)
            if current_pos and current_pos["direction"] == "long" and direction == "long":
                logger.info("skip_trade", reason="already_long")
                return None

            async with aiohttp.ClientSession() as session:
                async with session.get(f"https://api.binance.us/api/v3/ticker/price?symbol={symbol}") as resp:
                    data = await resp.json()
                    current_price = float(data["price"])

            if self.paper_mode:
                logger.info("PAPER TRADE", symbol=symbol, direction=direction, size_usd=size_usd)
                if self.redis_client and direction == "long":
                     await self.redis_client.set(f"entry_price:{symbol}", str(current_price))

                # Surface paper positions in dashboard (visible only in paper mode API response)
                live_state = await self._get_live_state()
                positions = [p for p in live_state.get("positions", []) if p.get("symbol") != symbol]
                asset_size = (size_usd / current_price) if current_price > 0 else 0.0
                positions.append({
                    "symbol": symbol,
                    "direction": direction,
                    "size_usd": size_usd,
                    "asset_size": asset_size,
                    "entry_price": current_price,
                    "current_price": current_price,
                    "unrealized": 0.0,
                })
                live_state["positions"] = positions
                live_state["total_value_locked"] = float(sum(float(p.get("size_usd", 0.0)) for p in positions))
                live_state["unrealized_pnl"] = float(sum(float(p.get("unrealized", 0.0)) for p in positions))
                await self._set_live_state(live_state)

                # Create a mock Trade object (assume Trade class exists in your models)
                class MockTrade:
                    id = 1
                    symbol = decision.symbol
                    paper = True
                    tx_hash = "mock_tx"
                trade = MockTrade()
                
                # Mock async process
                if self.kite_chain:
                    asyncio.create_task(self.kite_chain.log_trade_decision(trade, decision))
                return trade

            result = await self.uniswap.swap(token_in, token_out, size_usd, direction)
            if not result.success:
                logger.error("uniswap_swap_failed", error=result.error)
                if result.error and "insufficient funds" in result.error.lower():
                    if self.risk_manager:
                        self.risk_manager.is_halted = True
                
                # Write to AgentEvents table
                if self.db_session_factory:
                    try:
                        async with self.db_session_factory() as session:
                            event = AgentEvent(
                                event_type="SWAP_ERROR",
                                details={"error": result.error, "token_in": token_in, "token_out": token_out, "size": size_usd}
                            )
                            session.add(event)
                            await session.commit()
                    except Exception as db_e:
                        logger.error("failed_to_log_swap_error", error=str(db_e))
                return None

            if self.redis_client and direction == "long":
                await self.redis_client.set(f"entry_price:{symbol}", str(current_price))

            class MockTrade:
                id = 1
                symbol = decision.symbol
                paper = False
                tx_hash = result.tx_hash
            trade = MockTrade()
            
            if self.kite_chain:
                asyncio.create_task(self.kite_chain.log_trade_decision(trade, decision))
            return trade
            
        except Exception as e:
            logger.error("execute_failed", error=str(e), exc_info=True)
            return None

    async def close_position(self, symbol: str, reason: str = "signal") -> Any:
        current_pos = await self.portfolio.get_open_position(symbol)
        if not current_pos:
            return None
            
        tokens = SYMBOL_TO_TOKENS.get(symbol)
        if not tokens:
            return None

        # Determine balance
        balances = await self.portfolio.get_balances()
        token_key = "WETH" if symbol == "ETHUSDT" else "WBTC"
        token_balance = balances.get(token_key, 0.0)
        
        async with aiohttp.ClientSession() as session:
            async with session.get(f"https://api.binance.us/api/v3/ticker/price?symbol={symbol}") as resp:
                data = await resp.json()
                current_price = float(data["price"])

        size_usd = token_balance * current_price

        if self.paper_mode:
            logger.info("PAPER CLOSE", symbol=symbol, reason=reason)
            live_state = await self._get_live_state()
            positions = [p for p in live_state.get("positions", []) if p.get("symbol") != symbol]
            live_state["positions"] = positions
            live_state["total_value_locked"] = float(sum(float(p.get("size_usd", 0.0)) for p in positions))
            live_state["unrealized_pnl"] = float(sum(float(p.get("unrealized", 0.0)) for p in positions))
            await self._set_live_state(live_state)
            return True

        result = await self.uniswap.swap(tokens["base"], USDC_ADDRESS, size_usd, "short")
        if not result.success:
             logger.error("uniswap_close_failed", error=result.error)
             if result.error and "insufficient funds" in result.error.lower():
                 if getattr(self, "risk_manager", None):
                     self.risk_manager.is_halted = True
             if self.db_session_factory:
                 try:
                     async with self.db_session_factory() as session:
                         event = AgentEvent(
                             event_type="SWAP_CLOSE_ERROR",
                             details={"error": result.error, "asset": symbol}
                         )
                         session.add(event)
                         await session.commit()
                 except Exception as db_e:
                     pass
             return None
             
        # Mock Trade closed
        class MockClosedTrade:
            symbol = symbol
        trade = MockClosedTrade()
        
        if self.kite_chain:
             asyncio.create_task(self.kite_chain.log_trade_decision(trade, {"action": "close", "reason": reason}))
        return trade
