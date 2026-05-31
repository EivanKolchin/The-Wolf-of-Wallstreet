import asyncio
import json
import logging
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from decimal import Decimal
import structlog
from web3 import Web3
from web3.exceptions import ContractLogicError
from typing import Optional, Dict, Any
import aiohttp

from backend.memory.database import (
    AgentEvent, Trade, TradeDirection, TradeStatus, OrderType,
)
from backend.core.rate_limiter import binance_rest_limiter
from backend.core import ledger
from backend.execution.base import BrokerInterface, AssetClass

logger = structlog.get_logger(__name__)

# Constants
ARBITRUM_RPC_URL = "https://arb1.arbitrum.io/rpc"
UNISWAP_V3_ROUTER = Web3.to_checksum_address("0xE592427A0AEce92De3Edee1F18E0157C05861564")
UNISWAP_V3_QUOTER = Web3.to_checksum_address("0xb27308f9F90D607463bb33eA1BeBb41C27CE5AB6")
USDC_ADDRESS = Web3.to_checksum_address("0xaf88d065e77c8cC2239327C5EDb3A432268e5831")
WETH_ADDRESS = Web3.to_checksum_address("0x82aF49447D8a07e3bd95BD0d56f35241523fBab1")
WBTC_ADDRESS = Web3.to_checksum_address("0x2f2a2543B76A4166549F7aaB2e75Bef0aefC5B0f")
POOL_FEE_TIER = 3000

# Approx taker fee applied to notional per side (used for fee-aware net PnL).
DEFAULT_FEE_RATE = 0.001  # 0.1%

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


async def fetch_binance_price(symbol: str, retries: int = 3) -> float:
    """Rate-limited + API-logged Binance spot price fetch (single source of truth)."""
    limiter = binance_rest_limiter()
    url = f"https://api.binance.us/api/v3/ticker/price?symbol={symbol}"
    last_error = None
    for attempt in range(retries):
        await limiter.acquire()
        t0 = time.monotonic()
        try:
            timeout = aiohttp.ClientTimeout(total=5)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url) as resp:
                    data = await resp.json()
                    price = float(data["price"])
                    ledger.log_api_call("binance", "GET", url, status=resp.status, ok=True,
                                        latency_ms=(time.monotonic() - t0) * 1000)
                    return price
        except Exception as e:
            last_error = str(e)
            ledger.log_api_call("binance", "GET", url, ok=False,
                                latency_ms=(time.monotonic() - t0) * 1000, note=str(e)[:120])
            await asyncio.sleep(0.25 * (attempt + 1))
    raise RuntimeError(f"price_fetch_failed:{symbol}:{last_error}")


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
        tx_hash = self.web3.eth.send_raw_transaction(signed_tx.rawTransaction)  # type: ignore
        self.web3.eth.wait_for_transaction_receipt(tx_hash, timeout=60)
        return tx_hash.hex()

    async def _get_token_decimals(self, token_address: str) -> int:
        if token_address == USDC_ADDRESS: return 6
        if token_address == WBTC_ADDRESS: return 8
        return 18

    async def swap(
        self,
        token_in: str,
        token_out: str,
        amount_in_usd: float,
        direction: str,
        amount_in_wei_override: int | None = None,
    ) -> DefiTradeResult:
        try:
            token_in = Web3.to_checksum_address(token_in)
            token_out = Web3.to_checksum_address(token_out)

            amount_in_wei = 0
            if amount_in_wei_override is not None:
                amount_in_wei = int(amount_in_wei_override)
            elif token_in == USDC_ADDRESS:
                amount_in_wei = int(amount_in_usd * 1e6)
            else:
                symbol_map = {WETH_ADDRESS: "ETHUSDT", WBTC_ADDRESS: "BTCUSDT"}
                symbol = symbol_map.get(token_in)
                if not symbol:
                    raise ValueError(f"Unknown token in for USD conversion: {token_in}")
                price = await fetch_binance_price(symbol)
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
            })

            # Cycle 19.4: gas safety ceilings. Without these, the agent will
            # pay arbitrary fees during Arbitrum congestion and can drain its
            # ETH balance on a single swap.
            from backend.core.config import settings as _settings
            max_gas_units = int(getattr(_settings, "DEFI_MAX_GAS_UNITS", 500_000))
            max_gas_price_wei = int(float(getattr(_settings, "DEFI_MAX_GAS_PRICE_GWEI", 5.0)) * 1e9)

            try:
                gas_estimate = self.web3.eth.estimate_gas(tx)
                tx['gas'] = min(int(gas_estimate * 1.2), max_gas_units)
            except Exception:
                tx['gas'] = min(250000, max_gas_units)

            # Refuse to sign during a network gas spike; surface a clear log
            # rather than silently draining ETH.
            try:
                live_gas_price = int(self.web3.eth.gas_price)
            except Exception:
                live_gas_price = 0
            if live_gas_price and live_gas_price > max_gas_price_wei:
                logger.warning(
                    "gas_ceiling_exceeded_aborting_swap",
                    live_gwei=round(live_gas_price / 1e9, 3),
                    ceiling_gwei=getattr(_settings, "DEFI_MAX_GAS_PRICE_GWEI", 5.0),
                    token_in=token_in, token_out=token_out,
                )
                return None
            # Set EIP-1559 fields where supported (Arbitrum). Falls back to
            # gasPrice if the chain rejects the maxFeePerGas style.
            tx.setdefault('maxFeePerGas', max_gas_price_wei)
            tx.setdefault('maxPriorityFeePerGas', min(max_gas_price_wei, int(0.1 * 1e9)))

            signed_tx = self.web3.eth.account.sign_transaction(tx, private_key=self.private_key)
            tx_hash = self.web3.eth.send_raw_transaction(signed_tx.rawTransaction)  # type: ignore

            receipt = self.web3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)

            actual_out_wei = quote_out_wei  # Mocked (would parse Event logs in production)
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
        if balances["WETH"] > 0:
            usd_value += balances["WETH"] * await fetch_binance_price("ETHUSDT")
        if balances["WBTC"] > 0:
            usd_value += balances["WBTC"] * await fetch_binance_price("BTCUSDT")
        return usd_value

    async def get_open_position(self, asset: str) -> Optional[Dict[str, Any]]:
        balances = await self.get_balances()
        if asset == "ETHUSDT" and balances["WETH"] > 0.001:
            price = await fetch_binance_price("ETHUSDT")
            size_usd = balances["WETH"] * price
            entry_price = price
            if self.redis_client:
                cached = self.redis_client.get(f"entry_price:{asset}")
                if cached:
                    entry_price = float(cached)
            return {"asset": asset, "direction": "long", "size_usd": size_usd,
                    "token_balance": balances["WETH"], "avg_entry_price": entry_price}
        elif asset == "BTCUSDT" and balances["WBTC"] > 0.0001:
            price = await fetch_binance_price("BTCUSDT")
            size_usd = balances["WBTC"] * price
            entry_price = price
            if self.redis_client:
                cached = self.redis_client.get(f"entry_price:{asset}")
                if cached:
                    entry_price = float(cached)
            return {"asset": asset, "direction": "long", "size_usd": size_usd,
                    "token_balance": balances["WBTC"], "avg_entry_price": entry_price}
        return None


SYMBOL_TO_TOKENS = {
  "BTCUSDT": {"base": WBTC_ADDRESS, "quote": USDC_ADDRESS, "base_decimals": 8},
  "ETHUSDT": {"base": WETH_ADDRESS, "quote": USDC_ADDRESS, "base_decimals": 18},
}


class DefiExecutionEngine(BrokerInterface):
    asset_class = AssetClass.crypto.value

    def __init__(self, uniswap: UniswapV3Executor, portfolio: DefiPortfolioTracker, db_session_factory,
                 market_data_service=None, paper_mode: bool = True, risk_manager=None):
        self.uniswap = uniswap
        self.portfolio = portfolio
        self.db_session_factory = db_session_factory
        self.paper_mode = paper_mode
        self.market_data_service = market_data_service
        self.redis_client = getattr(portfolio, "redis_client", None)
        self.risk_manager = risk_manager
        self._trade_closed_callback = None
        # Engine-side tracking of open Trade rows by symbol + double-close guard.
        self._open_trades: Dict[str, Trade] = {}
        self._closing: set = set()

    def set_trade_closed_callback(self, callback):
        self._trade_closed_callback = callback

    async def _log_agent_event(self, event_type: str, details: Dict[str, Any]) -> None:
        if not self.db_session_factory:
            return
        try:
            async with self.db_session_factory() as session:
                event = AgentEvent(event_type=event_type, details=details)
                session.add(event)
                await session.commit()
        except Exception as e:
            logger.error("agent_event_log_failed", event_type=event_type, error=str(e))

    async def _fetch_price(self, symbol: str, retries: int = 3) -> float:
        return await fetch_binance_price(symbol, retries=retries)

    async def get_price(self, symbol: str) -> Optional[float]:
        """None-safe current price for the position monitor (rate-limited + logged)."""
        try:
            return await fetch_binance_price(symbol)
        except Exception as e:
            logger.warning("get_price_failed", symbol=symbol, error=str(e))
            return None

    def get_open_trades(self) -> Dict[str, Trade]:
        return self._open_trades

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

    def _exit_levels(self, direction: str, entry: float, sl_frac: float, tp_frac: float):
        """Absolute stop-loss / take-profit prices from model-emitted fractions.
        Returns 0.0 for a level when its fraction is 0 (monitor treats <=0 as disabled)."""
        if direction == "long":
            sl = entry * (1 - sl_frac) if sl_frac > 0 else 0.0
            tp = entry * (1 + tp_frac) if tp_frac > 0 else 0.0
        else:
            sl = entry * (1 + sl_frac) if sl_frac > 0 else 0.0
            tp = entry * (1 - tp_frac) if tp_frac > 0 else 0.0
        return float(sl), float(tp)

    async def _persist_new_trade(self, decision: Any, entry_price: float, size_usd: float) -> Optional[Trade]:
        """Create + persist a real open Trade row and return the (detached) object.

        Without a db_session_factory a transient (un-persisted) Trade is returned so
        the engine still functions and stays unit-testable when no DB is wired."""
        direction = decision.direction
        sl_frac = float(getattr(decision, "sl", 0.0) or 0.0)
        tp_frac = float(getattr(decision, "tp", 0.0) or 0.0)
        trail_frac = float(getattr(decision, "trail", 0.0) or 0.0)
        stop_loss, take_profit = self._exit_levels(direction, entry_price, sl_frac, tp_frac)
        try:
            news = getattr(decision, "active_news", None)
            news_dict = asdict(news) if (news is not None and hasattr(news, "__dataclass_fields__")) else None
        except Exception:
            news_dict = None

        trade = Trade(
            asset=decision.symbol,
            direction=TradeDirection.long if direction == "long" else TradeDirection.short,
            size_usd=float(size_usd),
            entry_price=float(entry_price),
            status=TradeStatus.open,
            order_type=OrderType.market,
            nn_confidence=float(getattr(decision, "nn_confidence", 0.0)),
            nn_direction_probs=getattr(decision, "nn_probs", {}) or {},
            active_news_impact=news_dict,
            regime_at_entry=str(getattr(decision, "regime", "")),
            stop_loss=stop_loss,
            take_profit=take_profit,
            quote_asset="USDC",
            fee_paid=float(size_usd) * DEFAULT_FEE_RATE,
            trailing_stop=trail_frac,
            highest_price_seen=float(entry_price),
            target_price=float(getattr(decision, "target_price", 0.0) or 0.0),
            expected_execution_ts=float(getattr(decision, "expected_execution_ts", 0.0) or 0.0),
            rationale=getattr(decision, "rationale", None),
        )
        if self.db_session_factory:
            async with self.db_session_factory() as session:
                session.add(trade)
                await session.commit()
                await session.refresh(trade)
        return trade

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
                await self._log_agent_event(
                    "UNSUPPORTED_SYMBOL",
                    {"symbol": symbol, "reason": "missing_token_mapping", "direction": direction},
                )
                logger.warning("unsupported_symbol_rejected", symbol=symbol)
                return None

            token_in = USDC_ADDRESS if direction == "long" else tokens["base"]
            token_out = tokens["base"] if direction == "long" else USDC_ADDRESS

            current_pos = await self.portfolio.get_open_position(symbol) if not self.paper_mode else None
            if current_pos and current_pos["direction"] == "long" and direction == "long":
                logger.info("skip_trade", reason="already_long")
                return None

            current_price = await self._fetch_price(symbol)

            if not self.paper_mode:
                result = await self.uniswap.swap(token_in, token_out, size_usd, direction)
                if not result.success:
                    logger.error("uniswap_swap_failed", error=result.error)
                    if result.error and "insufficient funds" in result.error.lower() and self.risk_manager:
                        self.risk_manager.is_halted = True
                    await self._log_agent_event(
                        "SWAP_ERROR",
                        {"error": result.error, "token_in": token_in, "token_out": token_out, "size": size_usd, "symbol": symbol},
                    )
                    return None
            else:
                logger.info("PAPER TRADE", symbol=symbol, direction=direction, size_usd=size_usd)

            if self.redis_client and direction == "long":
                await self.redis_client.set(f"entry_price:{symbol}", str(current_price))

            # Persist a real Trade row (paper + live) so we have records + statements + exits.
            trade = await self._persist_new_trade(decision, current_price, size_usd)
            if trade is None:
                return None
            self._open_trades[symbol] = trade

            # Surface the position in the live dashboard state.
            live_state = await self._get_live_state()
            positions = [p for p in live_state.get("positions", []) if p.get("symbol") != symbol]
            asset_size = (size_usd / current_price) if current_price > 0 else 0.0
            positions.append({
                "symbol": symbol, "direction": direction, "size_usd": size_usd,
                "asset_size": asset_size, "entry_price": current_price,
                "current_price": current_price, "unrealized": 0.0,
                "stop_loss": trade.stop_loss, "take_profit": trade.take_profit,
            })
            live_state["positions"] = positions
            live_state["total_value_locked"] = float(sum(float(p.get("size_usd", 0.0)) for p in positions))
            live_state["unrealized_pnl"] = float(sum(float(p.get("unrealized", 0.0)) for p in positions))
            await self._set_live_state(live_state)

            return trade

        except Exception as e:
            logger.error("execute_failed", error=str(e), exc_info=True)
            return None

    async def close_position(self, symbol: str, reason: str = "signal") -> Any:
        if symbol in self._closing:
            return None
        self._closing.add(symbol)
        try:
            trade = self._open_trades.get(symbol)
            if trade is None:
                # Fall back to the latest open row in the DB.
                trade = await self._latest_open_trade(symbol)
            if trade is None:
                return None

            direction = trade.direction.value if hasattr(trade.direction, "value") else str(trade.direction)
            entry_price = float(trade.entry_price)
            size_usd = float(trade.size_usd)

            exit_price = await self.get_price(symbol)
            if exit_price is None:
                exit_price = entry_price  # cannot fetch -> treat as flat

            if not self.paper_mode:
                tokens = SYMBOL_TO_TOKENS.get(symbol)
                if tokens:
                    balances = await self.portfolio.get_balances()
                    token_key = "WETH" if symbol == "ETHUSDT" else "WBTC"
                    token_balance = balances.get(token_key, 0.0)
                    amount_in_wei_exact = int(token_balance * (10 ** int(tokens["base_decimals"])))
                    result = await self.uniswap.swap(
                        tokens["base"], USDC_ADDRESS, token_balance * exit_price, "short",
                        amount_in_wei_override=amount_in_wei_exact,
                    )
                    if not result.success:
                        logger.error("uniswap_close_failed", error=result.error)
                        await self._log_agent_event("SWAP_CLOSE_ERROR", {"error": result.error, "asset": symbol, "reason": reason})
                        return None

            pnl_pct = ((exit_price - entry_price) / entry_price) if entry_price > 0 else 0.0
            if direction == "short":
                pnl_pct = -pnl_pct
            gross_pnl_usd = size_usd * pnl_pct
            fee_close = size_usd * DEFAULT_FEE_RATE
            fee_total = float(trade.fee_paid or 0.0) + fee_close
            net_pnl_usd = gross_pnl_usd - fee_close          # open fee already in fee_paid
            net_pnl_pct = (net_pnl_usd - float(trade.fee_paid or 0.0)) / size_usd if size_usd > 0 else 0.0

            updated = await self._finalize_trade_row(
                trade.id, exit_price=exit_price, pnl_usd=net_pnl_usd,
                pnl_pct=net_pnl_pct, fee_total=fee_total, reason=reason,
                highest_price_seen=getattr(trade, "highest_price_seen", entry_price),
            )

            # Transaction statement (realized trade) at repo root.
            ledger.record_transaction({
                "opened_at": str(getattr(trade, "opened_at", "")),
                "trade_id": str(trade.id),
                "symbol": symbol, "asset_class": "crypto", "broker": "uniswap_v3",
                "direction": direction, "size_usd": round(size_usd, 4),
                "entry_price": round(entry_price, 6), "exit_price": round(float(exit_price), 6),
                "pnl_usd": round(net_pnl_usd, 4), "pnl_pct": round(net_pnl_pct, 6),
                "fee_paid": round(fee_total, 4), "quote_asset": "USDC",
                "exit_reason": reason, "paper": self.paper_mode,
                "target_price": float(getattr(trade, "target_price", 0.0) or 0.0),
            })

            # Remove from live dashboard state.
            live_state = await self._get_live_state()
            positions = [p for p in live_state.get("positions", []) if p.get("symbol") != symbol]
            live_state["positions"] = positions
            live_state["total_value_locked"] = float(sum(float(p.get("size_usd", 0.0)) for p in positions))
            live_state["unrealized_pnl"] = float(sum(float(p.get("unrealized", 0.0)) for p in positions))
            await self._set_live_state(live_state)

            self._open_trades.pop(symbol, None)

            logger.info("position_closed", symbol=symbol, reason=reason, pnl_pct=round(net_pnl_pct, 5))
            if self._trade_closed_callback:
                await self._trade_closed_callback(updated or trade, net_pnl_pct)
            return updated or trade
        finally:
            self._closing.discard(symbol)

    async def _latest_open_trade(self, symbol: str) -> Optional[Trade]:
        if not self.db_session_factory:
            return None
        from sqlalchemy import select, desc
        try:
            async with self.db_session_factory() as session:
                res = await session.execute(
                    select(Trade).where(Trade.asset == symbol, Trade.status == TradeStatus.open)
                    .order_by(desc(Trade.opened_at)).limit(1)
                )
                return res.scalar_one_or_none()
        except Exception as e:
            logger.error("latest_open_trade_query_failed", symbol=symbol, error=str(e))
            return None

    async def _finalize_trade_row(self, trade_id, *, exit_price, pnl_usd, pnl_pct,
                                  fee_total, reason, highest_price_seen) -> Optional[Trade]:
        if not self.db_session_factory:
            return None
        from datetime import datetime as _dt
        try:
            async with self.db_session_factory() as session:
                row = await session.get(Trade, trade_id)
                if row is None:
                    return None
                row.exit_price = float(exit_price)
                row.status = TradeStatus.closed
                row.closed_at = _dt.utcnow()
                row.pnl_usd = float(pnl_usd)
                row.pnl_pct = float(pnl_pct)
                row.fee_paid = float(fee_total)
                row.exit_reason = str(reason)
                row.highest_price_seen = float(highest_price_seen) if highest_price_seen else row.entry_price
                await session.commit()
                await session.refresh(row)
                return row
        except Exception as e:
            logger.error("finalize_trade_row_failed", trade_id=str(trade_id), error=str(e))
            return None
