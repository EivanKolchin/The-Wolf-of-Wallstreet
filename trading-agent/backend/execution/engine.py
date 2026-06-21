import asyncio
import json
import structlog
from datetime import datetime

import ccxt
from sqlalchemy.ext.asyncio import async_sessionmaker

from backend.agents.nn_agent import TradeDecision
from backend.memory.database import Trade, TradeDirection, TradeStatus, OrderType

logger = structlog.get_logger(__name__)

class ExecutionError(Exception):
    pass

class ExecutionEngine:
    def __init__(
        self,
        exchange: ccxt.binance,
        db_session_factory: async_sessionmaker,
        paper_mode: bool = True
    ):
        self.exchange = exchange
        self.db_session_factory = db_session_factory
        self.paper_mode = paper_mode
        self._trade_closed_callback = None

    def set_trade_closed_callback(self, callback):
        self._trade_closed_callback = callback

    async def _run_exchange_call(self, fn, *args, retries: int = 2):
        last_error = None
        for attempt in range(retries + 1):
            try:
                return await asyncio.to_thread(fn, *args)
            except Exception as e:
                last_error = e
                if attempt < retries:
                    await asyncio.sleep(0.2 * (attempt + 1))
        raise last_error

    async def get_current_price(self, symbol: str) -> float:
        try:
            ticker = await self._run_exchange_call(self.exchange.fetch_ticker, symbol)
            return float(ticker["last"])
        except Exception as e:
            logger.error("failed_to_get_current_price", symbol=symbol, error=str(e))
            return 0.0

    async def execute(self, decision: TradeDecision, available_cash: float) -> Trade | None:
        size_usd = decision.size_pct * available_cash
        
        try:
            qty, price = await self.normalise_order(decision.symbol, size_usd)
        except ValueError as e:
            logger.warning("order_normalisation_failed", symbol=decision.symbol, error=str(e))
            return None

        # Use model-provided SL/TP from the decision
        sl_frac = float(getattr(decision, "sl", 0.0) or 0.0)
        tp_frac = float(getattr(decision, "tp", 0.0) or 0.0)
        if decision.direction == "long":
            stop_loss = price * (1 - sl_frac) if sl_frac > 0 else 0.0
            take_profit = price * (1 + tp_frac) if tp_frac > 0 else 0.0
        else:
            stop_loss = price * (1 + sl_frac) if sl_frac > 0 else 0.0
            take_profit = price * (1 - tp_frac) if tp_frac > 0 else 0.0

        active_news_json = decision.active_news.to_json() if decision.active_news and hasattr(decision.active_news, 'to_json') else (decision.active_news if decision.active_news else None)
        if isinstance(active_news_json, str):
             try:
                 active_news_json = json.loads(active_news_json)
             except Exception:
                 pass

        trade = Trade(
            asset=decision.symbol,
            direction=TradeDirection.long if decision.direction == "long" else TradeDirection.short,
            size_usd=size_usd,
            entry_price=price,
            status=TradeStatus.open,
            order_type=OrderType.market if self.select_order_type(decision) == "market" else OrderType.limit,
            nn_confidence=decision.nn_confidence,
            nn_direction_probs=decision.nn_probs,
            active_news_impact=active_news_json,
            regime_at_entry=decision.regime,
            stop_loss=stop_loss,
            take_profit=take_profit,
            opened_at=datetime.utcnow()
        )

        if self.paper_mode:
            slippage_bps = max(5, int(getattr(decision, "size_pct", 0.0) * 100))  # 5-20bps based on size
            slippage = price * slippage_bps / 10000.0
            if decision.direction == "long":
                price = price + slippage
            else:
                price = price - slippage
            logger.info("PAPER_TRADE_EXECUTED", symbol=decision.symbol, qty=qty, price=price,
                        direction=decision.direction, slippage_bps=slippage_bps)
        else:
            try:
                order_type = self.select_order_type(decision)
                side = "buy" if decision.direction == "long" else "sell"
                
                logger.info("live_order_submission", symbol=decision.symbol, type=order_type, side=side, qty=qty, price=price)
                
                if order_type == "market":
                    order = await self._run_exchange_call(self.exchange.create_market_order, decision.symbol, side, qty)
                else:
                    order = await self._run_exchange_call(self.exchange.create_limit_order, decision.symbol, side, qty, price)
                
                if not order or "id" not in order:
                    raise ExecutionError("Missing order ID in exchange response")
                    
                trade.entry_price = float(order.get("price") or price)
            except Exception as e:
                logger.error("live_execution_failed", symbol=decision.symbol, error=str(e))
                return None

        async with self.db_session_factory() as session:
            session.add(trade)
            await session.commit()
            await session.refresh(trade)

        return trade

    async def normalise_order(self, symbol: str, size_usd: float) -> tuple[float, float]:
        market = await self._run_exchange_call(self.exchange.market, symbol)
        ticker = await self._run_exchange_call(self.exchange.fetch_ticker, symbol)
        price = ticker["last"]
        
        limits = market.get("limits", {})
        cost_limits = limits.get("cost", {})
        min_notional = cost_limits.get("min")
        if min_notional is None:
            min_notional = 10.0
        else:
            min_notional = float(min_notional)
            
        raw_qty = size_usd / price
        qty_str = await self._run_exchange_call(self.exchange.amount_to_precision, symbol, raw_qty)
        qty = float(qty_str)
        
        if qty * price < min_notional:
            raise ValueError(f"Below min notional: {qty * price} < {min_notional}")
            
        return qty, price

    def select_order_type(self, decision: TradeDecision) -> str:
        if decision.active_news and decision.active_news.confidence > 0.75:
            return "market"
        if decision.size_pct > 0.10:
            return "limit"
        return "limit"