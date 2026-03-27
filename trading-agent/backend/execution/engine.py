import asyncio
import json
import structlog
from datetime import datetime

import ccxt
from sqlalchemy.ext.asyncio import async_sessionmaker

from backend.agents.nn_agent import TradeDecision 
from backend.execution.kite_chain import KiteChainClient
from backend.memory.database import Trade, TradeDirection, TradeStatus, OrderType

logger = structlog.get_logger(__name__)

class ExecutionError(Exception):
    pass

class ExecutionEngine:
    def __init__(
        self,
        exchange: ccxt.binance,
        kite_chain: KiteChainClient,
        db_session_factory: async_sessionmaker,
        paper_mode: bool = True
    ):
        self.exchange = exchange
        self.kite_chain = kite_chain
        self.db_session_factory = db_session_factory
        self.paper_mode = paper_mode

    async def execute(self, decision: TradeDecision, available_cash: float) -> Trade | None:
        size_usd = decision.size_pct * available_cash
        
        try:
            qty, price = self.normalise_order(decision.symbol, size_usd)
        except ValueError as e:
            logger.warning("order_normalisation_failed", symbol=decision.symbol, error=str(e))
            return None

        # Determine stop loss and take profit for record keeping
        stop_loss_pct = 0.05
        take_profit_pct = 0.10
        if decision.direction == "long":
            stop_loss = price * (1 - stop_loss_pct)
            take_profit = price * (1 + take_profit_pct)
        else:
            stop_loss = price * (1 + stop_loss_pct)
            take_profit = price * (1 - take_profit_pct)

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
            logger.info("PAPER_TRADE_EXECUTED", symbol=decision.symbol, qty=qty, price=price, direction=decision.direction)
        else:
            try:
                order_type = self.select_order_type(decision)
                side = "buy" if decision.direction == "long" else "sell"
                
                logger.info("live_order_submission", symbol=decision.symbol, type=order_type, side=side, qty=qty, price=price)
                
                if order_type == "market":
                    order = self.exchange.create_market_order(decision.symbol, side, qty)
                else:
                    order = self.exchange.create_limit_order(decision.symbol, side, qty, price)
                
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

        asyncio.create_task(self.kite_chain.log_trade_decision(trade, decision))
        
        return trade

    def normalise_order(self, symbol: str, size_usd: float) -> tuple[float, float]:
        market = self.exchange.market(symbol)
        ticker = self.exchange.fetch_ticker(symbol)
        price = ticker["last"]
        
        limits = market.get("limits", {})
        cost_limits = limits.get("cost", {})
        min_notional = cost_limits.get("min")
        if min_notional is None:
            min_notional = 10.0
        else:
            min_notional = float(min_notional)
            
        raw_qty = size_usd / price
        qty_str = self.exchange.amount_to_precision(symbol, raw_qty)
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