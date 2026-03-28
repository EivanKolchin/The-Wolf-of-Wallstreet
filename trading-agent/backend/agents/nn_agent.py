import asyncio
import ctypes
import multiprocessing
from collections import deque
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass

import numpy as np
import structlog

from backend.data.market_feed import BinanceMarketFeed
from backend.signals.features import FeatureVectorBuilder
from backend.signals.regime import RegimeDetector
from backend.agents.nn_model import PersistentTradingModel, TradeExperience
from backend.memory.redis_client import PriorityNewsQueue, NewsImpact, HeartbeatClient
from backend.memory.database import Trade, TradeStatus

logger = structlog.get_logger(__name__)

@dataclass
class TradeDecision:
    symbol: str
    direction: str        # "long" / "short" / "hold"
    size_pct: float       # fraction of available capital [0, 0.20]
    nn_confidence: float  # max(probs.values())
    nn_probs: dict
    regime: str
    active_news: NewsImpact | None
    timestamp: datetime

class NNTradingAgent:
    def __init__(
        self,
        market_feed: BinanceMarketFeed,
        feature_builder: FeatureVectorBuilder,
        regime_detector: RegimeDetector,
        model: PersistentTradingModel,
        risk_manager,
        execution_engine,
        news_queue: PriorityNewsQueue,
        severe_flag: multiprocessing.Value,
        symbols: list[str],
        cycle_interval_seconds: float = 5.0
    ):
        self.market_feed = market_feed
        self.feature_builder = feature_builder
        self.regime_detector = regime_detector
        self.model = model
        self.risk_manager = risk_manager
        self.execution_engine = execution_engine
        self.news_queue = news_queue
        self.severe_flag = severe_flag
        self.cycle_interval_seconds = cycle_interval_seconds
        self.symbols = symbols
        
        self.heartbeat_client = HeartbeatClient(news_queue.redis)

        self.feature_sequences: dict[str, deque] = {
            sym: deque(maxlen=self.model.SEQUENCE_LENGTH) for sym in self.symbols
        }
        self.open_trades: dict[str, Trade] = {}
        
        self.current_news_impact: NewsImpact | None = None
        self.news_impact_expires_at: datetime | None = None

    async def run(self) -> None:
        logger.info("nn_trading_agent_started", symbols=self.symbols)
        
        while True:
            try:
                # 1. Check severe flag
                if self.severe_flag.value:
                    await self._emergency_protocol()
                    await asyncio.sleep(60)
                    continue

                # 2. Drain news queue
                while True:
                    impact = await self.news_queue.get_nowait()
                    if not impact:
                        break
                    
                    if impact.severity == "SEVERE":
                        self.severe_flag.value = True
                        logger.error("severe_news_received_triggering_emergency", impact=impact.to_json() if hasattr(impact, 'to_json') else str(impact))
                        await self._emergency_protocol()
                        break
                    elif impact.severity == "SIGNIFICANT":
                        self.current_news_impact = impact
                        self.news_impact_expires_at = datetime.utcnow() + timedelta(minutes=impact.t_max_minutes)

                if self.severe_flag.value:
                    continue

                # 3. Expire news impact
                if self.current_news_impact and self.news_impact_expires_at:
                    if datetime.utcnow() > self.news_impact_expires_at:
                        logger.info("news_impact_expired")
                        self.current_news_impact = None
                        self.news_impact_expires_at = None

                # 4. Process each symbol
                for symbol in self.symbols:
                    try:
                        df = await self.market_feed.get_dataframe(symbol)
                    except AttributeError as e:
                        logger.error("market_feed_missing_attribute", error=str(e))
                        continue
                        
                    if df is None or df.empty:
                        continue

                    orderbook = await self.market_feed.get_orderbook(symbol)
                    bids = orderbook.get("bids", []) if orderbook else []
                    asks = orderbook.get("asks", []) if orderbook else []
                    
                    trades = await self.market_feed.get_recent_trades(symbol, n=100)
                    
                    sr_levels = []
                    
                    regime_name, regime_conf = self.regime_detector.detect(df, self.current_news_impact)
                    
                    # Build feature vector
                    vector = await self.feature_builder.build(
                        symbol=symbol,
                        df=df,
                        bids=bids,
                        asks=asks,
                        trades=trades,
                        sr_levels=sr_levels,
                        regime=regime_name,
                        news_impact=self.current_news_impact
                    )
                    
                    self.feature_sequences[symbol].append(vector)
                    
                    if len(self.feature_sequences[symbol]) < self.model.SEQUENCE_LENGTH:
                        continue
                        
                    sequence = np.stack(self.feature_sequences[symbol])
                    
                    decision_str, size_pct, probs = self.model.infer(sequence)
                    nn_confidence = max(probs.values())
                    
                    decision = TradeDecision(
                        symbol=symbol,
                        direction=decision_str,
                        size_pct=size_pct,
                        nn_confidence=nn_confidence,
                        nn_probs=probs,
                        regime=regime_name,
                        active_news=self.current_news_impact,
                        timestamp=datetime.utcnow()
                    )
                    
                    portfolio_state = {"available_cash": 10000.0} 
                    
                    approved, reason = self.risk_manager.approve(decision, portfolio_state)
                    if approved:
                        trade = await self.execution_engine.execute(decision, portfolio_state["available_cash"])
                        if trade:
                            self.open_trades[symbol] = trade
                            logger.info("trade_executed", symbol=symbol, direction=decision_str, size=size_pct)
                    else:
                        if decision_str != "hold":
                            logger.debug("trade_rejected", symbol=symbol, reason=reason)

                await self.heartbeat_client.ping("nn_trading_agent")
                
            except Exception as e:
                logger.error("nn_agent_loop_error", error=str(e))
                
            await asyncio.sleep(self.cycle_interval_seconds)

    async def _emergency_protocol(self) -> None:
        logger.error("emergency_protocol_activated", open_trades=len(self.open_trades))
        
        for symbol, trade in list(self.open_trades.items()):
            logger.info("attempting_emergency_close", symbol=symbol, trade_id=str(trade.id))
            try:
                # Execution engine handles closing. Cancel trade object locally.
                trade.status = TradeStatus.cancelled
            except Exception as e:
                logger.error("emergency_close_failed", symbol=symbol, error=str(e))
                
        self.open_trades.clear()
        
    async def _on_trade_closed(self, trade: Trade, pnl_pct: float) -> None:
        logger.info("trade_closed", trade_id=str(trade.id), pnl_pct=pnl_pct)
        # Using zero array for dummy sequence to satisfy model learning signature constraints cleanly
        dummy_seq = np.zeros((self.model.SEQUENCE_LENGTH, 62), dtype=np.float32)
        
        direction_map = {"long": 0, "short": 1, "hold": 2}
        dir_taken = direction_map.get(trade.direction.value if trade.direction else "hold", 2)
        
        experience = TradeExperience(
            features_sequence=dummy_seq,
            direction_taken=dir_taken,
            actual_pnl_pct=pnl_pct
        )
        
        self.model.online_update(experience)
        
        if self.model.check_and_rollback(recent_pnl_pct=pnl_pct):
            logger.warning("rollback_triggered_by_recent_pnl")
        
        if trade.symbol in self.open_trades:
            del self.open_trades[trade.symbol]