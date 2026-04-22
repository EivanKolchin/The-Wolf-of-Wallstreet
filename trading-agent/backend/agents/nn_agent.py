import asyncio
import ctypes
import multiprocessing
import json
from collections import deque
from datetime import datetime, timedelta, timezone
import time
from dataclasses import dataclass

import numpy as np
import structlog

from backend.data.market_feed import BinanceMarketFeed
from backend.signals.features import FeatureVectorBuilder
from backend.signals.regime import RegimeDetector
from backend.agents.nn_model import PersistentTradingModel, TradeExperience
from backend.memory.redis_client import PriorityNewsQueue, NewsImpact, HeartbeatClient
from backend.memory.database import Trade, TradeStatus, get_session
from backend.execution.position_manager import PositionManager
from sqlalchemy import select, func
from backend.core.config import settings

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
        defi_execution_engine = None,
        news_queue: PriorityNewsQueue,
        severe_flag: multiprocessing.Value,
        symbols: list[str],
        db_session_factory: any = None,
        cycle_interval_seconds: float = 5.0
    ):
        self.market_feed = market_feed
        self.feature_builder = feature_builder
        self.regime_detector = regime_detector
        self.model = model
        self.risk_manager = risk_manager
        self.execution_engine = execution_engine
        self.defi_execution_engine = defi_execution_engine
        self.news_queue = news_queue
        self.severe_flag = severe_flag
        self.cycle_interval_seconds = cycle_interval_seconds
        self.symbols = symbols
        self.db_session_factory = db_session_factory
        
        self.heartbeat_client = HeartbeatClient(news_queue.redis)

        self.feature_sequences: dict[str, deque] = {
            sym: deque(maxlen=self.model.SEQUENCE_LENGTH) for sym in self.symbols
        }
        self.open_trades: dict[str, Trade] = {}
        
        self.position_manager = PositionManager(
            open_trades=self.open_trades,
            cex_execution_engine=self.execution_engine,
            defi_execution_engine=self.defi_execution_engine,
            db_session_factory=self.db_session_factory
        )

        self.current_news_impact: NewsImpact | None = None
        self.news_impact_expires_at: datetime | None = None

    async def _background_predictions_loop(self) -> None:
        """Asynchronous background task to broadcast CNN prediction visualization for the UI, decoupled from trading loop"""
        while True:
            try:
                for symbol in self.symbols:
                    if len(self.feature_sequences[symbol]) < self.model.SEQUENCE_LENGTH:
                        continue
                    
                    df = await self.market_feed.get_dataframe(symbol)
                    if df is None or df.empty:
                        continue
                    
                    recent_trades = await self.market_feed.get_recent_trades(symbol, n=1)
                    if recent_trades and 'price' in recent_trades[-1]:
                        current_price = recent_trades[-1]['price']
                    else:
                        current_price = df.iloc[-1]['close']
                        
                    sequence = np.stack(self.feature_sequences[symbol])
                    
                    # Offload purely for visual rendering
                    decision_str, size_pct, probs = self.model.infer(sequence)
                    
                    p_long = probs.get("long", 0.0)
                    p_short = probs.get("short", 0.0)
                    
                    # Convert AI intention directly to a slope
                    # High long probability = strong upward slope. High short probability = strong downward slope.
                    slope_magnitude = (p_long - p_short) * current_price * 0.005 # Base slope modifier
                    
                    # Size pct drives confidence and volatility cone
                    confidence = max(0.1, size_pct / 0.20) # Normalize to 0-1 range based on max position limit
                    base_expansion = current_price * 0.002 * (1.0 - confidence + 0.1) # Higher confidence = tighter cone

                    predictions = []
                    projected_price = current_price
                    velocity = slope_magnitude
                    
                    for i in range(1, 13):
                        projected_price += velocity
                        velocity *= 0.85 # Decay the velocity to prevent runaway charts (curve shaping)
                        
                        expansion = base_expansion * np.sqrt(i)
                        
                        predictions.append({
                            "step": i,
                            "open": projected_price - (velocity * 0.5),
                            "high": projected_price + expansion,
                            "low": projected_price - expansion,
                            "close": projected_price + (velocity * 0.5)
                        })
                    
                    target_price = predictions[-1]['close']
                    confidence_pct = max(probs.values()) * 100
                    
                    time_range = "10 to 15 minutes"
                    
                    target_buy_price = None
                    target_sell_price = None
                    
                    if target_price > current_price:
                        target_sell_price = target_price
                    else:
                        target_buy_price = target_price

                    if decision_str == "hold":
                        if target_price > current_price:
                            # It expects the price to go UP. It should buy now while it is cheap.
                            price_diff = ((target_price - current_price) / current_price) * 100
                            if price_diff > 0.05:
                                thought = f"Detected significant upside potential (+{price_diff:.2f}%). Price is currently low (~${current_price:,.2f}). Considering entering LONG position to target ~${target_price:,.2f} within {time_range}."
                            else:
                                thought = f"Projecting mild upside to ~${target_price:,.2f} within {time_range}, but current price (~${current_price:,.2f}) isn't low enough for a strong entry. Waiting for better risk/reward."
                        else:
                            price_diff = ((current_price - target_price) / current_price) * 100
                            thought = f"Projecting a dip of ~{price_diff:.2f}% down to ~${target_price:,.2f} over the next {time_range}. Holding off buys until price drops to that support level."
                    elif decision_str == "long":
                        thought = f"Optimal buying conditions met! Current price (~${current_price:,.2f}) is favorable. Executing LONG order to capture projected run-up to ~${target_price:,.2f} ({confidence_pct:.1f}% confidence)."
                    elif decision_str == "short":
                        thought = f"Price is overextended at ~${current_price:,.2f}. Executing SHORT order to capture projected drop to ~${target_price:,.2f} ({confidence_pct:.1f}% confidence)."
                    else:
                        thought = f"Analyzing momentum across recent history and news impact..."

                    payload = {
                        "symbol": symbol,
                        "predictions": predictions,
                        "current_price": current_price,
                        "target_buy_price": target_buy_price,
                        "target_sell_price": target_sell_price,
                        "thought": thought
                    }
                    await self.heartbeat_client.redis.set("agent_visual_predictions", json.dumps(payload))
            except Exception as e:
                logger.error("background_predictions_loop_error", error=str(e))
                
            await asyncio.sleep(self.cycle_interval_seconds)

    async def run(self) -> None:
        logger.info("nn_trading_agent_started", symbols=self.symbols)
        self.started_at = time.time()
        
        # Recover open trades from DB
        if self.db_session_factory:
            try:
                async with self.db_session_factory() as session:
                    stmt = select(Trade).where(Trade.status == TradeStatus.open)
                    result = await session.execute(stmt)
                    recovered_trades = result.scalars().all()
                    for t in recovered_trades:
                        self.open_trades[t.asset] = t
                    logger.info("recovered_open_trades", count=len(recovered_trades))
            except Exception as e:
                logger.error("failed_to_recover_trades", error=str(e))

        self.position_manager.start_monitoring()

        # Pre-fill historical sequences
        for symbol in self.symbols:
            logger.info("prefilling_historical_features", symbol=symbol)
            try:
                df = await self.market_feed.get_dataframe(symbol)
                if df is not None and not df.empty:
                    req = self.model.SEQUENCE_LENGTH
                    available = len(df); logger.info("df_length_check", length=available)
                    if available > req:
                        start_idx = max(0, available - req)
                        for i in range(start_idx, available):
                            sub_df = df.iloc[:i+1] 
                            regime_name, regime_conf = self.regime_detector.detect(sub_df, None)
                            vector = await self.feature_builder.build(
                                symbol=symbol,
                                df=sub_df,
                                bids=[],
                                asks=[],
                                trades=[],
                                sr_levels=[],
                                regime=regime_name,
                                regime_confidence=regime_conf,
                                news_impact=None
                            )
                            self.feature_sequences[symbol].append(vector)
                        logger.info("buffer_filled_successfully", buf_len=len(self.feature_sequences[symbol]))
            except Exception as e:
                logger.error("failed_to_prefill_buffer", symbol=symbol, error=str(e))
                
        # Start background predictions loop
        asyncio.create_task(self._background_predictions_loop())
        
        while True:
            try:
                # Expose state to Redis for the frontend timer and check for manual halt
                try:
                    longest_seq = max([len(seq) for seq in self.feature_sequences.values()]) if self.feature_sequences else 0
                    is_forced_stop = await self.heartbeat_client.redis.get("agent_force_stopped")
                    
                    status_payload = {
                        "is_halted": is_forced_stop == b"true",
                        "buffer_current": longest_seq,
                        "buffer_required": self.model.SEQUENCE_LENGTH,
                        "cycle_interval": self.cycle_interval_seconds,
                        "started_at": self.started_at,
                        "has_market_data": False
                    }
                    
                    # Update has_market_data if any symbol has dataframe
                    for sym in self.symbols:
                        try:
                            df = await self.market_feed.get_dataframe(sym)
                            if df is not None and not df.empty:
                                status_payload["has_market_data"] = True
                        except:
                            pass

                    await self.heartbeat_client.redis.set("agent_frontend_status", json.dumps(status_payload))
                    
                    if is_forced_stop == b"true":
                        await asyncio.sleep(self.cycle_interval_seconds)
                        continue
                except Exception as e:
                    logger.error("redis_status_sync_error", error=str(e))

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
                        regime_confidence=regime_conf,
                        news_impact=self.current_news_impact
                    )
                    
                    self.feature_sequences[symbol].append(vector)
                    
                    if len(self.feature_sequences[symbol]) < self.model.SEQUENCE_LENGTH:
                        continue
                        
                    if time.time() - self.started_at < 300:
                        logger.debug("agent_warming_up", symbol=symbol, remaining_seconds=300 - (time.time() - self.started_at))
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
                    
                    # Calculate true portfolio state
                    from backend.memory.database import get_session
                    async with get_session() as session:
                        # 1. Start with initial amount
                        initial_usdc = settings.INITIAL_USDC_AMOUNT

                        # 2. Add Realized PnL from closed trades
                        result = await session.execute(select(func.sum(Trade.pnl_usd)).where(Trade.status == TradeStatus.closed))
                        realized_pnl = result.scalar() or 0.0

                        # 3. Subtract Locked Cash from open trades
                        result_open = await session.execute(select(func.sum(Trade.size_usd)).where(Trade.status == TradeStatus.open))
                        locked_cash = result_open.scalar() or 0.0

                        available_cash = initial_usdc + realized_pnl - locked_cash
                        if available_cash < 0:
                            available_cash = 0.0
                            
                    portfolio_state = {"available_cash": available_cash, "available_usdc": available_cash}
                    
                    # Prevent opening a duplicate trade in the same direction if we are already holding one
                    if symbol in self.open_trades and hasattr(self.open_trades[symbol], 'direction'):
                        if self.open_trades[symbol].direction.value == decision_str:
                            decision_str = "hold"
                            decision.direction = "hold"
                    
                    approved, reason = self.risk_manager.approve(decision, portfolio_state)
                    if approved:
                        # Routing logic: DeFi if ETH and high confidence
                        if self.defi_execution_engine and symbol == "ETHUSDT" and nn_confidence > 0.85:
                            logger.info("routing_to_defi", symbol=symbol, confidence=nn_confidence)
                            trade = await self.defi_execution_engine.execute(decision, portfolio_state)
                        else:
                            trade = await self.execution_engine.execute(decision, portfolio_state)

                        if trade:
                            self.open_trades[symbol] = trade
                            logger.info("trade_executed", symbol=symbol, direction=decision_str, size=size_pct)
                    else:
                        if decision_str != "hold":
                            logger.info("trade_rejected", symbol=symbol, reason=reason, direction=decision_str)
                        else:
                            logger.info("trade_hold_decision", symbol=symbol, probs=probs, seq_mean=float(sequence.mean()), seq_max=float(sequence.max()))

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
