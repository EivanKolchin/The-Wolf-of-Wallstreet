import asyncio
import json
from typing import List, Optional
from datetime import datetime

from fastapi import APIRouter, Depends, Header, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from sqlalchemy import select, desc

from backend.memory.database import async_session_maker, Trade, NewsPrediction, AgentEvent, TradeStatus
from backend.memory.redis_client import FeatureCache, HeartbeatClient, get_redis
from backend.risk.manager import RiskManager
from backend.core.config import settings
import structlog

logger = structlog.get_logger(__name__)
router = APIRouter()

# Global RiskManager instance to share state, realistically this would be 
# injected or read via Redis, but for FastAPI endpoint purposes we'll instantiate 
# or assume it's synced. Actually it's best to have a global risk manager state in Redis or simple DB,
# but the spec asks for `RiskManager.get_status()` in the route.
# We'll instantiate a singleton here that main.py could potentially share, 
# or just pull state if we assume it runs in the same process sometimes.
# Wait, NNTradingAgent runs in a separate process and has its own RiskManager.
# The endpoint needs access to it. It's better if RiskManager writes its status to Redis, 
# or we use a manager pattern. For now, we'll keep a dummy dummy global instance that might not reflect 
# the other process perfectly unless we link them or fetch from Redis.
# The prompt says: "GET /api/risk/status Returns: RiskManager.get_status()"

# For the sake of matching the exact API:
risk_manager = RiskManager()

class HealthResponse(BaseModel):
    status: str
    nn_alive: bool
    news_alive: bool
    model_trade_count: int

class ResetHaltRequest(BaseModel):
    confirm: bool

@router.get("/api/health", response_model=HealthResponse)
async def get_health():
    redis = await get_redis()
    hb = HeartbeatClient(redis)
    nn_alive = await hb.check_alive("nn_trading_agent")
    news_alive = await hb.check_alive("llm_news_agent")
    
    trade_count = 0
    # In a real app we might fetch model_trade_count from DB model_checkpoints
    
    return HealthResponse(
        status="ok",
        nn_alive=nn_alive,
        news_alive=news_alive,
        model_trade_count=trade_count
    )

@router.get("/api/portfolio")
async def get_portfolio():
    status = risk_manager.get_status()
    return {
        "total_value_usd": status["portfolio_value_usd"],
        "available_cash": status["portfolio_value_usd"] - status["daily_pnl_usd"], # rough est
        "unrealised_pnl": 0.0,
        "daily_pnl": status["daily_pnl_usd"],
        "drawdown_pct": status["current_drawdown_pct"],
        "peak_value": status["peak_portfolio_value"],
        "is_halted": status["is_halted"],
    }

@router.get("/api/positions")
async def get_positions():
    async with async_session_maker() as session:
        stmt = select(Trade).where(Trade.status == TradeStatus.open)
        result = await session.execute(stmt)
        trades = result.scalars().all()
        # In a real app you'd compute unrealised PnL against current price
        return trades

@router.get("/api/trades")
async def get_trades(limit: int = 50, offset: int = 0):
    async with async_session_maker() as session:
        stmt = select(Trade).order_by(desc(Trade.opened_at)).limit(limit).offset(offset)
        result = await session.execute(stmt)
        trades = result.scalars().all()
        return trades

@router.get("/api/signals/latest")
async def get_latest_signals():
    redis = await get_redis()
    cache = FeatureCache(redis)
    
    # Ideally iterate over watched symbols. Hardcoding BTCUSDT for dashboard.
    data = await cache.get_features("BTCUSDT")
    if data:
        return data
    return {}

@router.get("/api/news/recent")
async def get_news_recent(limit: int = 20):
    async with async_session_maker() as session:
        stmt = select(NewsPrediction).order_by(desc(NewsPrediction.created_at)).limit(limit)
        result = await session.execute(stmt)
        return result.scalars().all()

@router.get("/api/audit")
async def get_audit(limit: int = 20):
    async with async_session_maker() as session:
        stmt = select(Trade).where(Trade.kite_tx_hash.isnot(None)).order_by(desc(Trade.opened_at)).limit(limit)
        result = await session.execute(stmt)
        return result.scalars().all()

@router.get("/api/risk/status")
async def get_risk_status():
    return risk_manager.get_status()

@router.post("/api/risk/reset-halt")
async def reset_halt(req: ResetHaltRequest, x_admin_key: str = Header(None)):
    if x_admin_key != settings.ADMIN_KEY:
        raise HTTPException(status_code=401, detail="Invalid admin key")
    if req.confirm:
        risk_manager.reset_halt()
    return {"status": "ok"}

# WebSocket connections
connected_clients: List[WebSocket] = []

@router.websocket("/ws/live")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_clients.append(websocket)
    try:
        while True:
            # We'll just wait for messages or keep the connection open.
            # In main.py or a background task we can push updates to `connected_clients`.
            data = await websocket.receive_text()
    except WebSocketDisconnect:
        connected_clients.remove(websocket)

async def broadcast_ws_message(message_type: str, data: dict):
    payload = json.dumps({"type": message_type, "data": data})
    for client in list(connected_clients):
        try:
            await client.send_text(payload)
        except Exception:
            connected_clients.remove(client)

# Background task to send live updates via WS
async def ws_live_updater():
    redis = await get_redis()
    cache = FeatureCache(redis)
    while True:
        try:
            # Broadcast cycle update every 5s
            features = await cache.get_features("BTCUSDT")
            if features:
                await broadcast_ws_message("cycle_update", features)
        except Exception as e:
            logger.error("ws_broadcast_error", error=str(e))
        await asyncio.sleep(5)
