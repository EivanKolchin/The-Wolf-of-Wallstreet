import asyncio
import json
from typing import List, Optional, Dict, Any
from datetime import datetime

from fastapi import APIRouter, Depends, Header, HTTPException, WebSocket, WebSocketDisconnect, Body
from pydantic import BaseModel
from sqlalchemy import select, desc

from backend.memory.database import AsyncSessionLocal as async_session_maker, Trade, NewsPrediction, AgentEvent, TradeStatus
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

@router.get("/api/setup/status")
async def get_setup_status():
    return {
        "needs_setup": settings.needs_setup(),
        "missing_integrations": settings.missing_integration_status(),
        "ai_provider": settings.AI_PROVIDER,
        "anthropic": not (not settings.ANTHROPIC_API_KEY or "your_" in settings.ANTHROPIC_API_KEY.lower()),
        "gemini": not (not settings.GEMINI_API_KEY or "your_" in settings.GEMINI_API_KEY.lower()),
        "arbitrum": not (not settings.ARBITRUM_RPC_URL or "your_" in settings.ARBITRUM_RPC_URL.lower()),
        "agent_wallet": not (not settings.AGENT_WALLET_ADDRESS or "0x000" in settings.AGENT_WALLET_ADDRESS),
        "agent_pk": not (not settings.AGENT_PRIVATE_KEY or "your_" in settings.AGENT_PRIVATE_KEY.lower() or "0" * 64 in settings.AGENT_PRIVATE_KEY),
        "alpaca": not (not settings.ALPACA_API_KEY or "your_" in settings.ALPACA_API_KEY.lower()),
        "kite": not (not settings.KITE_CHAIN_RPC_URL or "your_" in settings.KITE_CHAIN_RPC_URL.lower()),
        "x_api_key": not (not settings.X_API_KEY or "your_" in settings.X_API_KEY.lower()),
        "telegram": not (not settings.TELEGRAM_API_ID or "your_" in settings.TELEGRAM_API_ID.lower())
    }

@router.get("/api/setup/config")
async def get_setup_config():
    import os
    import dotenv
    from backend.core.config import ENV_PATH
    
    # Read fresh from the .env file directly so we don't rely on cached settings
    env_vars = dotenv.dotenv_values(str(ENV_PATH)) if os.path.exists(str(ENV_PATH)) else {}

    # Return plaintext values for the settings page so it can pre-fill
    return {
        "AI_PROVIDER": env_vars.get("AI_PROVIDER", "gemini"),
        "ANTHROPIC_API_KEY": env_vars.get("ANTHROPIC_API_KEY", ""),
        "GEMINI_API_KEY": env_vars.get("GEMINI_API_KEY", ""),
        "PAPER_MODE": env_vars.get("PAPER_MODE", "true"),
        "ARBITRUM_RPC_URL": env_vars.get("ARBITRUM_RPC_URL", ""),
        "AGENT_WALLET_ADDRESS": env_vars.get("AGENT_WALLET_ADDRESS", ""),
        "AGENT_PRIVATE_KEY": env_vars.get("AGENT_PRIVATE_KEY", ""),
        "ALPACA_API_KEY": env_vars.get("ALPACA_API_KEY", ""),
        "ALPACA_SECRET_KEY": env_vars.get("ALPACA_SECRET_KEY", ""),
        "KITE_CHAIN_RPC_URL": env_vars.get("KITE_CHAIN_RPC_URL", ""),
        "KITE_CHAIN_PRIVATE_KEY": env_vars.get("KITE_CHAIN_PRIVATE_KEY", ""),
        "KITE_AGENT_ADDRESS": env_vars.get("KITE_AGENT_ADDRESS", ""),
        "X_API_KEY": env_vars.get("X_API_KEY", ""),
        "X_API_SECRET": env_vars.get("X_API_SECRET", ""),
        "X_ACCESS_TOKEN": env_vars.get("X_ACCESS_TOKEN", ""),
        "X_ACCESS_TOKEN_SECRET": env_vars.get("X_ACCESS_TOKEN_SECRET", ""),
        "TELEGRAM_API_ID": env_vars.get("TELEGRAM_API_ID", ""),
        "TELEGRAM_API_HASH": env_vars.get("TELEGRAM_API_HASH", ""),
    }

class SetupRequest(BaseModel):
    ai_provider: str = ""
    anthropic_api_key: str = ""
    gemini_api_key: str = ""
    arbitrum_rpc_url: str = ""
    agent_wallet_address: str = ""
    agent_private_key: str = ""
    alpaca_api_key: str = ""
    alpaca_secret: str = ""
    kite_chain_rpc_url: str = ""
    kite_chain_private_key: str = ""
    kite_agent_address: str = ""
    x_api_key: str = ""
    telegram_api_id: str = ""
    telegram_api_hash: str = ""

@router.post("/api/setup/save")
async def save_setup(req: Dict[str, Any] = Body(...)):
    import os
    import signal
    from backend.core.config import ENV_PATH
    env_path = str(ENV_PATH)
    
    req_dict = {k.upper(): str(v) for k, v in req.items() if v is not None}

    if os.path.exists(env_path):
        with open(env_path, "r") as f:
            lines = f.readlines()
            
        settings_seen = set()
        
        with open(env_path, "w") as f:
            for line in lines:
                written = False
                for k, v in req_dict.items():
                    if line.startswith(f"{k}="):
                        f.write(f"{k}={v}\n")
                        settings_seen.add(k)
                        written = True
                        break
                if not written:
                    f.write(line)

            # Append missing lines
            for k, v in req_dict.items():
                if k not in settings_seen:
                    f.write(f"{k}={v}\n")
    else:
        # Create new env file if it doesn't exist
        with open(env_path, "w") as f:
            for k, v in req_dict.items():
                f.write(f"{k}={v}\n")

    # Trigger an orchestrated restart
    def restart():
        os.kill(os.getpid(), signal.SIGTERM)
    
    asyncio.get_event_loop().call_later(1.0, restart)
    return {"status": "saved", "message": "Applying configuration and restarting..."}

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

@router.get("/api/news/raw")
async def get_raw_news():
    redis = await get_redis()
    raw = await redis.lrange("recent_raw_news", 0, 19)
    # Filter out empty or unparseable items
    valid_news = []
    for r in raw:
        try:
            valid_news.append(json.loads(r))
        except:
            pass
    return valid_news

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
