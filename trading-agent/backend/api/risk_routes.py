import json
import time

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel

from backend.core.config import settings
from backend.memory.redis_client import get_redis
from backend.risk.manager import RiskManager

router = APIRouter()
risk_manager = RiskManager()


class ResetHaltRequest(BaseModel):
    confirm: bool


@router.get("/api/risk/status")
async def get_risk_status():
    redis = await get_redis()
    raw = await redis.get("risk:status")
    if raw:
        try:
            data = json.loads(raw)
            last_update_ts = float(data.get("updated_at_ts", 0.0))
            age_seconds = max(0.0, time.time() - last_update_ts) if last_update_ts > 0 else None
            data["stale"] = age_seconds is None or age_seconds > 15.0
            data["age_seconds"] = age_seconds
            return data
        except Exception:
            pass
    fallback = risk_manager.get_status()
    fallback["stale"] = True
    fallback["age_seconds"] = None
    fallback["source"] = "fallback_local_instance"
    return fallback


@router.post("/api/risk/reset-halt")
async def reset_halt(req: ResetHaltRequest, x_admin_key: str = Header(None)):
    if x_admin_key != settings.ADMIN_KEY:
        raise HTTPException(status_code=401, detail="Invalid admin key")
    if req.confirm:
        redis = await get_redis()
        await redis.set("risk:reset_requested", "true", ex=30)
        risk_manager.reset_halt()
    return {"status": "ok"}
