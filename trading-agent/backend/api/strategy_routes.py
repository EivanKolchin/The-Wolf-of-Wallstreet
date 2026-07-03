"""StrategyAgent paper-book endpoint — serves the dashboard's allocation + net-worth view.

The StrategyAgent runs in a SEPARATE process from this FastAPI app, so it publishes its paper
book snapshot to Redis (``strategy:portfolio``) after each rebalance; this endpoint just reads
and returns it. When the agent hasn't published yet (not enabled / first tick pending) it
returns an ``inactive`` payload the frontend renders as an empty-state, never an error.
"""
from __future__ import annotations

import json

import structlog
from fastapi import APIRouter

from backend.memory.redis_client import get_redis

logger = structlog.get_logger(__name__)
router = APIRouter()

_EMPTY = {
    "active": False,
    "total_value": 0.0,
    "initial_value": 0.0,
    "cash": 0.0,
    "gross_exposure": 0.0,
    "net_exposure": 0.0,
    "total_pnl": 0.0,
    "total_pnl_pct": 0.0,
    "allocations": [],
    "history": [],
    "paper": True,
}


@router.get("/api/strategy/portfolio")
async def strategy_portfolio():
    """The managed-beta + TS-momentum paper book: total value, cash, per-asset allocations
    (dollar value + weight + side), and the net-worth history for the equity curve."""
    try:
        r = await get_redis()
        raw = await r.get("strategy:portfolio")
        if not raw:
            return _EMPTY
        view = json.loads(raw if isinstance(raw, str) else raw.decode())
        view["active"] = True
        return view
    except Exception as e:
        logger.debug("strategy_portfolio_read_failed", error=str(e))
        return _EMPTY
