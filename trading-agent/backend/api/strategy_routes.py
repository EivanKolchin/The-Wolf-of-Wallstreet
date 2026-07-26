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


@router.get("/api/strategy/journal")
async def strategy_journal(limit: int = 100):
    """Recent paper-book activity (rebalances + routed orders) from the StrategyAgent's
    JSONL trade journal. The agent writes it to disk; this process reads the same file.
    Drives the audit page so it shows real paper activity even when no discrete broker
    trades exist in the Trade table."""
    try:
        from backend.agents.trade_journal import TradeJournal
        rows = TradeJournal().read(kinds=["rebalance", "order"])
        rows.sort(key=lambda r: float(r.get("ts", 0)), reverse=True)
        return {"active": bool(rows), "events": rows[: max(1, min(limit, 500))]}
    except Exception as e:
        logger.debug("strategy_journal_read_failed", error=str(e))
        return {"active": False, "events": []}


@router.get("/api/strategy/health")
async def strategy_health():
    """Book-health metrics the audit showed were missing: realized vol vs target, realized
    (annualized) Sharpe, current/worst drawdown, uptime coverage of the mark cadence, active
    news de-gears, and anomaly-flag count. Computed from the trade journal + strategy:status +
    strategy:anomalies. This is the panel that would have surfaced 'running at half risk' months ago."""
    import math
    try:
        from backend.agents.trade_journal import TradeJournal
        rows = TradeJournal().read(kinds=["mark", "rebalance"])
    except Exception:
        rows = []
    marks = [r for r in rows if r.get("kind") == "mark"]
    rebs = [r for r in rows if r.get("kind") == "rebalance"]

    status: dict = {}
    anomalies: dict = {}
    try:
        r = await get_redis()
        raw = await r.get("strategy:status")
        status = json.loads(raw if isinstance(raw, str) else raw.decode()) if raw else {}
        raw = await r.get("strategy:anomalies")
        anomalies = json.loads(raw if isinstance(raw, str) else raw.decode()) if raw else {}
    except Exception:
        pass

    rets = [float(m.get("book_return", 0.0)) for m in marks]
    ts = [float(m.get("ts", 0.0)) for m in marks if m.get("ts")]
    eq = [float(m.get("equity", 0.0)) for m in marks if m.get("equity")]

    # annualization factor from the median mark cadence
    ann = 0.0
    median_dt = 0.0
    if len(ts) >= 3:
        dts = sorted(b - a for a, b in zip(ts, ts[1:]) if b > a)
        if dts:
            median_dt = dts[len(dts) // 2]
            if median_dt > 0:
                ann = math.sqrt(365.0 * 86400.0 / median_dt)

    def _std(xs):
        if len(xs) < 2:
            return 0.0
        m = sum(xs) / len(xs)
        return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))

    realized_vol = _std(rets) * ann if ann else 0.0
    mean_ret = (sum(rets) / len(rets)) if rets else 0.0
    sharpe = (mean_ret / _std(rets) * ann) if (_std(rets) > 0 and ann) else 0.0

    # drawdown from the equity curve
    cur_dd = worst_dd = 0.0
    if eq:
        peak = eq[0]
        for e in eq:
            peak = max(peak, e)
            dd = (e / peak - 1.0) if peak > 0 else 0.0
            worst_dd = min(worst_dd, dd)
        cur_dd = (eq[-1] / max(eq) - 1.0) if max(eq) > 0 else 0.0

    # uptime coverage: fraction of expected marks present over the observed span (gaps = downtime)
    coverage = 1.0
    gaps = 0
    if len(ts) >= 3 and median_dt > 0:
        span = ts[-1] - ts[0]
        expected = span / median_dt + 1
        coverage = min(1.0, len(ts) / expected) if expected > 0 else 1.0
        gaps = sum(1 for a, b in zip(ts, ts[1:]) if (b - a) > 3 * median_dt)

    # active news de-gears from the most recent rebalance
    news_degears = {}
    if rebs:
        news_degears = rebs[-1].get("news") or {}
    flags = (anomalies.get("flags") if isinstance(anomalies, dict) else None) or {}

    book_vt = float(status.get("book_vol_target", 0.0))
    target_vol = book_vt if book_vt > 0 else float(status.get("sleeve_vol_target", 0.0))

    return {
        "active": bool(marks),
        "marks": len(marks),
        "realized_vol": round(realized_vol, 4),
        "target_vol": round(target_vol, 4),
        "vol_utilization": round(realized_vol / target_vol, 3) if target_vol > 0 else None,
        "realized_sharpe": round(sharpe, 3),
        "current_drawdown": round(cur_dd, 4),
        "worst_drawdown": round(worst_dd, 4),
        "uptime_coverage": round(coverage, 3),
        "mark_gaps": gaps,
        "median_mark_minutes": round(median_dt / 60.0, 1) if median_dt else None,
        "gross": status.get("gross"),
        "equity": status.get("equity"),
        "mode": status.get("mode"),
        "blocked": status.get("blocked"),
        "news_degears": news_degears,
        "anomaly_flags": {k: len(v) for k, v in flags.items()} if isinstance(flags, dict) else {},
        "rebalances": len(rebs),
    }


@router.get("/api/strategy/graph")
async def strategy_graph():
    """The live entity graph: nodes + directed edges with curated PRIORS and the current
    correlation-validated LIVE weights (0 = the narrative is dead right now). Drives the
    dashboard's cross-asset propagation network view."""
    try:
        r = await get_redis()
        raw = await r.get("strategy:graph")
        if not raw:
            return {"active": False, "nodes": [], "edges": []}
        g = json.loads(raw if isinstance(raw, str) else raw.decode())
        g["active"] = True
        return g
    except Exception as e:
        logger.debug("strategy_graph_read_failed", error=str(e))
        return {"active": False, "nodes": [], "edges": []}
