"""Model performance / health statistics endpoints.

Aggregates trade-level PnL + win/loss, news-pipeline classification counts,
outbound API latency, and live runtime metadata (uptime, buffers, cycles) into
a single ``/api/stats/performance`` payload the frontend can poll. Provides
``/api/stats/reset`` to baseline the counters from "now" forward.
"""
from __future__ import annotations

import json
import math
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import structlog
from fastapi import APIRouter
from sqlalchemy import select, func

from backend.memory.database import (
    AsyncSessionLocal as async_session_maker,
    Trade, NewsPrediction, TradeStatus, Severity,
)
from backend.memory.redis_client import get_redis

logger = structlog.get_logger(__name__)
router = APIRouter()

ROOT = Path(__file__).resolve().parents[2]
API_CALLS_JSONL = ROOT / "statements" / "api_calls.jsonl"
RESET_BASELINE_KEY = "stats:reset_baseline_ts"


# ---------------------------------------------------------- helpers
def _utcnow() -> datetime:
    return datetime.utcnow()


def _today_utc_start() -> datetime:
    n = _utcnow()
    return datetime(n.year, n.month, n.day)


def _sharpe(returns: list[float], annualization: float = 365.0) -> float:
    """Annualized Sharpe of a per-trade return series. Returns 0.0 if undefined."""
    if not returns or len(returns) < 2:
        return 0.0
    mu = sum(returns) / len(returns)
    var = sum((r - mu) ** 2 for r in returns) / max(len(returns) - 1, 1)
    sd = math.sqrt(var)
    if sd <= 0:
        return 0.0
    return float(mu / sd * math.sqrt(annualization))


def _sortino(returns: list[float], annualization: float = 365.0) -> float:
    if not returns or len(returns) < 2:
        return 0.0
    mu = sum(returns) / len(returns)
    downside = [min(0.0, r) ** 2 for r in returns]
    dd = math.sqrt(sum(downside) / len(returns))
    if dd <= 0:
        return float("inf") if mu > 0 else 0.0
    return float(mu / dd * math.sqrt(annualization))


def _percentile(xs: list[float], q: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    k = (len(s) - 1) * q
    f = int(math.floor(k)); c = int(math.ceil(k))
    if f == c:
        return float(s[f])
    return float(s[f] + (s[c] - s[f]) * (k - f))


async def _baseline_ts() -> float | None:
    try:
        r = await get_redis()
        raw = await r.get(RESET_BASELINE_KEY)
        if raw is None:
            return None
        if isinstance(raw, bytes):
            raw = raw.decode()
        return float(raw)
    except Exception:
        return None


# ---------------------------------------------------------- trade stats core
def _trade_stats(trades: list[Trade]) -> dict:
    """Pure-Python aggregation over an in-memory list of Trade rows."""
    n = len(trades)
    open_trades = [t for t in trades if t.status == TradeStatus.open]
    closed = [t for t in trades if t.status == TradeStatus.closed]
    n_open = len(open_trades); n_closed = len(closed)

    pnls_usd = [float(t.pnl_usd or 0.0) for t in closed]
    pnls_pct = [float(t.pnl_pct or 0.0) for t in closed]
    wins = [p for p in pnls_pct if p > 0]
    losses = [p for p in pnls_pct if p < 0]
    win_rate = (len(wins) / n_closed) if n_closed else 0.0
    avg_win = (sum(wins) / len(wins)) if wins else 0.0
    avg_loss = (sum(losses) / len(losses)) if losses else 0.0
    expectancy = win_rate * avg_win + (1.0 - win_rate) * avg_loss
    profit_factor = (sum(wins) / abs(sum(losses))) if losses and sum(losses) != 0 else (float("inf") if wins else 0.0)

    # Holding times (only for closed)
    hold_minutes = []
    for t in closed:
        if t.opened_at and t.closed_at:
            hold_minutes.append((t.closed_at - t.opened_at).total_seconds() / 60.0)
    avg_hold_min = (sum(hold_minutes) / len(hold_minutes)) if hold_minutes else 0.0

    # Direction breakdown
    n_long = sum(1 for t in trades if getattr(t.direction, "value", None) == "long")
    n_short = sum(1 for t in trades if getattr(t.direction, "value", None) == "short")

    # Exit reason breakdown
    exit_reasons: dict = {}
    for t in closed:
        k = (t.exit_reason or "unknown").lower()
        exit_reasons[k] = exit_reasons.get(k, 0) + 1

    # Confidence + size summaries
    confs = [float(t.nn_confidence or 0.0) for t in trades]
    sizes = [float(t.size_usd or 0.0) for t in trades]

    # Time bracket
    if trades:
        first_open = min((t.opened_at for t in trades if t.opened_at), default=None)
    else:
        first_open = None
    if first_open:
        elapsed_days = max((datetime.utcnow() - first_open).total_seconds() / 86400.0, 1e-6)
        trades_per_day = n / elapsed_days
    else:
        trades_per_day = 0.0

    # Target-price prediction error (Phase 13 ledger)
    abs_errs = []
    for t in closed:
        tgt = getattr(t, "target_price", None) or 0.0
        ent = getattr(t, "entry_price", None) or 0.0
        exit_p = getattr(t, "exit_price", None) or 0.0
        if tgt and ent and exit_p:
            err = abs((exit_p - tgt) / ent)
            abs_errs.append(err)
    avg_abs_target_err = (sum(abs_errs) / len(abs_errs)) if abs_errs else 0.0

    # Best / worst
    best = max(pnls_pct) if pnls_pct else 0.0
    worst = min(pnls_pct) if pnls_pct else 0.0
    biggest_pnl_usd = (max(pnls_usd) if pnls_usd else 0.0)
    largest_loss_usd = (min(pnls_usd) if pnls_usd else 0.0)

    # Streaks
    longest_win_streak = 0; longest_loss_streak = 0
    cur_w = cur_l = 0
    for p in pnls_pct:
        if p > 0:
            cur_w += 1; cur_l = 0
            longest_win_streak = max(longest_win_streak, cur_w)
        elif p < 0:
            cur_l += 1; cur_w = 0
            longest_loss_streak = max(longest_loss_streak, cur_l)
        else:
            cur_w = cur_l = 0

    return {
        "total_trades": n,
        "open_trades": n_open,
        "closed_trades": n_closed,
        "long_trades": n_long,
        "short_trades": n_short,
        "win_rate": round(win_rate, 4),
        "wins": len(wins),
        "losses": len(losses),
        "avg_win_pct": round(avg_win * 100.0, 4),
        "avg_loss_pct": round(avg_loss * 100.0, 4),
        "expectancy_pct": round(expectancy * 100.0, 4),
        "profit_factor": round(profit_factor, 4) if math.isfinite(profit_factor) else None,
        "best_trade_pct": round(best * 100.0, 4),
        "worst_trade_pct": round(worst * 100.0, 4),
        "biggest_pnl_usd": round(biggest_pnl_usd, 2),
        "largest_loss_usd": round(largest_loss_usd, 2),
        "cumulative_pnl_usd": round(sum(pnls_usd), 2),
        "cumulative_pnl_pct": round(sum(pnls_pct) * 100.0, 4),
        "avg_holding_minutes": round(avg_hold_min, 1),
        "trades_per_day": round(trades_per_day, 3),
        "sharpe_per_trade": round(_sharpe(pnls_pct), 3),
        "sortino_per_trade": round(_sortino(pnls_pct), 3),
        "avg_nn_confidence": round((sum(confs) / len(confs)) if confs else 0.0, 4),
        "avg_size_usd": round((sum(sizes) / len(sizes)) if sizes else 0.0, 2),
        "exit_reasons": exit_reasons,
        "longest_win_streak": longest_win_streak,
        "longest_loss_streak": longest_loss_streak,
        "avg_abs_target_err_pct": round(avg_abs_target_err * 100.0, 4),
    }


# ---------------------------------------------------------- news + latency
def _news_stats(news: list[NewsPrediction]) -> dict:
    n = len(news)
    sev_counts = {"NEUTRAL": 0, "SIGNIFICANT": 0, "SEVERE": 0}
    for p in news:
        try:
            sev_counts[p.severity.value] = sev_counts.get(p.severity.value, 0) + 1
        except Exception:
            pass
    confs = [float(p.confidence or 0.0) for p in news]
    avg_conf = (sum(confs) / len(confs)) if confs else 0.0
    # Predicted-vs-actual: only rows where outcome_checked
    scored = [p for p in news if getattr(p, "outcome_checked", False)
              and p.actual_move_pct is not None and p.prediction_score is not None]
    avg_pred_score = (sum(p.prediction_score for p in scored) / len(scored)) if scored else 0.0
    avg_actual_pct = (sum(p.actual_move_pct for p in scored) / len(scored)) if scored else 0.0
    return {
        "total": n,
        "minor_count": sev_counts.get("NEUTRAL", 0),       # NEUTRAL == "minor" semantically
        "significant_count": sev_counts.get("SIGNIFICANT", 0),
        "severe_count": sev_counts.get("SEVERE", 0),
        "avg_confidence": round(avg_conf, 4),
        "outcome_checked": len(scored),
        "avg_prediction_score": round(avg_pred_score, 4),
        "avg_actual_move_pct": round(avg_actual_pct * 100.0, 4),
    }


def _latency_stats(since_ts: float | None = None) -> dict:
    """Read the api_calls.jsonl ledger (Phase 9) and aggregate by provider."""
    if not API_CALLS_JSONL.exists():
        return {"providers": {}, "totals": {"calls": 0, "errors": 0, "error_rate": 0.0}}
    by_provider: dict = {}
    total = 0; total_err = 0
    try:
        with API_CALLS_JSONL.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    e = json.loads(line)
                except Exception:
                    continue
                if since_ts is not None:
                    try:
                        ts = datetime.fromisoformat(e["ts"].replace("Z", "+00:00")).timestamp()
                        if ts < since_ts:
                            continue
                    except Exception:
                        pass
                prov = e.get("provider") or "unknown"
                lat = e.get("latency_ms")
                ok = e.get("ok")
                d = by_provider.setdefault(prov, {"calls": 0, "errors": 0, "lat_ms": []})
                d["calls"] += 1
                total += 1
                if ok is False:
                    d["errors"] += 1
                    total_err += 1
                if isinstance(lat, (int, float)):
                    d["lat_ms"].append(float(lat))
    except Exception as ex:
        logger.warning("latency_read_failed", error=str(ex))
        return {"providers": {}, "totals": {"calls": 0, "errors": 0, "error_rate": 0.0}}

    providers = {}
    for prov, d in by_provider.items():
        lats = d["lat_ms"]
        providers[prov] = {
            "calls": d["calls"],
            "errors": d["errors"],
            "error_rate": round((d["errors"] / d["calls"]) if d["calls"] else 0.0, 4),
            "avg_latency_ms": round((sum(lats) / len(lats)) if lats else 0.0, 1),
            "p50_latency_ms": round(_percentile(lats, 0.50), 1),
            "p95_latency_ms": round(_percentile(lats, 0.95), 1),
        }
    return {
        "providers": providers,
        "totals": {
            "calls": total, "errors": total_err,
            "error_rate": round((total_err / total) if total else 0.0, 4),
        },
    }


# ---------------------------------------------------------- runtime meta
async def _runtime_meta() -> dict:
    """Pull live model/agent runtime markers from Redis (set by NN agent)."""
    out: dict = {
        "agent_uptime_seconds": None,
        "buffer_current": None,
        "buffer_required": None,
        "cycle_interval": None,
        "has_market_data": None,
        "is_halted": None,
        "attention_state": {},
        "open_position_count": 0,
        "feature_version": None,
    }
    try:
        r = await get_redis()
        raw = await r.get("agent_frontend_status")
        if raw:
            s = json.loads(raw if isinstance(raw, str) else raw.decode())
            started = s.get("started_at")
            if isinstance(started, (int, float)):
                out["agent_uptime_seconds"] = max(0.0, time.time() - float(started))
            out["buffer_current"] = s.get("buffer_current")
            out["buffer_required"] = s.get("buffer_required")
            out["cycle_interval"] = s.get("cycle_interval")
            out["has_market_data"] = s.get("has_market_data")
            out["is_halted"] = s.get("is_halted")
        att = await r.get("attention:state")
        if att:
            try:
                out["attention_state"] = json.loads(att if isinstance(att, str) else att.decode())
            except Exception:
                pass
    except Exception as e:
        logger.debug("runtime_meta_read_failed", error=str(e))
    try:
        from backend.signals import feature_spec as fs
        out["feature_version"] = fs.VERSION
    except Exception:
        pass
    return out


# ============================================================ endpoints
@router.get("/api/stats/performance")
async def stats_performance(scope: str = "all", symbol: str | None = None):
    """scope = 'all' | 'today' (after baseline reset, 'all' means since-reset).

    Returns a single payload with:
      meta:        runtime markers (uptime, buffers, feature version)
      overall:     aggregated trade stats
      today:       same shape, scoped to today
      per_symbol:  {symbol: trade_stats}
      news:        severity counts + scoring averages
      latency:     per-provider api latency + error rate
      reset_at:    epoch seconds of last /reset (None if never reset)
    """
    baseline = await _baseline_ts()
    baseline_dt = datetime.utcfromtimestamp(baseline) if baseline else None
    today = _today_utc_start()

    async with async_session_maker() as session:
        q = select(Trade)
        if baseline_dt:
            q = q.where(Trade.opened_at >= baseline_dt)
        if symbol:
            q = q.where(Trade.asset == symbol)
        trades = list((await session.execute(q)).scalars().all())

        q_today = select(Trade).where(Trade.opened_at >= today)
        if baseline_dt and baseline_dt > today:
            q_today = q_today.where(Trade.opened_at >= baseline_dt)
        trades_today = list((await session.execute(q_today)).scalars().all())

        nq = select(NewsPrediction)
        if baseline_dt:
            nq = nq.where(NewsPrediction.created_at >= baseline_dt)
        news = list((await session.execute(nq)).scalars().all())

    # Group per-symbol
    per_symbol_groups: dict = {}
    for t in trades:
        per_symbol_groups.setdefault(t.asset, []).append(t)

    payload = {
        "meta": await _runtime_meta(),
        "overall": _trade_stats(trades),
        "today": _trade_stats(trades_today),
        "per_symbol": {sym: _trade_stats(rows) for sym, rows in sorted(per_symbol_groups.items())},
        "news": _news_stats(news),
        "latency": _latency_stats(since_ts=baseline),
        "reset_at": baseline,
        "scope": scope,
        "generated_at": datetime.utcnow().isoformat(),
    }
    return payload


@router.post("/api/stats/reset")
async def stats_reset():
    """Mark a baseline timestamp; subsequent performance queries count from here."""
    try:
        r = await get_redis()
        ts = time.time()
        await r.set(RESET_BASELINE_KEY, str(ts))
        return {"ok": True, "reset_at": ts}
    except Exception as e:
        logger.warning("stats_reset_failed", error=str(e))
        return {"ok": False, "error": str(e)}


@router.delete("/api/stats/reset")
async def stats_reset_clear():
    """Clear the baseline so subsequent queries count from genesis again."""
    try:
        r = await get_redis()
        await r.delete(RESET_BASELINE_KEY)
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}
