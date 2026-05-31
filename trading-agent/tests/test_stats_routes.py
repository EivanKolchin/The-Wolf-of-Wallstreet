"""Smoke tests for the model-performance stats endpoints."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))


def test_stats_router_importable_and_routes_registered():
    from backend.api.stats_routes import router
    paths = {r.path for r in router.routes}
    assert "/api/stats/performance" in paths
    assert "/api/stats/reset" in paths


def test_trade_stats_pure_aggregator_empty_returns_zero_block():
    from backend.api.stats_routes import _trade_stats
    s = _trade_stats([])
    assert s["total_trades"] == 0
    assert s["win_rate"] == 0.0
    assert s["cumulative_pnl_usd"] == 0.0
    assert s["expectancy_pct"] == 0.0
    assert s["sharpe_per_trade"] == 0.0


def test_trade_stats_handles_synthetic_trades():
    from types import SimpleNamespace
    from backend.api.stats_routes import _trade_stats
    from backend.memory.database import TradeStatus, TradeDirection
    from datetime import datetime, timedelta

    now = datetime.utcnow()
    rows = [
        SimpleNamespace(
            asset="BTCUSDT", direction=TradeDirection.long, status=TradeStatus.closed,
            opened_at=now - timedelta(minutes=30), closed_at=now - timedelta(minutes=20),
            pnl_usd=10.0, pnl_pct=0.005, nn_confidence=0.6, size_usd=100.0,
            exit_reason="take_profit", entry_price=100.0, exit_price=100.5,
            target_price=100.5,
        ),
        SimpleNamespace(
            asset="BTCUSDT", direction=TradeDirection.short, status=TradeStatus.closed,
            opened_at=now - timedelta(minutes=15), closed_at=now - timedelta(minutes=10),
            pnl_usd=-3.0, pnl_pct=-0.002, nn_confidence=0.5, size_usd=80.0,
            exit_reason="stop_loss", entry_price=200.0, exit_price=200.4,
            target_price=199.0,
        ),
        SimpleNamespace(
            asset="ETHUSDT", direction=TradeDirection.long, status=TradeStatus.open,
            opened_at=now - timedelta(minutes=5), closed_at=None,
            pnl_usd=None, pnl_pct=None, nn_confidence=0.7, size_usd=60.0,
            exit_reason=None, entry_price=3000.0, exit_price=None,
            target_price=3050.0,
        ),
    ]
    s = _trade_stats(rows)
    assert s["total_trades"] == 3
    assert s["open_trades"] == 1
    assert s["closed_trades"] == 2
    assert s["wins"] == 1 and s["losses"] == 1
    assert s["win_rate"] == 0.5
    assert s["long_trades"] == 2 and s["short_trades"] == 1
    assert s["cumulative_pnl_usd"] == 7.0
    assert "take_profit" in s["exit_reasons"] and "stop_loss" in s["exit_reasons"]
    assert s["avg_holding_minutes"] > 0


def test_news_stats_groups_by_severity():
    from types import SimpleNamespace
    from backend.api.stats_routes import _news_stats
    from backend.memory.database import Severity

    rows = [
        SimpleNamespace(severity=Severity.NEUTRAL, confidence=0.4, outcome_checked=False, actual_move_pct=None, prediction_score=None),
        SimpleNamespace(severity=Severity.SIGNIFICANT, confidence=0.7, outcome_checked=True, actual_move_pct=0.01, prediction_score=0.8),
        SimpleNamespace(severity=Severity.SEVERE, confidence=0.9, outcome_checked=True, actual_move_pct=0.05, prediction_score=0.6),
    ]
    s = _news_stats(rows)
    assert s["total"] == 3
    assert s["minor_count"] == 1
    assert s["significant_count"] == 1
    assert s["severe_count"] == 1
    assert s["outcome_checked"] == 2


def test_sharpe_and_sortino_helpers_basic():
    from backend.api.stats_routes import _sharpe, _sortino, _percentile
    assert _sharpe([]) == 0.0
    s = _sharpe([0.01, 0.02, -0.005, 0.015])
    assert s != 0.0
    assert _sortino([]) == 0.0
    assert _percentile([], 0.5) == 0.0
    assert _percentile([10, 20, 30, 40, 50], 0.5) == 30
