"""Phase 4 tests: All-Assets overview stats + extended-hours data add-on."""
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))


def _kline(close, vol):
    # Binance-shaped: [openTime, open, high, low, close, volume, ...]
    return [0, close, close, close, close, vol, 0]


def test_compute_kline_stats_basic():
    from backend.api.market_routes import _compute_kline_stats
    kl = [_kline(100, 10), _kline(110, 20), _kline(105, 15)]
    s = _compute_kline_stats(kl)
    assert s["bars"] == 3
    assert s["last_price"] == 105
    assert s["volume"] == 45
    # +5% net change over the window
    assert abs(s["price_change_pct"] - 5.0) < 1e-9
    assert s["volatility_pct"] is not None and s["volatility_pct"] > 0
    assert s["spark"] == [100, 110, 105]


def test_compute_kline_stats_empty_is_safe():
    from backend.api.market_routes import _compute_kline_stats
    s = _compute_kline_stats([])
    assert s["last_price"] is None and s["bars"] == 0 and s["spark"] == []


def test_spark_downsamples_to_32():
    from backend.api.market_routes import _compute_kline_stats
    kl = [_kline(100 + i, 1) for i in range(200)]
    s = _compute_kline_stats(kl)
    assert len(s["spark"]) == 32


def test_extended_hours_disabled_returns_none(monkeypatch):
    from backend.core.config import settings
    from backend.data import extended_hours_feed as eh
    monkeypatch.setattr(settings, "EXTENDED_HOURS_DATA_ENABLED", False, raising=False)
    res = asyncio.run(eh.get_extended_hours_quote("AMD"))
    assert res is None
