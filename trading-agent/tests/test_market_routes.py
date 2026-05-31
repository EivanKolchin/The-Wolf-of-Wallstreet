"""Smoke tests for the multi-asset market-data proxy."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))


def test_market_router_registers_expected_paths():
    from backend.api.market_routes import router
    paths = {r.path for r in router.routes}
    assert "/api/market/klines" in paths
    assert "/api/market/quote" in paths
    assert "/api/market/universe" in paths
    assert "/api/market/depth" in paths


def test_is_us_stock_classifies_known_tickers():
    from backend.api.market_routes import _is_us_stock
    assert _is_us_stock("AMD") is True
    assert _is_us_stock("amd") is True
    assert _is_us_stock("BTCUSDT") is False
    assert _is_us_stock("ETHUSDT") is False


def test_alpaca_tf_map_covers_common_intervals():
    from backend.api.market_routes import _ALPACA_TF_MAP
    for tf in ("1m", "5m", "15m", "1h", "4h", "1d"):
        assert tf in _ALPACA_TF_MAP


@pytest.mark.asyncio
async def test_universe_endpoint_returns_8_cryptos_and_5_stocks():
    from backend.api.market_routes import get_market_universe
    u = await get_market_universe()
    assert "crypto" in u and "stocks" in u
    assert len(u["crypto"]) >= 2
    assert "AMD" in u["stocks"] and "BE" in u["stocks"]


@pytest.mark.asyncio
async def test_klines_stock_path_without_alpaca_credentials_returns_empty_bars(monkeypatch):
    """Without Alpaca keys, the stock path must return a structured empty
    response (never explode and never call Binance with a stock ticker)."""
    import backend.api.market_routes as mr
    monkeypatch.setattr(mr, "_alpaca_available", lambda: False)
    res = await mr.get_market_klines("AMD", "5m", 100)
    assert isinstance(res, dict)
    assert res.get("error") == "alpaca_credentials_missing"
    assert res.get("bars") == []


def test_alpaca_start_iso_returns_well_formed_iso_strings():
    from backend.api.market_routes import _alpaca_start_iso
    start, end = _alpaca_start_iso("5Min", None)
    assert start.endswith("Z") and "T" in start
    # With no end_ms, we fix start but leave end unspecified.
    assert end is None
    start2, end2 = _alpaca_start_iso("1Day", 1716000000000)
    assert start2.endswith("Z") and end2 and end2.endswith("Z")
    assert start2 < end2


def test_stocks_websocket_route_is_registered():
    from backend.api.market_routes import router
    paths = {r.path for r in router.routes}
    assert "/ws/stocks" in paths
