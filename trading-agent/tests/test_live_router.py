"""Tests for the live execution path: venue routing, weight→notional math, safety gates.
Deterministic only — no network (fake brokers)."""
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

import pytest  # noqa: E402

from backend.execution.live_router import LiveRouter, is_perp, make_live_router  # noqa: E402
from backend.execution.binance_futures_broker import quantity_from_notional, sign_query  # noqa: E402
from backend.execution.alpaca_broker import AlpacaBroker  # noqa: E402


class FakeBroker:
    def __init__(self):
        self.calls = []

    async def submit_notional(self, symbol, notional_usd, *, reduce_only=False):
        self.calls.append((symbol, notional_usd, reduce_only))
        return {"status": "paper", "symbol": symbol}


# ── venue classification ──
def test_is_perp_classification():
    assert is_perp("BTCUSDT")
    assert is_perp("ETHUSDT", {"ETHUSDT"})
    assert not is_perp("SPY")
    assert not is_perp("BTC-USD")        # yfinance spot → Alpaca, not a perp


# ── router notional + routing ──
def test_router_routes_perp_to_binance_and_stock_to_alpaca():
    al, bi = FakeBroker(), FakeBroker()
    r = LiveRouter(equity=100_000, alpaca=al, binance=bi, perp_symbols={"BTCUSDT"})
    asyncio.run(r("BTCUSDT", 0.1, 0.1))
    asyncio.run(r("SPY", 0.2, 0.2))
    assert bi.calls and bi.calls[0][0] == "BTCUSDT"
    assert al.calls and al.calls[0][0] == "SPY"


def test_router_notional_is_delta_times_equity():
    bi = FakeBroker()
    r = LiveRouter(equity=100_000, binance=bi, perp_symbols={"BTCUSDT"})
    asyncio.run(r("BTCUSDT", 0.05, 0.05))
    assert bi.calls[0][1] == pytest.approx(0.05 * 100_000)


def test_router_caps_order_notional():
    bi = FakeBroker()
    r = LiveRouter(equity=1_000_000, binance=bi, perp_symbols={"BTCUSDT"}, max_order_notional=5_000)
    asyncio.run(r("BTCUSDT", 0.5, 0.5))           # would be 500k → capped to 5k
    assert bi.calls[0][1] == pytest.approx(5_000)


def test_router_reduce_only_when_shrinking():
    bi = FakeBroker()
    r = LiveRouter(equity=100_000, binance=bi, perp_symbols={"BTCUSDT"})
    # current = target - delta = 0.1 - (-0.06) = 0.16 ; |target|=0.1 < |current|=0.16 → reduce_only
    asyncio.run(r("BTCUSDT", 0.1, -0.06))
    assert bi.calls[0][2] is True


def test_router_not_reduce_only_when_growing():
    bi = FakeBroker()
    r = LiveRouter(equity=100_000, binance=bi, perp_symbols={"BTCUSDT"})
    asyncio.run(r("BTCUSDT", 0.2, 0.1))           # current 0.1 → 0.2, growing
    assert bi.calls[0][2] is False


def test_router_skips_below_min_notional():
    bi = FakeBroker()
    r = LiveRouter(equity=100_000, binance=bi, perp_symbols={"BTCUSDT"}, min_order_notional=10.0)
    res = asyncio.run(r("BTCUSDT", 0.00001, 0.00001))   # 0.00001*100k = $1 < $10 → skipped
    assert res["status"] == "skipped" and not bi.calls


def test_router_missing_broker_errors_gracefully():
    r = LiveRouter(equity=100_000, alpaca=None, binance=None, perp_symbols={"BTCUSDT"})
    res = asyncio.run(r("BTCUSDT", 0.1, 0.1))
    assert res["status"] == "error"


# ── safety gate: default settings (PAPER_TRADING=true) → no live router ──
def test_make_live_router_none_when_paper(monkeypatch):
    from backend.core.config import settings
    monkeypatch.setattr(settings, "PAPER_TRADING", True, raising=False)
    assert make_live_router(equity=100_000) is None


def test_make_live_router_none_without_credentials(monkeypatch):
    from backend.core.config import settings
    monkeypatch.setattr(settings, "PAPER_TRADING", False, raising=False)
    monkeypatch.setattr(settings, "ALPACA_API_KEY", "", raising=False)
    monkeypatch.setattr(settings, "BINANCE_FUTURES_API_KEY", "", raising=False)
    assert make_live_router(equity=100_000) is None   # flag flipped but no keys → still paper


# ── binance helpers ──
def test_quantity_from_notional_floors_to_step():
    assert quantity_from_notional(1000, 50000, step=0.001) == pytest.approx(0.02)
    assert quantity_from_notional(0, 50000) == 0.0
    assert quantity_from_notional(1000, 0) == 0.0


def test_sign_query_appends_signature():
    qs = sign_query({"symbol": "BTCUSDT", "side": "BUY"}, "secret")
    assert "signature=" in qs and "symbol=BTCUSDT" in qs


# ── alpaca symbol normalization ──
def test_alpaca_normalize_symbol():
    assert AlpacaBroker.normalize_symbol("BTC-USD") == "BTC/USD"
    assert AlpacaBroker.normalize_symbol("SPY") == "SPY"
