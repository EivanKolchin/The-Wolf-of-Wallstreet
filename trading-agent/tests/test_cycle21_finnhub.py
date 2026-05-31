"""Cycle 21 — Finnhub fallback for Alpaca historical bars."""
import sys
from pathlib import Path
from unittest.mock import patch, AsyncMock, MagicMock

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))


def test_finnhub_resolution_map_covers_common_intervals():
    from backend.api.market_routes import _FINNHUB_RESOLUTION
    for tf in ("1m", "5m", "15m", "1h", "4h", "1d"):
        assert tf in _FINNHUB_RESOLUTION
    # Alpaca-form intervals must also map (so callers using either form work).
    for tf in ("5Min", "1Hour", "1Day"):
        assert tf in _FINNHUB_RESOLUTION


def test_finnhub_available_false_when_no_key(monkeypatch):
    import backend.api.market_routes as mr
    from backend.core.config import settings
    monkeypatch.setattr(settings, "FINNHUB_API_KEY", "")
    assert mr._finnhub_available() is False


def test_finnhub_available_false_when_placeholder(monkeypatch):
    import backend.api.market_routes as mr
    from backend.core.config import settings
    monkeypatch.setattr(settings, "FINNHUB_API_KEY", "your_finnhub_key_here")
    assert mr._finnhub_available() is False


def test_finnhub_available_true_when_real_key(monkeypatch):
    import backend.api.market_routes as mr
    from backend.core.config import settings
    monkeypatch.setattr(settings, "FINNHUB_API_KEY", "abc123xyz")
    assert mr._finnhub_available() is True


@pytest.mark.asyncio
async def test_fallback_fires_on_alpaca_transport_error(monkeypatch):
    """Alpaca request raises → Finnhub returns bars → bars get propagated."""
    import backend.api.market_routes as mr
    from backend.core.config import settings

    monkeypatch.setattr(settings, "ALPACA_API_KEY", "PK_test_key")
    monkeypatch.setattr(settings, "ALPACA_SECRET_KEY", "secret_44_chars_long_aaaaaaaaaaaaaaaaaaaaaaa")
    monkeypatch.setattr(settings, "FINNHUB_API_KEY", "fnh_test_key")
    monkeypatch.setattr(mr, "_alpaca_available", lambda: True)
    monkeypatch.setattr(mr, "_finnhub_available", lambda: True)

    # Force the Alpaca path to raise (transport-level failure).
    failing_session = MagicMock()
    failing_session.__aenter__ = AsyncMock(side_effect=RuntimeError("boom"))
    monkeypatch.setattr(mr, "make_resilient_session", lambda **kw: failing_session)

    # Finnhub returns a known dataset.
    bars_payload = [[1700000000000, "100", "101", "99", "100.5", "1000", 1700000000000, "0", 0, "0", "0", "0"]]
    monkeypatch.setattr(mr, "_finnhub_bars_to_klines", AsyncMock(return_value=bars_payload))

    res = await mr._alpaca_bars_to_klines("AMD", "5m", 50)
    assert isinstance(res, list)
    assert res == bars_payload


@pytest.mark.asyncio
async def test_fallback_silent_when_finnhub_unavailable(monkeypatch):
    """Without a Finnhub key, transport-level Alpaca failure surfaces the
    original error rather than silently empty."""
    import backend.api.market_routes as mr
    from backend.core.config import settings

    monkeypatch.setattr(settings, "ALPACA_API_KEY", "PK_test")
    monkeypatch.setattr(settings, "ALPACA_SECRET_KEY", "secret_44_chars_long_aaaaaaaaaaaaaaaaaaaaaaa")
    monkeypatch.setattr(settings, "FINNHUB_API_KEY", "")
    monkeypatch.setattr(mr, "_alpaca_available", lambda: True)

    failing_session = MagicMock()
    failing_session.__aenter__ = AsyncMock(side_effect=RuntimeError("network down"))
    monkeypatch.setattr(mr, "make_resilient_session", lambda **kw: failing_session)

    res = await mr._alpaca_bars_to_klines("AMD", "5m", 50)
    assert isinstance(res, dict)
    assert res["error"] == "alpaca_request_failed"
    assert "network down" in res["detail"]
    assert res["bars"] == []
