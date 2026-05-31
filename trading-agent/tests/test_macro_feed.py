"""Phase 14 tests: macro/derivatives feed fetchers + cache publishing."""
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from backend.data import macro_feed as mf
from backend.memory.redis_client import FeatureCache, get_redis


def _mock_get(payload, status=200):
    """Helper: produce an `aiohttp.ClientSession.get` MagicMock matching the same
    async-context-manager pattern used in test_defi_engine.py / test_phase7b.py.
    MagicMock auto-handles `__aenter__` / `__aexit__` as async since Py3.8."""
    resp = AsyncMock()
    resp.json = AsyncMock(return_value=payload)
    resp.status = status
    cm = MagicMock()
    cm.return_value.__aenter__.return_value = resp
    return cm


# ----------------------------------------------------------- individual fetchers
@pytest.mark.asyncio
async def test_fetch_fear_greed_parses_alternative_me():
    with patch("aiohttp.ClientSession.get", _mock_get({"data": [{"value": "55"}]})):
        v = await mf.fetch_fear_greed()
    assert v == pytest.approx(0.55)


@pytest.mark.asyncio
async def test_fetch_fear_greed_returns_none_on_bad_payload():
    with patch("aiohttp.ClientSession.get", _mock_get({})):
        assert await mf.fetch_fear_greed() is None


@pytest.mark.asyncio
async def test_fetch_btc_dominance_parses_coingecko():
    payload = {"data": {"market_cap_percentage": {"btc": 51.4, "eth": 17.2}}}
    with patch("aiohttp.ClientSession.get", _mock_get(payload)):
        v = await mf.fetch_btc_dominance()
    assert v == pytest.approx(0.514)


@pytest.mark.asyncio
async def test_fetch_funding_rate_normalises_to_unit_range():
    # Typical funding 0.0001 -> * 10000 -> 1.0 (capped). Use a smaller value to test scaling.
    with patch("aiohttp.ClientSession.get", _mock_get([{"fundingRate": "0.00005"}])):
        v = await mf.fetch_funding_rate("BTCUSDT")
    assert v == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_fetch_funding_rate_clips_extremes():
    with patch("aiohttp.ClientSession.get", _mock_get([{"fundingRate": "0.001"}])):
        v = await mf.fetch_funding_rate("BTCUSDT")
    assert v == 1.0   # clipped


@pytest.mark.asyncio
async def test_fetch_oi_change_computes_pct():
    # 1% OI increase in 5m -> 0.01 / 0.05 = 0.2
    payload = [{"sumOpenInterest": "100.0"}, {"sumOpenInterest": "101.0"}]
    with patch("aiohttp.ClientSession.get", _mock_get(payload)):
        v = await mf.fetch_oi_change("BTCUSDT")
    assert v == pytest.approx(0.2)


@pytest.mark.asyncio
async def test_fetch_oi_change_clips_5_pct_to_unit():
    payload = [{"sumOpenInterest": "100.0"}, {"sumOpenInterest": "105.0"}]
    with patch("aiohttp.ClientSession.get", _mock_get(payload)):
        v = await mf.fetch_oi_change("BTCUSDT")
    assert v == 1.0


# ----------------------------------------------------------- aggregator
@pytest.mark.asyncio
async def test_collect_macro_returns_all_4_keys_when_all_succeed(monkeypatch):
    monkeypatch.setattr(mf, "fetch_fear_greed", AsyncMock(return_value=0.42))
    monkeypatch.setattr(mf, "fetch_btc_dominance", AsyncMock(return_value=0.51))
    monkeypatch.setattr(mf, "fetch_funding_rate", AsyncMock(return_value=0.3))
    monkeypatch.setattr(mf, "fetch_oi_change", AsyncMock(return_value=-0.1))
    payload = await mf.collect_macro(["BTCUSDT"])
    assert payload == {
        "fear_greed_norm": pytest.approx(0.42),
        "btc_dominance_norm": pytest.approx(0.51),
        "funding_rate_norm": pytest.approx(0.3),
        "oi_change_norm": pytest.approx(-0.1),
    }


@pytest.mark.asyncio
async def test_collect_macro_partial_failure_omits_missing_keys(monkeypatch):
    monkeypatch.setattr(mf, "fetch_fear_greed", AsyncMock(return_value=0.42))
    monkeypatch.setattr(mf, "fetch_btc_dominance", AsyncMock(return_value=None))
    monkeypatch.setattr(mf, "fetch_funding_rate", AsyncMock(side_effect=RuntimeError("boom")))
    monkeypatch.setattr(mf, "fetch_oi_change", AsyncMock(return_value=0.0))
    payload = await mf.collect_macro(["BTCUSDT"])
    # Successful keys present; failed keys (None, exception) omitted.
    assert payload == {"fear_greed_norm": pytest.approx(0.42), "oi_change_norm": 0.0}


# ----------------------------------------------------------- end-to-end run_once
@pytest.mark.asyncio
async def test_macro_feed_run_once_publishes_to_feature_cache(monkeypatch):
    monkeypatch.setattr(mf, "fetch_fear_greed", AsyncMock(return_value=0.7))
    monkeypatch.setattr(mf, "fetch_btc_dominance", AsyncMock(return_value=0.55))
    monkeypatch.setattr(mf, "fetch_funding_rate", AsyncMock(return_value=0.1))
    monkeypatch.setattr(mf, "fetch_oi_change", AsyncMock(return_value=-0.05))

    r = await get_redis()
    try:
        await r.delete("features:macro")
    except Exception:
        pass

    feed = mf.MacroFeed(r, symbols=["BTCUSDT"])
    payload = await feed.run_once()
    assert payload["fear_greed_norm"] == pytest.approx(0.7)

    # Verify the live feature builder will now read populated values.
    cached = await FeatureCache(r).get_macro()
    assert cached["fear_greed_norm"] == pytest.approx(0.7)
    assert cached["btc_dominance_norm"] == pytest.approx(0.55)
    assert cached["funding_rate_norm"] == pytest.approx(0.1)
    assert cached["oi_change_norm"] == pytest.approx(-0.05)
