"""Phase 12 tests: AttentionController.replace_overrides + /api/attention GET/POST."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from backend.agents.attention_controller import AttentionController, Attention
from backend.api.routes import get_attention, set_attention, AttentionOverrideRequest


def test_replace_overrides_applies_and_clears():
    ac = AttentionController()
    ac.replace_overrides({"BTCUSDT": "high", "ETHUSDT": "low"})
    assert ac.get_override("BTCUSDT") == Attention.HIGH
    assert ac.get_override("ETHUSDT") == Attention.LOW
    ac.replace_overrides({})                                  # clear all
    assert ac.get_override("BTCUSDT") is None and ac.get_override("ETHUSDT") is None


def test_replace_overrides_ignores_garbage_values():
    ac = AttentionController()
    ac.replace_overrides({"BTCUSDT": "nonsense", "ETHUSDT": "high"})
    assert ac.get_override("BTCUSDT") is None
    assert ac.get_override("ETHUSDT") == Attention.HIGH


@pytest.mark.asyncio
async def test_api_attention_post_then_get_roundtrip():
    # Clear any prior state from earlier tests
    from backend.memory.redis_client import get_redis
    r = await get_redis()
    try:
        await r.delete("attention:overrides")
    except Exception:
        pass

    # POST set
    res = await set_attention(AttentionOverrideRequest(symbol="BTCUSDT", attention="high"))
    assert res["ok"] is True and res["overrides"].get("BTCUSDT") == "high"

    payload = await get_attention()
    assert payload["overrides"].get("BTCUSDT") == "high"

    # POST clear (auto)
    res2 = await set_attention(AttentionOverrideRequest(symbol="BTCUSDT", attention="auto"))
    assert "BTCUSDT" not in res2["overrides"]


@pytest.mark.asyncio
async def test_api_attention_post_rejects_bad_value():
    from fastapi import HTTPException
    with pytest.raises(HTTPException):
        await set_attention(AttentionOverrideRequest(symbol="BTCUSDT", attention="ultra"))
