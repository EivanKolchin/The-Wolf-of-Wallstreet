"""Phase 15 tests: higher-timeframe (1h/4h) features + provider cache.

Mirrors the patterns in tests/test_macro_feed.py (aiohttp MagicMock outer +
AsyncMock inner) and tests/test_feature_spec.py (bare-import via sys.path
insert of project root + backend/).
"""
import json
import sys
import types
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from backend.signals import feature_spec as fs
from backend.signals import htf as htf_mod
from backend.signals.htf import (
    HTFFeatureProvider,
    compute_htf_features_for_tf,
    fetch_binance_klines,
)


# ---------------------------------------------------------------- helpers
def _mock_get(payload, status=200):
    """aiohttp.ClientSession.get mock — same pattern as test_macro_feed.py."""
    resp = AsyncMock()
    resp.json = AsyncMock(return_value=payload)
    resp.status = status
    cm = MagicMock()
    cm.return_value.__aenter__.return_value = resp
    return cm


def _make_ohlcv(n: int = 120, base: float = 100.0, drift: float = 0.001) -> pd.DataFrame:
    """Synthetic OHLCV with mild upward drift + small noise. Length >= 50 so the
    talib indicators inside compute_htf_features_for_tf are well-defined."""
    rng = np.random.default_rng(0)
    closes = base * np.exp(np.cumsum(drift + rng.normal(0, 0.001, size=n)))
    highs = closes * (1.0 + 0.002)
    lows = closes * (1.0 - 0.002)
    opens = np.concatenate([[base], closes[:-1]])
    vols = rng.uniform(80, 120, size=n)
    return pd.DataFrame({
        "open": opens, "high": highs, "low": lows, "close": closes, "volume": vols,
    })


def _binance_klines_payload(df: pd.DataFrame) -> list:
    """Pack an OHLCV frame into the Binance klines REST shape."""
    out = []
    for i, row in df.reset_index(drop=True).iterrows():
        out.append([
            int(i * 60_000),                       # open time
            f"{row['open']}", f"{row['high']}", f"{row['low']}", f"{row['close']}",
            f"{row['volume']}",
            int((i + 1) * 60_000),                 # close time
            "0", 0, "0", "0", "0",                 # quote_vol, trades, taker_*, ignore
        ])
    return out


# ---------------------------------------------------------------- compute_htf_features_for_tf
@pytest.mark.skipif(not htf_mod.HAS_TALIB, reason="talib not installed")
def test_compute_htf_1h_returns_4_finite_floats_in_expected_ranges():
    df = _make_ohlcv(120, drift=0.0008)
    v = compute_htf_features_for_tf(df, "1h")
    assert v.shape == (4,) and v.dtype == np.float32
    assert np.all(np.isfinite(v))
    # rsi_norm in [-1, 1]
    assert -1.0 <= v[0] <= 1.0
    # ema21_dist clipped to [-0.1, 0.1]
    assert -0.1 <= v[1] <= 0.1
    # macd_hist_norm clipped to [-1, 1]
    assert -1.0 <= v[2] <= 1.0
    # atr_norm in [0, 1] (atr/close clipped to 0..0.1 then *10)
    assert 0.0 <= v[3] <= 1.0
    # With an upward drift the MACD histogram should be non-zero.
    assert v[2] != 0.0


@pytest.mark.skipif(not htf_mod.HAS_TALIB, reason="talib not installed")
def test_compute_htf_4h_emits_trend_dir_in_slot_2_not_macd():
    # Strong uptrend → ema21 > ema50 → trend_dir = +1
    df_up = _make_ohlcv(200, drift=0.003)
    v_up = compute_htf_features_for_tf(df_up, "4h")
    assert v_up[2] == pytest.approx(1.0)

    # Strong downtrend → ema21 < ema50 → trend_dir = -1
    df_dn = _make_ohlcv(200, drift=-0.003)
    v_dn = compute_htf_features_for_tf(df_dn, "4h")
    assert v_dn[2] == pytest.approx(-1.0)


def test_compute_htf_short_df_returns_zero_vector():
    df = _make_ohlcv(10)   # below the 50-row minimum
    v = compute_htf_features_for_tf(df, "1h")
    assert v.shape == (4,)
    assert np.allclose(v, 0.0)


def test_compute_htf_handles_none_input():
    v = compute_htf_features_for_tf(None, "1h")  # type: ignore[arg-type]
    assert v.shape == (4,) and np.allclose(v, 0.0)


# ---------------------------------------------------------------- fetch_binance_klines
@pytest.mark.asyncio
async def test_fetch_binance_klines_parses_payload_and_floats_cols():
    df_src = _make_ohlcv(60)
    payload = _binance_klines_payload(df_src)
    with patch("aiohttp.ClientSession.get", _mock_get(payload)):
        df = await fetch_binance_klines("BTCUSDT", "1h", limit=60)
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert len(df) == len(df_src)
    for c in df.columns:
        assert df[c].dtype.kind == "f"
    assert df["close"].iloc[-1] == pytest.approx(df_src["close"].iloc[-1])


@pytest.mark.asyncio
async def test_fetch_binance_klines_returns_empty_df_on_network_error():
    """A raise inside the async with → return an empty frame, not propagate."""
    bad_cm = MagicMock()
    bad_cm.return_value.__aenter__.side_effect = RuntimeError("boom")
    with patch("aiohttp.ClientSession.get", bad_cm):
        df = await fetch_binance_klines("BTCUSDT", "1h")
    assert df.empty
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]


@pytest.mark.asyncio
async def test_fetch_binance_klines_returns_empty_df_on_non_list_payload():
    with patch("aiohttp.ClientSession.get", _mock_get({"code": -1, "msg": "bad"})):
        df = await fetch_binance_klines("BTCUSDT", "1h")
    assert df.empty


# ---------------------------------------------------------------- HTFFeatureProvider
class _FakeRedis:
    """Captures setex calls so we can assert the JSON shape."""
    def __init__(self):
        self.store: dict[str, tuple[int, str]] = {}

    async def setex(self, key, ttl, value):
        self.store[key] = (ttl, value)


@pytest.mark.asyncio
@pytest.mark.skipif(not htf_mod.HAS_TALIB, reason="talib not installed")
async def test_provider_refresh_one_populates_inmem_and_redis():
    r = _FakeRedis()
    provider = HTFFeatureProvider(r, symbols=["BTCUSDT"])
    df_1h = _make_ohlcv(120, drift=0.001)
    df_4h = _make_ohlcv(120, drift=0.001)

    async def fake_fetch(symbol, interval, limit=200):
        return df_1h if interval == "1h" else df_4h

    with patch.object(htf_mod, "fetch_binance_klines", side_effect=fake_fetch):
        await provider._refresh_one("BTCUSDT")

    vec = provider.get_features("BTCUSDT")
    assert vec.shape == (8,) and vec.dtype == np.float32
    assert not np.allclose(vec, 0.0), "expected non-zero HTF vector"
    # Redis persistence
    assert "features:htf:BTCUSDT" in r.store
    ttl, raw = r.store["features:htf:BTCUSDT"]
    assert ttl == 1800
    persisted = json.loads(raw)
    assert len(persisted) == 8
    np.testing.assert_allclose(np.asarray(persisted, dtype=np.float32), vec, atol=1e-5)


@pytest.mark.asyncio
async def test_provider_refresh_failure_preserves_last_good():
    provider = HTFFeatureProvider(None, symbols=["BTCUSDT"])
    sentinel = np.arange(8, dtype=np.float32) * 0.1
    provider._latest["BTCUSDT"] = sentinel.copy()

    async def explode(*_a, **_kw):
        raise RuntimeError("rest down")

    with patch.object(htf_mod, "fetch_binance_klines", side_effect=explode):
        await provider._refresh_one("BTCUSDT")

    np.testing.assert_allclose(provider.get_features("BTCUSDT"), sentinel)


def test_provider_get_features_unknown_symbol_returns_zeros():
    provider = HTFFeatureProvider(None, symbols=["BTCUSDT"])
    v = provider.get_features("DOESNOTEXIST")
    assert v.shape == (8,) and np.allclose(v, 0.0)


# ---------------------------------------------------------------- _extend_with_htf integration
def test_extend_with_htf_uses_provider_for_htf_slots():
    """Bind the agent method to a tiny stub and confirm slots [62:70] equal the
    provider's get_features output, while the BASE slots stay intact."""
    from backend.agents.nn_agent import NNTradingAgent

    stub = types.SimpleNamespace()
    # Forge an HTF provider that returns a fixed 8-vec.
    htf_vec = np.array([0.5, -0.04, 0.3, 0.6, 0.2, 0.01, 1.0, 0.4], dtype=np.float32)
    stub.htf_provider = types.SimpleNamespace(get_features=lambda s: htf_vec)

    base = np.arange(fs.BASE, dtype=np.float32) / 100.0
    out = NNTradingAgent._extend_with_htf(stub, base, symbol="BTCUSDT")  # type: ignore[arg-type]

    assert out.shape == (fs.INPUT,)
    np.testing.assert_allclose(out[:fs.BASE], base, atol=1e-6)
    # HTF now fills exactly [HTF_START:HTF_END]; the NEWS_EMBED block follows.
    np.testing.assert_allclose(out[fs.HTF_START:fs.HTF_END], htf_vec, atol=1e-6)
    # The stub has no news embedder, so the NEWS_EMBED block stays zero.
    np.testing.assert_allclose(out[fs.NEWS_EMBED], 0.0)


def test_extend_with_htf_zero_pads_when_provider_absent():
    from backend.agents.nn_agent import NNTradingAgent

    stub = types.SimpleNamespace()
    stub.htf_provider = None

    base = np.arange(fs.BASE, dtype=np.float32) / 100.0
    out = NNTradingAgent._extend_with_htf(stub, base, symbol="BTCUSDT")  # type: ignore[arg-type]

    assert out.shape == (fs.INPUT,)
    np.testing.assert_allclose(out[:fs.BASE], base, atol=1e-6)
    np.testing.assert_allclose(out[fs.HTF_START:fs.HTF_END], 0.0)
    np.testing.assert_allclose(out[fs.NEWS_EMBED], 0.0)


def test_extend_with_htf_passthrough_when_already_full_input():
    from backend.agents.nn_agent import NNTradingAgent

    stub = types.SimpleNamespace()
    stub.htf_provider = None
    full = np.arange(fs.INPUT, dtype=np.float32)
    out = NNTradingAgent._extend_with_htf(stub, full, symbol="BTCUSDT")  # type: ignore[arg-type]
    np.testing.assert_allclose(out, full)
