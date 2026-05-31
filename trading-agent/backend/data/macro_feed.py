"""Phase 14 — macro / derivatives / on-chain feature feed.

Populates the 4 macro slots the live feature builder already reads from
`FeatureCache.get_macro()` (see backend/signals/features.py:217-226). Before this
module existed, those slots trained on neutral defaults forever — every prediction
saw `fear_greed=0.5, btc_dominance=0.5, funding_rate=0, oi_change=0`.

Fetchers (all free):
  - fear_greed     : alternative.me crypto F&G index           -> 0..1
  - btc_dominance  : CoinGecko /global market_cap_percentage   -> 0..1 (btc share)
  - funding_rate   : Binance Futures /fapi/v1/fundingRate      -> symmetric ~[-1, 1]
  - oi_change      : Binance Futures /futures/data/openInterestHist -> symmetric ~[-1, 1]

Rate-limited via `backend.core.rate_limiter.binance_rest_limiter` for Binance calls.
Every outbound REST is also recorded via `backend/core/ledger.log_api_call(...)`.

Designed to be spawned as a long-running asyncio task in the NN agent process
(`asyncio.create_task(MacroFeed(redis, symbols=[...]).run())`). Single fetcher
failure does NOT take the loop down — each call is independently guarded and the
last-good values stay in Redis.
"""
from __future__ import annotations

import asyncio
import time
from typing import Iterable

import aiohttp
import structlog

from backend.core import ledger
from backend.core.rate_limiter import binance_rest_limiter
from backend.memory.redis_client import FeatureCache

logger = structlog.get_logger(__name__)

ALTERNATIVE_ME_FNG = "https://api.alternative.me/fng/?limit=1"
COINGECKO_GLOBAL = "https://api.coingecko.com/api/v3/global"
BINANCE_FAPI = "https://fapi.binance.com"  # futures endpoints aren't on binance.us


async def _get_json(url: str, *, provider: str, timeout_s: float = 8.0,
                    headers: dict | None = None, params: dict | None = None):
    """Rate-limited GET with API-call logging. Returns parsed JSON or None on failure."""
    if provider == "binance":
        await binance_rest_limiter().acquire()
    t0 = time.monotonic()
    try:
        timeout = aiohttp.ClientTimeout(total=timeout_s)
        async with aiohttp.ClientSession(timeout=timeout) as s:
            async with s.get(url, headers=headers, params=params) as r:
                data = await r.json(content_type=None)
                ledger.log_api_call(provider, "GET", url, status=r.status, ok=True,
                                    latency_ms=(time.monotonic() - t0) * 1000)
                return data
    except Exception as e:
        ledger.log_api_call(provider, "GET", url, ok=False,
                            latency_ms=(time.monotonic() - t0) * 1000, note=str(e)[:120])
        return None


# ----------------------------------------------------------- individual fetchers
async def fetch_fear_greed() -> float | None:
    """Crypto Fear & Greed index, 0..1 (0 = extreme fear, 1 = extreme greed)."""
    data = await _get_json(ALTERNATIVE_ME_FNG, provider="alternative.me")
    try:
        if data and "data" in data and data["data"]:
            return float(data["data"][0]["value"]) / 100.0
    except Exception:
        pass
    return None


async def fetch_btc_dominance() -> float | None:
    """BTC share of total crypto market cap, 0..1."""
    data = await _get_json(COINGECKO_GLOBAL, provider="coingecko")
    try:
        if data and "data" in data:
            return float(data["data"]["market_cap_percentage"]["btc"]) / 100.0
    except Exception:
        pass
    return None


async def fetch_funding_rate(symbol: str) -> float | None:
    """Latest perpetual funding rate, normalised to ~[-1, 1] by * 10000.

    Typical funding is 0.0001 .. 0.001 per 8h (= 0.01% .. 0.1%). Multiplying by
    10000 puts the common range inside [-1, +1] without clipping in normal times;
    we clip the tail anyway for safety.
    """
    url = f"{BINANCE_FAPI}/fapi/v1/fundingRate"
    data = await _get_json(url, provider="binance", params={"symbol": symbol, "limit": 1})
    try:
        if isinstance(data, list) and data:
            raw = float(data[0]["fundingRate"])
            return max(-1.0, min(1.0, raw * 10000.0))
    except Exception:
        pass
    return None


async def fetch_oi_change(symbol: str, period: str = "5m") -> float | None:
    """5-min open-interest % change, clipped to ~[-1, 1] by dividing by 5%.

    A 5% OI change in 5 minutes is already extreme; 1% normalises to 0.2.
    """
    url = f"{BINANCE_FAPI}/futures/data/openInterestHist"
    data = await _get_json(url, provider="binance",
                           params={"symbol": symbol, "period": period, "limit": 2})
    try:
        if isinstance(data, list) and len(data) >= 2:
            prev = float(data[-2]["sumOpenInterest"])
            curr = float(data[-1]["sumOpenInterest"])
            if prev > 0:
                pct = (curr - prev) / prev
                return max(-1.0, min(1.0, pct / 0.05))   # 5% move -> ±1.0
    except Exception:
        pass
    return None


# ----------------------------------------------------------- aggregator + loop
async def collect_macro(symbols: Iterable[str]) -> dict:
    """Run all fetchers concurrently. Missing values are simply omitted so the
    feature builder's per-key defaults stay in force for whatever failed."""
    symbols = list(symbols) or ["BTCUSDT"]
    primary = symbols[0]
    fg_t, dom_t, fr_t, oi_t = await asyncio.gather(
        fetch_fear_greed(),
        fetch_btc_dominance(),
        fetch_funding_rate(primary),
        fetch_oi_change(primary),
        return_exceptions=True,
    )
    out: dict = {}
    def _put(k, v):
        if v is None or isinstance(v, Exception):
            return
        out[k] = float(v)
    _put("fear_greed_norm", fg_t)
    _put("btc_dominance_norm", dom_t)
    _put("funding_rate_norm", fr_t)
    _put("oi_change_norm", oi_t)
    return out


class MacroFeed:
    """Periodic macro feed — publishes to FeatureCache every `poll_seconds`."""

    def __init__(self, redis_client, symbols: Iterable[str], poll_seconds: float = 300.0):
        self.redis = redis_client
        self.cache = FeatureCache(redis_client)
        self.symbols = list(symbols)
        self.poll_seconds = float(poll_seconds)
        self._running = False

    async def run_once(self) -> dict:
        payload = await collect_macro(self.symbols)
        if payload:
            try:
                await self.cache.set_macro(payload)
                logger.info("macro_feed_published", keys=sorted(payload.keys()))
            except Exception as e:
                logger.warning("macro_feed_publish_failed", error=str(e))
        else:
            logger.warning("macro_feed_no_data")
        return payload

    async def run(self) -> None:
        self._running = True
        logger.info("macro_feed_started", symbols=self.symbols, poll_seconds=self.poll_seconds)
        while self._running:
            try:
                await self.run_once()
            except Exception as e:
                logger.error("macro_feed_loop_error", error=str(e))
            await asyncio.sleep(self.poll_seconds)

    def stop(self) -> None:
        self._running = False
