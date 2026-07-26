"""Live data providers for the two-sleeve StrategyAgent: 4h crypto bars for the TS-momentum sleeve,
and a recent-news view for the directional news overlay.

  * Binance4hBarProvider.get_bars(symbol) → 4h OHLCV DataFrame (TTL-cached) from Binance futures
    klines — the feed the TS-momentum sleeve's live target-weights are computed from.
  * RecentNewsStore — an in-process, TTL-windowed store of NewsImpacts (add / impacts_for); right
    for a single-process run or tests.
  * RedisNewsProvider — reads recent NewsImpacts from a Redis ZSET (``news:recent``) the news agent
    publishes to, so the trading process and the (separate) news process share a view. Best-effort:
    any redis hiccup yields an empty list (news is an optional overlay, never a hard dependency).

Both news providers expose ``impacts_for(symbol) -> list`` — exactly what StrategyAgent.news_provider
consumes (it also tolerates a coroutine, so the Redis one can be async)."""
from __future__ import annotations

import json
import time
from collections import deque
from typing import Deque, List, Optional, Tuple

import pandas as pd

try:
    import structlog
    logger = structlog.get_logger("live_providers")
except Exception:  # pragma: no cover
    import logging
    logger = logging.getLogger("live_providers")

from backend.memory.redis_client import NewsImpact

RECENT_NEWS_KEY = "news:recent"


# ───────────────────────── 4h crypto bars (TS-momentum sleeve) ─────────────────────────
class Binance4hBarProvider:
    """Fetch 4h klines from Binance USD-M futures with a short TTL cache (the sleeve rebalances
    far less often than the cache lifetime, so this re-pulls a couple of times a day)."""

    def __init__(self, *, limit: int = 500, ttl_seconds: float = 3600.0, testnet: Optional[bool] = None):
        self.limit = limit
        self.ttl = ttl_seconds
        # ALWAYS mainnet for market DATA — testnet klines are synthetic (thin, fabricated prices)
        # and the sleeve's breakout/ATR/EMA signals are meaningless on them. Order ROUTING (a
        # separate broker) still honours the testnet flag; public data must be real.
        self.base = "https://fapi.binance.com"
        self._cache: dict[str, Tuple[float, pd.DataFrame]] = {}

    def get_bars(self, symbol: str) -> Optional[pd.DataFrame]:
        now = time.time()
        hit = self._cache.get(symbol)
        if hit and now - hit[0] < self.ttl:
            return hit[1]
        try:
            import requests
            url = f"{self.base}/fapi/v1/klines?symbol={symbol}&interval=4h&limit={self.limit}"
            rows = requests.get(url, timeout=10).json()
            if not isinstance(rows, list) or not rows:
                return hit[1] if hit else None
            df = pd.DataFrame({
                "timestamp": pd.to_datetime([int(r[0]) for r in rows], unit="ms"),
                "open": [float(r[1]) for r in rows], "high": [float(r[2]) for r in rows],
                "low": [float(r[3]) for r in rows], "close": [float(r[4]) for r in rows],
                "volume": [float(r[5]) for r in rows],
            })
            self._cache[symbol] = (now, df)
            return df
        except Exception as e:
            logger.warning("binance_4h_fetch_failed", symbol=symbol, error=str(e)[:100])
            return hit[1] if hit else None


# ───────────────── multi-timeframe bars (QuantileTCN/DMN overlay gate) ─────────────────
class BinanceMultiTFProvider:
    """5m + 1h + 4h klines for one symbol — the hybrid-feature contract's inputs. 5m depth
    defaults to 1200 (the rolling z-score needs ~1100). One request per TF, TTL-cached."""

    _INTERVALS = {"5m": 1200, "1h": 300, "4h": 300}

    def __init__(self, *, ttl_seconds: float = 900.0, testnet: Optional[bool] = None):
        self.ttl = ttl_seconds
        self.base = "https://fapi.binance.com"   # market data is always mainnet (see Binance4hBarProvider)
        self._cache: dict[str, Tuple[float, dict]] = {}

    def _klines(self, symbol: str, interval: str, limit: int) -> Optional[pd.DataFrame]:
        try:
            import requests
            url = (f"{self.base}/fapi/v1/klines?symbol={symbol}"
                   f"&interval={interval}&limit={limit}")
            rows = requests.get(url, timeout=10).json()
            if not isinstance(rows, list) or not rows:
                return None
            return pd.DataFrame({
                "timestamp": pd.to_datetime([int(r[0]) for r in rows], unit="ms"),
                "open": [float(r[1]) for r in rows], "high": [float(r[2]) for r in rows],
                "low": [float(r[3]) for r in rows], "close": [float(r[4]) for r in rows],
                "volume": [float(r[5]) for r in rows],
            })
        except Exception as e:
            logger.warning("binance_multitf_fetch_failed", symbol=symbol,
                           interval=interval, error=str(e)[:100])
            return None

    def get_multi_tf(self, symbol: str) -> Optional[dict]:
        now = time.time()
        hit = self._cache.get(symbol)
        if hit and now - hit[0] < self.ttl:
            return hit[1]
        out = {tf: self._klines(symbol, tf, n) for tf, n in self._INTERVALS.items()}
        if any(v is None for v in out.values()):
            return hit[1] if hit else None
        self._cache[symbol] = (now, out)
        return out


# ───────────────────────── recent-news views (overlay) ─────────────────────────
def _relevant(impact: NewsImpact, symbol: str) -> bool:
    s = str(symbol).upper()
    if str(getattr(impact, "asset", "")).upper() == s:
        return True
    rel = getattr(impact, "symbol_relevance", None)
    return bool(isinstance(rel, dict) and (rel.get(symbol) or rel.get(s)))


class RecentNewsStore:
    """In-process TTL store of NewsImpacts. The news agent calls ``add``; the StrategyAgent reads
    ``impacts_for``. Window defaults to 6h (matches the overlay's stale horizon)."""

    def __init__(self, ttl_seconds: float = 6 * 3600.0, maxlen: int = 500):
        self.ttl = ttl_seconds
        self._items: Deque[Tuple[float, NewsImpact]] = deque(maxlen=maxlen)

    def add(self, impact: NewsImpact) -> None:
        self._items.append((time.time(), impact))

    def _live(self) -> List[NewsImpact]:
        cutoff = time.time() - self.ttl
        return [imp for (ts, imp) in self._items if ts >= cutoff]

    def impacts_for(self, symbol: str) -> List[NewsImpact]:
        return [imp for imp in self._live() if _relevant(imp, symbol)]


class RedisNewsProvider:
    """Reads recent NewsImpacts from the ``news:recent`` ZSET (scored by epoch seconds) the news
    agent publishes to. async ``impacts_for`` — StrategyAgent awaits it. Best-effort + read-only."""

    def __init__(self, redis, *, ttl_seconds: float = 6 * 3600.0, key: str = RECENT_NEWS_KEY):
        self.redis = redis
        self.ttl = ttl_seconds
        self.key = key

    @staticmethod
    async def publish(redis, impact: NewsImpact, *, key: str = RECENT_NEWS_KEY,
                      keep_seconds: float = 6 * 3600.0) -> None:
        """Called by the news agent: record an impact (scored by now) and trim the stale tail."""
        try:
            now = time.time()
            await redis.zadd(key, {impact.to_json(): now})
            await redis.zremrangebyscore(key, 0, now - keep_seconds)
        except Exception as e:
            logger.warning("recent_news_publish_failed", error=str(e)[:100])

    async def impacts_for(self, symbol: str) -> List[NewsImpact]:
        try:
            cutoff = time.time() - self.ttl
            raw = await self.redis.zrange(self.key, 0, -1, withscores=True)
            out: List[NewsImpact] = []
            for member, score in raw or []:
                if score < cutoff:
                    continue
                try:
                    imp = NewsImpact.from_json(member if isinstance(member, str) else member.decode())
                except Exception:
                    continue
                if _relevant(imp, symbol):
                    out.append(imp)
            return out
        except Exception as e:
            logger.warning("recent_news_read_failed", symbol=symbol, error=str(e)[:100])
            return []
