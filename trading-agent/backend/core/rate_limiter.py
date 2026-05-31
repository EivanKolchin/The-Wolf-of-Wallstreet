"""Async token-bucket rate limiter to keep outbound API calls under exchange limits.

A process-local registry hands out a shared limiter per named bucket (e.g. the
Binance REST bucket) so every component in a process throttles against the same
budget. Each agent process has its own buckets; defaults are conservative enough
that the sum across processes stays well under Binance's ~1200 weight/min.
"""
from __future__ import annotations

import asyncio
import time


class AsyncRateLimiter:
    def __init__(self, rate_per_sec: float, burst: float | None = None, name: str = ""):
        self.rate = max(0.001, float(rate_per_sec))
        self.capacity = float(burst) if burst is not None else max(1.0, float(rate_per_sec))
        self.tokens = self.capacity
        self.updated = time.monotonic()
        self.name = name
        self._lock = asyncio.Lock()

    async def acquire(self, cost: float = 1.0) -> None:
        """Block until `cost` tokens are available, then consume them."""
        async with self._lock:
            while True:
                now = time.monotonic()
                self.tokens = min(self.capacity, self.tokens + (now - self.updated) * self.rate)
                self.updated = now
                if self.tokens >= cost:
                    self.tokens -= cost
                    return
                await asyncio.sleep((cost - self.tokens) / self.rate)


_LIMITERS: dict[str, AsyncRateLimiter] = {}


def get_limiter(name: str, rate_per_sec: float, burst: float | None = None) -> AsyncRateLimiter:
    """Return the shared limiter for `name`, creating it on first use."""
    lim = _LIMITERS.get(name)
    if lim is None:
        lim = AsyncRateLimiter(rate_per_sec, burst, name)
        _LIMITERS[name] = lim
    return lim


def binance_rest_limiter() -> AsyncRateLimiter:
    """Shared limiter for all Binance public REST calls in this process."""
    try:
        from backend.core.config import settings
        rate = float(getattr(settings, "BINANCE_RATE_LIMIT_PER_SEC", 8.0))
    except Exception:
        rate = 8.0
    return get_limiter("binance_rest", rate_per_sec=rate, burst=max(rate, 10.0))
