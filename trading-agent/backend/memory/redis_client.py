import json
from dataclasses import dataclass, asdict
from typing import Optional
from redis.asyncio import Redis, ConnectionPool
from backend.core.config import settings

# Redis connection pool
try:
    import fakeredis.aioredis
    _has_fake_redis = True
except ImportError:
    _has_fake_redis = False

redis_pool = ConnectionPool.from_url(settings.REDIS_URL, decode_responses=True)

# Try real redis first, fallback to fakeredis if connection fails or fake requested implicitly
_fake_redis_instance = None

async def get_redis() -> Redis:
    """Returns a Redis connection. Uses a mock if Redis isn't running."""
    global _fake_redis_instance
    if _fake_redis_instance:
        return _fake_redis_instance

    real_redis = Redis(connection_pool=redis_pool)
    try:
        # Actually attempt to execute a command to force a connection check
        await real_redis.ping()
        return real_redis
    except Exception as e:
        # Server not found or connection refused, fallback to fakeredis
        import logging
        logging.getLogger(__name__).warning("Real Redis unreachable. Falling back to FakeRedis.")

    if _has_fake_redis:
        if not _fake_redis_instance:
            _fake_redis_instance = fakeredis.aioredis.FakeRedis(decode_responses=True)
        return _fake_redis_instance
    else:
        # Re-raise or return broken real_redis to let it fail if no fakeredis
        return real_redis


@dataclass
class NewsImpact:
    severity: str
    asset: str
    direction: str
    magnitude_pct_low: float
    magnitude_pct_high: float
    confidence: float
    t_min_minutes: int
    t_max_minutes: int
    rationale: str
    source_domain: str
    trust_score: float
    created_at: str
    symbol_relevance: dict[str, float] | None = None
    matched_keywords: dict[str, list[str]] | None = None

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, data: str) -> "NewsImpact":
        return cls(**json.loads(data))


class PriorityNewsQueue:
    QUEUE_KEY = "news:priority_queue"

    def __init__(self, redis: Redis):
        self.redis = redis

    async def put(self, impact: NewsImpact) -> None:
        score = 0 if impact.severity == "SEVERE" else 1
        # Store as standard string mapped to its score.
        # We can add a unique identifier if multiple identical payloads are expected,
        # but ZADD inherently handles uniqueness by the member string.
        await self.redis.zadd(self.QUEUE_KEY, {impact.to_json(): score})

    async def get_nowait(self) -> Optional[NewsImpact]:
        # Pop the element with the lowest score using legacy Redis 3.0 support
        items = await self.redis.zrange(self.QUEUE_KEY, 0, 0, withscores=True)
        if not items:
            return None
        
        member, _score = items[0]
        await self.redis.zrem(self.QUEUE_KEY, member)
        return NewsImpact.from_json(member)

    async def size(self) -> int:
        return await self.redis.zcard(self.QUEUE_KEY)


class FeatureCache:
    def __init__(self, redis: Redis):
        self.redis = redis

    async def set_features(self, asset: str, features: dict) -> None:
        key = f"features:asset:{asset}"
        await self.redis.setex(key, 60, json.dumps(features))

    async def get_features(self, asset: str) -> Optional[dict]:
        key = f"features:asset:{asset}"
        data = await self.redis.get(key)
        if data:
            return json.loads(data)
        return None

    async def set_macro(self, data: dict) -> None:
        key = "features:macro"
        await self.redis.setex(key, 900, json.dumps(data))

    async def get_macro(self) -> Optional[dict]:
        key = "features:macro"
        data = await self.redis.get(key)
        if data:
            return json.loads(data)
        return None


class HeartbeatClient:
    def __init__(self, redis: Redis):
        self.redis = redis

    async def ping(self, process_name: str) -> None:
        import time
        key = f"heartbeat:{process_name}"
        timestamp = str(time.time())
        await self.redis.setex(key, 10, timestamp)

    async def check_alive(self, process_name: str) -> bool:
        key = f"heartbeat:{process_name}"
        result = await self.redis.exists(key)
        return result > 0
