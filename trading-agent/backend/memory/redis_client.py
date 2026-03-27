import json
from dataclasses import dataclass, asdict
from typing import Optional
from redis.asyncio import Redis, ConnectionPool
from core.config import settings

# Redis connection pool
redis_pool = ConnectionPool.from_url(settings.REDIS_URL, decode_responses=True)

async def get_redis() -> Redis:
    """Returns a Redis connection from the pool."""
    return Redis(connection_pool=redis_pool)


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
        # Pop the element with the lowest score
        result = await self.redis.zpopmin(self.QUEUE_KEY, count=1)
        if not result:
            return None
        
        member, _score = result[0]
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
