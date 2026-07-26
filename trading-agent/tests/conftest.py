import os
import sys
import tempfile
from pathlib import Path

# The live model now REFUSES to cold-start on random weights (a deliberate safety guard:
# never trade an untrained net live). The test suite, by design, constructs untrained
# models with random weights, so it explicitly opts into the unsafe path. This keeps the
# production safety check intact while letting tests build fresh models.
os.environ.setdefault("FORCE_UNSAFE_START", "true")

# ─────────────────────────────────────────────────────────────────────────────────────────
# TEST ISOLATION — keep the suite OFF the owner's live stores.
#
# `redis_client.get_redis()` pings the REAL Redis first (main.py auto-starts one on :6379) and
# only falls back to FakeRedis when that ping fails. So running pytest while the backend is up
# made tests read and WRITE production state.
#
# Not hypothetical: on 2026-07-09 test_strategy_agent.py (a paper book over the synthetic universe
# A/B/C) drove StrategyAgent.rebalance_once(), which published straight into the owner's live
# `strategy:paperbook:state`. The real book was clobbered — equity 98,995 -> 15,698, phantom
# positions A/B — and the next backend restart restored the wreck.
#
# Defence in depth (plus, in the code itself: rebalance_once() no longer publishes, and
# _restore_paper_book() rejects a state holding symbols outside its universe):
#   1. rewrite the store env vars BEFORE backend.core.config is imported, so no pool/engine can
#      ever be built against a live store;
#   2. an autouse fixture pins redis_client._fake_redis_instance, which get_redis() checks FIRST —
#      this also covers modules that did `from ... import get_redis` (rebinding the module
#      attribute would not have caught those).
# ─────────────────────────────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).parent.parent
for _p in (str(_ROOT), str(_ROOT / "backend")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_TMP_STORE = Path(tempfile.mkdtemp(prefix="wow-tests-"))
# Port 6399 is deliberately closed → ECONNREFUSED → get_redis() falls back to FakeRedis.
os.environ["REDIS_URL"] = "redis://127.0.0.1:6399/0"
os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{(_TMP_STORE / 'test.db').as_posix()}"

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock


@pytest.fixture(scope="session", autouse=True)
def _isolate_shared_stores():
    """Hard-pin FakeRedis + a throwaway trade journal for the whole session."""
    from backend.memory import redis_client
    import fakeredis.aioredis
    redis_client._fake_redis_instance = fakeredis.aioredis.FakeRedis(decode_responses=True)

    # TradeJournal's default_factory reads this module global at construction time.
    from backend.agents import trade_journal
    original_journal = trade_journal.DEFAULT_PATH
    trade_journal.DEFAULT_PATH = _TMP_STORE / "trade_journal.jsonl"

    yield

    trade_journal.DEFAULT_PATH = original_journal
    redis_client._fake_redis_instance = None


@pytest.fixture(autouse=True)
def _refuse_live_redis():
    """Fail loudly if anything re-points Redis at a live server mid-suite."""
    assert os.environ.get("REDIS_URL", "").endswith(":6399/0"), \
        "a test re-pointed REDIS_URL at a live Redis — refusing to run"

# Assuming correct imports map to actual backend structure:
# from backend.db.database import get_session_factory
# from backend.data.redis_client import RedisClient
# from backend.agents.news_agent import NewsImpact
# from backend.agents.nn_agent import TradeDecision

@pytest.fixture
def mock_redis():
    """Mock Redis client."""
    redis = MagicMock()
    redis.get = AsyncMock(return_value=None)
    redis.set = AsyncMock()
    redis.zadd = AsyncMock()
    redis.zpopmin = AsyncMock(return_value=[])
    return redis

@pytest.fixture
def async_session():
    """Mock async DB session."""
    session = AsyncMock()
    session.execute = AsyncMock()
    session.commit = AsyncMock()
    session.refresh = AsyncMock()
    return session

@pytest.fixture
def sample_feature_vector():
    """Returns a realistic 62-element feature vector."""
    vec = np.zeros(62, dtype=np.float32)
    # Give some features realistic sample values
    vec[0] = 0.05  # price_pct_change
    vec[3] = 0.8   # volume_norm
    vec[11] = 0.4  # rsi
    vec[40] = 0.2  # whale_activity
    vec[50] = 0.0  # news_magnitude
    vec[61] = 0.8  # regime_confidence
    return vec

@pytest.fixture
def sample_ohlcv_df():
    """Generates 200 rows of realistic BTC OHLCV data."""
    dates = pd.date_range(end=datetime.utcnow(), periods=200, freq='5min')
    df = pd.DataFrame(index=dates, columns=['open', 'high', 'low', 'close', 'volume', 'timestamp'])
    
    # Random walk
    df['open'] = 60000.0 + np.random.randn(200).cumsum() * 50
    df['high'] = df['open'] + np.random.uniform(10, 100, 200)
    df['low'] = df['open'] - np.random.uniform(10, 100, 200)
    df['close'] = df['open'] + np.random.randn(200) * 20
    df['volume'] = np.random.uniform(5, 100, 200)
    df['timestamp'] = dates
    
    return df

@pytest.fixture
def sample_news_impact():
    """Mock NewsImpact object."""
    class DummyNewsImpact:
        severity = "SIGNIFICANT"
        asset = "BTC"
        direction = "down"
        confidence = 0.75
        magnitude_pct_low = 2.0
        magnitude_pct_high = 5.0
        t_min_minutes = 10
        t_max_minutes = 60
        rationale = "Test."
    return DummyNewsImpact()

@pytest.fixture
def sample_trade_decision(sample_news_impact):
    """Mock TradeDecision."""
    class DummyDecision:
        symbol = "BTCUSDT"
        direction = "long"
        size_pct = 0.05
        nn_confidence = 0.65
        nn_probs = {"long": 0.65, "short": 0.20, "hold": 0.15}
        regime = "ranging"
        active_news = sample_news_impact
        timestamp = datetime.utcnow()
    return DummyDecision()
