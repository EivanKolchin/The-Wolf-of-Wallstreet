import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

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
