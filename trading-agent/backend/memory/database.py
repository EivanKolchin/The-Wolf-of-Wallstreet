import enum
import uuid
from datetime import datetime
from contextlib import asynccontextmanager

from sqlalchemy import Column, String, Float, Integer, Text, Boolean, DateTime, Enum as SQLEnum, JSON
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from core.config import settings

# Enums
class Severity(enum.Enum):
    NEUTRAL = "NEUTRAL"
    SIGNIFICANT = "SIGNIFICANT"
    SEVERE = "SEVERE"

class Direction(enum.Enum):
    up = "up"
    down = "down"
    neutral = "neutral"

class TradeDirection(enum.Enum):
    long = "long"
    short = "short"

class TradeStatus(enum.Enum):
    open = "open"
    closed = "closed"
    cancelled = "cancelled"

class OrderType(enum.Enum):
    market = "market"
    limit = "limit"
    twap = "twap"

# Database setup
# Ensure the URL uses the asyncpg driver
db_url = settings.DATABASE_URL
if db_url.startswith("postgresql://"):
    db_url = db_url.replace("postgresql://", "postgresql+asyncpg://", 1)

engine = create_async_engine(db_url, echo=False)
AsyncSessionLocal = async_sessionmaker(
    bind=engine, class_=AsyncSession, expire_on_commit=False
)

class Base(DeclarativeBase):
    pass

class SourceTrustScore(Base):
    __tablename__ = "source_trust_scores"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    source_domain: Mapped[str] = mapped_column(String, unique=True, index=True)
    base_score: Mapped[float] = mapped_column(Float, default=1.0)
    current_score: Mapped[float] = mapped_column(Float, default=1.0)
    total_predictions: Mapped[int] = mapped_column(Integer, default=0)
    correct_predictions: Mapped[int] = mapped_column(Integer, default=0)
    last_updated: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class NewsPrediction(Base):
    __tablename__ = "news_predictions"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    source_domain: Mapped[str] = mapped_column(String)
    headline: Mapped[str] = mapped_column(Text)
    article_hash: Mapped[str] = mapped_column(String, unique=True, index=True)
    severity: Mapped[Severity] = mapped_column(SQLEnum(Severity))
    asset: Mapped[str] = mapped_column(String)
    direction: Mapped[Direction] = mapped_column(SQLEnum(Direction))
    magnitude_pct_low: Mapped[float] = mapped_column(Float)
    magnitude_pct_high: Mapped[float] = mapped_column(Float)
    confidence: Mapped[float] = mapped_column(Float)
    t_min_minutes: Mapped[int] = mapped_column(Integer)
    t_max_minutes: Mapped[int] = mapped_column(Integer)
    rationale: Mapped[str] = mapped_column(Text)
    trust_score_at_time: Mapped[float] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    outcome_checked: Mapped[bool] = mapped_column(Boolean, default=False)
    actual_move_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    prediction_score: Mapped[float | None] = mapped_column(Float, nullable=True)

class Trade(Base):
    __tablename__ = "trades"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    asset: Mapped[str] = mapped_column(String, index=True)
    direction: Mapped[TradeDirection] = mapped_column(SQLEnum(TradeDirection))
    size_usd: Mapped[float] = mapped_column(Float)
    entry_price: Mapped[float] = mapped_column(Float)
    exit_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    status: Mapped[TradeStatus] = mapped_column(SQLEnum(TradeStatus))
    order_type: Mapped[OrderType] = mapped_column(SQLEnum(OrderType))
    nn_confidence: Mapped[float] = mapped_column(Float)
    nn_direction_probs: Mapped[dict] = mapped_column(JSON)
    active_news_impact: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    regime_at_entry: Mapped[str] = mapped_column(String)
    stop_loss: Mapped[float] = mapped_column(Float)
    take_profit: Mapped[float] = mapped_column(Float)
    opened_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    closed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    pnl_usd: Mapped[float | None] = mapped_column(Float, nullable=True)
    pnl_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    kite_tx_hash: Mapped[str | None] = mapped_column(String, nullable=True)

class ModelCheckpoint(Base):
    __tablename__ = "model_checkpoints"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    trade_count: Mapped[int] = mapped_column(Integer)
    file_path: Mapped[str] = mapped_column(String)
    cumulative_pnl: Mapped[float] = mapped_column(Float)
    win_rate_7d: Mapped[float] = mapped_column(Float)
    sharpe_24h: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    is_current: Mapped[bool] = mapped_column(Boolean, default=False)

class AgentEvent(Base):
    __tablename__ = "agent_events"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    event_type: Mapped[str] = mapped_column(String)
    details: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

@asynccontextmanager
async def get_session() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()