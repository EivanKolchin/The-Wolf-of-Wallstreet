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
    symbol_relevance: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    matched_keywords: Mapped[dict | None] = mapped_column(JSON, nullable=True)

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
    quote_asset: Mapped[str | None] = mapped_column(String, nullable=True, default="USDC")
    fee_paid: Mapped[float | None] = mapped_column(Float, nullable=True, default=0.0)
    exit_reason: Mapped[str | None] = mapped_column(String, nullable=True)
    trailing_stop: Mapped[float | None] = mapped_column(Float, nullable=True, default=None)
    highest_price_seen: Mapped[float | None] = mapped_column(Float, nullable=True, default=None)
    broker: Mapped[str | None] = mapped_column(String, nullable=True, default="uniswap_v3")
    account_id: Mapped[str | None] = mapped_column(String, nullable=True)
    asset_class: Mapped[str | None] = mapped_column(String, nullable=True, default="crypto")
    target_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    expected_execution_ts: Mapped[float | None] = mapped_column(Float, nullable=True)
    rationale: Mapped[dict | None] = mapped_column(JSON, nullable=True)

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


class KeywordWeight(Base):
    """Learnable keyword bank: per (symbol, keyword) weight nudged by news outcomes.
    Seeded from CRYPTO_KEYWORD_BANK; near-zero weights are effectively pruned."""
    __tablename__ = "keyword_weights"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    symbol: Mapped[str] = mapped_column(String, index=True)
    keyword: Mapped[str] = mapped_column(String, index=True)
    weight: Mapped[float] = mapped_column(Float, default=1.0)
    hits: Mapped[int] = mapped_column(Integer, default=0)
    correct: Mapped[int] = mapped_column(Integer, default=0)
    last_updated: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class SeverityCalibration(Base):
    """Tracks predicted vs realized magnitude per severity bucket for LLM calibration."""
    __tablename__ = "severity_calibration"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    severity_bucket: Mapped[str] = mapped_column(String, unique=True, index=True)
    sum_predicted: Mapped[float] = mapped_column(Float, default=0.0)
    sum_realized: Mapped[float] = mapped_column(Float, default=0.0)
    count: Mapped[int] = mapped_column(Integer, default=0)


class CorrelationSnapshot(Base):
    """Phase 10 append-only correlation matrix snapshot (never overwritten).
    Each training cycle / periodic compute appends a row so structural shifts in
    cross-asset relationships are preserved for the relational engine + audit."""
    __tablename__ = "correlation_snapshots"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    version: Mapped[int] = mapped_column(Integer, index=True)
    matrix: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class Account(Base):
    """A broker account (one per broker+venue), with a base settlement currency."""
    __tablename__ = "accounts"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    broker: Mapped[str] = mapped_column(String, index=True)        # uniswap_v3 | alpaca | ibkr
    venue: Mapped[str] = mapped_column(String)                     # uniswap | nasdaq | nyse | lse
    base_currency: Mapped[str] = mapped_column(String, default="USD")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class CashLedger(Base):
    """Per-account, per-currency balance. FX-minimizing: proceeds settle in the
    instrument's quote currency; conversion only via explicit deposit/withdraw."""
    __tablename__ = "cash_ledger"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    account_id: Mapped[str] = mapped_column(String, index=True)
    currency: Mapped[str] = mapped_column(String, index=True)
    balance: Mapped[float] = mapped_column(Float, default=0.0)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


def _ensure_columns(sync_conn):
    """Idempotently ADD COLUMN for new fields on existing tables (SQLite-safe).
    create_all() creates missing tables but never alters existing ones."""
    from sqlalchemy import inspect as _inspect, text as _text
    insp = _inspect(sync_conn)
    tables = set(insp.get_table_names())
    wanted = {
        "trades": {
            "quote_asset": "VARCHAR",
            "fee_paid": "FLOAT",
            "exit_reason": "VARCHAR",
            "trailing_stop": "FLOAT",
            "highest_price_seen": "FLOAT",
            "broker": "VARCHAR",
            "account_id": "VARCHAR",
            "asset_class": "VARCHAR",
            "target_price": "FLOAT",
            "expected_execution_ts": "FLOAT",
            "rationale": "JSON",
        },
        "news_predictions": {
            "symbol_relevance": "JSON",
            "matched_keywords": "JSON",
        },
    }
    for table, cols in wanted.items():
        if table not in tables:
            continue
        have = {c["name"] for c in insp.get_columns(table)}
        for col, coltype in cols.items():
            if col not in have:
                sync_conn.execute(_text(f"ALTER TABLE {table} ADD COLUMN {col} {coltype}"))


async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await conn.run_sync(_ensure_columns)

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