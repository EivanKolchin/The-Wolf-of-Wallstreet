"""Phase 6 tests: news persistence bug fix + learnable keyword bank + outcome feedback."""
import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from sqlalchemy import select
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

from backend.memory.database import (
    Base, NewsPrediction, KeywordWeight, SeverityCalibration, Severity, Direction,
)
from backend.training.backbone import extract_symbol_relevance
from backend.agents.news_agent import LLMNewsAgent


async def _make_db(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path.as_posix()}/news.db")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    return async_sessionmaker(engine, expire_on_commit=False)


def _make_agent(SessionLocal):
    nq = MagicMock()
    nq.redis = MagicMock()
    return LLMNewsAgent(
        news_pipeline=MagicMock(), credibility_engine=MagicMock(), news_queue=nq,
        llm_service=MagicMock(), db_session_factory=SessionLocal,
    )


def _prediction(**kw):
    base = dict(
        source_domain="reuters.com", headline="Bitcoin ETF approved", article_hash="h1",
        severity=Severity.SIGNIFICANT, asset="BTC-USD", direction=Direction.down,
        magnitude_pct_low=2.0, magnitude_pct_high=5.0, confidence=0.8,
        t_min_minutes=5, t_max_minutes=30, rationale="r", trust_score_at_time=0.9,
        created_at=datetime.utcnow(),
    )
    base.update(kw)
    return NewsPrediction(**base)


@pytest.mark.asyncio
async def test_newsprediction_persists_relevance_columns(tmp_path):
    # The bug: these kwargs used to raise (columns absent) -> news never persisted.
    SessionLocal = await _make_db(tmp_path)
    async with SessionLocal() as s:
        p = _prediction(symbol_relevance={"BTCUSDT": 0.8}, matched_keywords={"BTCUSDT": ["bitcoin"]})
        s.add(p)
        await s.commit()
        await s.refresh(p)
        assert p.symbol_relevance == {"BTCUSDT": 0.8}
        assert p.matched_keywords == {"BTCUSDT": ["bitcoin"]}


def test_weighted_relevance_respects_weights():
    text = "Bitcoin ETF approved by BlackRock"
    base, _ = extract_symbol_relevance(text)
    boosted, _ = extract_symbol_relevance(text, weights={
        ("BTCUSDT", "bitcoin"): 3.0, ("BTCUSDT", "etf"): 3.0, ("BTCUSDT", "blackrock"): 3.0,
    })
    assert boosted.get("BTCUSDT", 0.0) >= base.get("BTCUSDT", 0.0) > 0.0


@pytest.mark.asyncio
async def test_keyword_bank_seeds_and_feedback_updates(tmp_path):
    SessionLocal = await _make_db(tmp_path)
    agent = _make_agent(SessionLocal)

    weights = await agent._load_keyword_weights()  # seeds from CRYPTO_KEYWORD_BANK
    assert ("BTCUSDT", "bitcoin") in weights

    async with SessionLocal() as s:
        p = _prediction(matched_keywords={"BTCUSDT": ["bitcoin"]})
        s.add(p)
        await s.commit()
        await s.refresh(p)
        await agent._update_keyword_feedback(s, p, score=1.0)            # good outcome
        await agent._record_severity_calibration(s, p, actual_move_pct=-3.0)
        await s.commit()

    async with SessionLocal() as s:
        row = (await s.execute(
            select(KeywordWeight).where(KeywordWeight.symbol == "BTCUSDT", KeywordWeight.keyword == "bitcoin")
        )).scalar_one()
        assert row.weight > 1.0 and row.hits == 1 and row.correct == 1
        cal = (await s.execute(select(SeverityCalibration))).scalar_one()
        assert cal.count == 1 and cal.sum_realized == pytest.approx(3.0)


@pytest.mark.asyncio
async def test_bad_outcome_lowers_keyword_weight(tmp_path):
    SessionLocal = await _make_db(tmp_path)
    agent = _make_agent(SessionLocal)
    await agent._load_keyword_weights()
    async with SessionLocal() as s:
        p = _prediction(matched_keywords={"BTCUSDT": ["bitcoin"]})
        s.add(p)
        await s.commit()
        await s.refresh(p)
        await agent._update_keyword_feedback(s, p, score=0.0)  # wrong prediction
        await s.commit()
    async with SessionLocal() as s:
        row = (await s.execute(
            select(KeywordWeight).where(KeywordWeight.symbol == "BTCUSDT", KeywordWeight.keyword == "bitcoin")
        )).scalar_one()
        assert row.weight < 1.0 and row.correct == 0
