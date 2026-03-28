import json
import re
from typing import Tuple, Any

from sqlalchemy import select
from structlog import get_logger

from data.news_feed import NewsArticle
from memory.database import SourceTrustScore

log = get_logger("agents.credibility")

BASE_TRUST_SCORES = {
    "reuters.com": 0.95, 
    "bbc.co.uk": 0.93, 
    "bloomberg.com": 0.92,
    "nytimes.com": 0.88, 
    "ft.com": 0.87, 
    "wsj.com": 0.86,
    "coindesk.com": 0.80, 
    "yahoo.com": 0.78, 
    "cointelegraph.com": 0.65,
    "cryptopanic.com": 0.55, 
    "unknown": 0.25
}

FAST_LANE_THRESHOLD = 0.85

class CredibilityEngine:
    def __init__(self, db_session_factory: Any, llm_service: Any):
        self.db_session_factory = db_session_factory
        self.llm_service = llm_service

    async def get_trust_score(self, source_domain: str) -> float:
        async with self.db_session_factory() as session:
            result = await session.execute(
                select(SourceTrustScore).where(SourceTrustScore.source_domain == source_domain)
            )
            score_record = result.scalar_one_or_none()

            if score_record is not None:
                return float(score_record.current_score)
            
            # Not found - Insert new
            base_score = float(BASE_TRUST_SCORES.get(source_domain, 0.25))
            new_record = SourceTrustScore(
                source_domain=source_domain,
                base_score=base_score,
                current_score=base_score,
                total_predictions=0,
                correct_predictions=0
            )
            session.add(new_record)
            # Commit handled by the asynccontextmanager in database.py
            return base_score

    async def check_plausibility(self, article: NewsArticle) -> float:
        prompt = f"""Rate the plausibility of this financial news headline from 0.0 to 1.0.
Respond with a single float only. No explanation.
Headline: {article.headline}
Source: {article.source_domain}"""

        try:
            text = await self.llm_service.generate_text(prompt, tier="haiku", max_tokens=20)
            text = text.strip()
            # Extract just the float in case LLM added extra chars
            match = re.search(r"[-+]?\d*\.\d+|\d+", text)
            
            if match:
                val = float(match.group())
                return float(max(0.0, min(val, 1.0)))
            return 0.5

        except Exception as e:
            log.warning("Plausibility check failed", error=str(e), source=article.source_domain)
            return 0.5

    async def score_article(self, article: NewsArticle) -> Tuple[float, bool]:
        base = await self.get_trust_score(article.source_domain)
        
        if base >= FAST_LANE_THRESHOLD:
            return (base, True)

        plausibility = await self.check_plausibility(article)
        final = (0.7 * base) + (0.3 * plausibility)
        
        return (final, False)

    async def update_trust_score(self, source_domain: str, prediction_score: float) -> None:
        async with self.db_session_factory() as session:
            result = await session.execute(
                select(SourceTrustScore).where(SourceTrustScore.source_domain == source_domain)
            )
            score_record = result.scalar_one_or_none()
            
            if not score_record:
                log.warning("Attempted to update trust score for missing domain", domain=source_domain)
                return
                
            current = float(score_record.current_score)
            new_score = (0.92 * current) + (0.08 * float(prediction_score))
            new_score = max(0.10, min(new_score, 0.97))
            
            score_record.current_score = new_score
            score_record.total_predictions += 1
            
            if prediction_score > 0.0:
                 # Depending on what prediction_score implies, we might increment correctly differently but 
                 # assuming any > 0 indicates a 'correct' mapping or partial correct.
                score_record.correct_predictions += 1
