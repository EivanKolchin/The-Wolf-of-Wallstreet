import asyncio
import json
import re
from datetime import datetime, timezone, timedelta
import structlog

from sqlalchemy.ext.asyncio import async_sessionmaker
from sqlalchemy import select

from backend.data.news_feed import NewsIngestionPipeline, NewsArticle
from backend.agents.credibility import CredibilityEngine
from backend.memory.redis_client import PriorityNewsQueue, NewsImpact, HeartbeatClient
from backend.memory.database import NewsPrediction, Severity, Direction

logger = structlog.get_logger(__name__)

SEVERITY_CLASSIFICATION_PROMPT = """You are a financial severity classifier for a live trading system.
Classify the market impact of this news article.

Article headline: {headline}
Article body (first 500 chars): {body}
Source trust score: {trust_score}

Severity rules:
- NEUTRAL: routine news, confidence < 0.5, magnitude < 1%, or unrelated to markets
- SIGNIFICANT: clear directional catalyst, confidence > 0.6, magnitude 1-10%, 
               no systemic risk, does NOT require immediate position closure
- SEVERE: systemic risk event — exchange collapse, regulatory ban, protocol exploit, 
          stablecoin depeg, magnitude likely > 10%

Respond in raw JSON only. No preamble, no markdown, no explanation:
{{
  "severity": "SIGNIFICANT",
  "asset": "BTC-USD",
  "direction": "down",
  "magnitude_pct_low": 3.0,
  "magnitude_pct_high": 8.0,
  "confidence": 0.78,
  "t_min_minutes": 5,
  "t_max_minutes": 30,
  "rationale": "one sentence"
}}
If NEUTRAL, respond: {{"severity": "NEUTRAL"}}"""

def extract_json(text: str) -> dict:
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return {}

class LLMNewsAgent:
    def __init__(
        self,
        news_pipeline: NewsIngestionPipeline,
        credibility_engine: CredibilityEngine,
        news_queue: PriorityNewsQueue,
        llm_service,
        db_session_factory: async_sessionmaker,
        kite_chain = None,
        market_feed=None,
        min_trust_to_analyse: float = 0.40
    ):
        self.news_pipeline = news_pipeline
        self.credibility_engine = credibility_engine
        self.news_queue = news_queue
        self.llm_service = llm_service
        self.db_session_factory = db_session_factory
        self.kite_chain = kite_chain
        self.market_feed = market_feed
        self.min_trust_to_analyse = min_trust_to_analyse
        self.heartbeat_client = HeartbeatClient(news_queue.redis)

    async def analyse_article(self, article: NewsArticle, trust_score: float) -> NewsImpact | None:
        prompt = SEVERITY_CLASSIFICATION_PROMPT.format(
            headline=article.headline,
            body=article.body[:500],
            trust_score=trust_score
        )

        try:
            text = await self.llm_service.generate_text(prompt, tier="sonnet", max_tokens=300)
            data = extract_json(text)
        except Exception as e:
            logger.error("llm_classification_error", error=str(e), headline=article.headline)
            return None

        severity_str = data.get("severity", "NEUTRAL").upper()
        if severity_str == "NEUTRAL":
            return None

        try:
            direction_str = data.get("direction", "neutral").lower()
            impact = NewsImpact(
                severity=severity_str,
                asset=data.get("asset", "UNKNOWN"),
                direction=direction_str,
                magnitude_pct_low=float(data.get("magnitude_pct_low", 0.0)),
                magnitude_pct_high=float(data.get("magnitude_pct_high", 0.0)),
                confidence=float(data.get("confidence", 0.0)),
                t_min_minutes=int(data.get("t_min_minutes", 0)),
                t_max_minutes=int(data.get("t_max_minutes", 0)),
                rationale=data.get("rationale", ""),
                source_domain=article.source_domain,
                trust_score=trust_score,
                created_at=datetime.utcnow().isoformat()
            )

            async with self.db_session_factory() as session:
                prediction = NewsPrediction(
                    source_domain=impact.source_domain,
                    headline=article.headline,
                    article_hash=article.article_hash,
                    severity=Severity(impact.severity),
                    asset=impact.asset,
                    direction=Direction(impact.direction),
                    magnitude_pct_low=impact.magnitude_pct_low,
                    magnitude_pct_high=impact.magnitude_pct_high,
                    confidence=impact.confidence,
                    t_min_minutes=impact.t_min_minutes,
                    t_max_minutes=impact.t_max_minutes,
                    rationale=impact.rationale,
                    trust_score_at_time=impact.trust_score,
                    created_at=datetime.fromisoformat(impact.created_at)
                )
                session.add(prediction)
                await session.commit()

            return impact

        except Exception as e:
            logger.error("impact_parsing_error", error=str(e), data=data)
            return None

    async def run_pipeline_processor(self):
        article_queue = asyncio.Queue()

        async def _on_article(article: NewsArticle):
            await article_queue.put(article)

        self.news_pipeline.on_article = _on_article
        
        # Start only what we can
        try:
            await self.news_pipeline.start()
        except Exception as e:
            logger.warning("news_pipeline_start_warning", error=str(e), detail="Some feeds may be unavailable. Running in partial mode.")

        while True:
            try:
                article = await article_queue.get()
                
                # Push raw article to redis for the frontend dashboard widget
                try:
                    raw_dict = {
                        "headline": article.headline,
                        "source": article.source_domain,
                        "time": datetime.utcnow().isoformat()
                    }
                    raw_data = json.dumps(raw_dict)
                    
                    await self.news_queue.redis.lpush("recent_raw_news", raw_data)
                    await self.news_queue.redis.ltrim("recent_raw_news", 0, 19)
                    
                    # Bypass FakeRedis multiprocessing limits using a JSON file cache
                    if "FakeRedis" in str(type(self.news_queue.redis)):
                        import os
                        cache_file = os.path.join(os.getcwd(), "raw_news_cache.json")
                        existing = []
                        if os.path.exists(cache_file):
                            try:
                                with open(cache_file, "r") as f:
                                    existing = json.load(f)
                            except: pass
                        existing.insert(0, raw_dict)
                        existing = existing[:20]
                        with open(cache_file, "w") as f:
                            json.dump(existing, f)
                except Exception as e:
                    logger.warning("failed_to_push_raw_news", error=str(e))

                trust_score, is_fast = await self.credibility_engine.score_article(article)
                if trust_score < self.min_trust_to_analyse:
                    continue
                
                # x402 Agent-to-Agent Payment before inference
                if self.kite_chain:
                    # Mock payment to data provider
                    await self.kite_chain.transfer_usdc(to="0x742d35Cc6634C0532925a3b844Bc454e4438f44e", amount=0.01)

                impact = await self.analyse_article(article, trust_score)
                if impact is not None:
                    await self.news_queue.put(impact)
                    logger.info("news_impact_detected", impact=impact.to_json() if hasattr(impact, 'to_json') else str(impact))
            except Exception as e:
                import traceback
                logger.error("news_processor_error", error=str(e), traceback=traceback.format_exc())
            finally:
                await asyncio.sleep(0.1)

    async def heartbeat_loop(self):
        while True:
            try:
                await self.heartbeat_client.ping("llm_news_agent")
            except Exception:
                pass
            await asyncio.sleep(30)

    def score_news_prediction(self, prediction: NewsPrediction, actual_move_pct: float) -> float:
        if prediction.direction == Direction.up and actual_move_pct > 0:
            dir_correct = 1.0
        elif prediction.direction == Direction.down and actual_move_pct < 0:
            dir_correct = 1.0
        elif prediction.direction == Direction.neutral and abs(actual_move_pct) < 1.0:
            dir_correct = 1.0
        else:
            dir_correct = 0.0

        in_range = 1.0 if prediction.magnitude_pct_low <= abs(actual_move_pct) <= prediction.magnitude_pct_high else 0.5
        
        return (dir_correct * 0.7) + (in_range * 0.3)

    async def check_prediction_outcomes(self):
        while True:
            try:
                await asyncio.sleep(300)
                async with self.db_session_factory() as session:
                    now = datetime.utcnow()
                    
                    stmt = select(NewsPrediction).where(NewsPrediction.outcome_checked == False)
                    result = await session.execute(stmt)
                    predictions = result.scalars().all()

                    for p in predictions:
                        expires_at = p.created_at + timedelta(minutes=p.t_max_minutes)
                        if now > expires_at:
                            actual_move_pct = 0.0
                            if self.market_feed:
                                pass # Implement logic to check feed cache real moves

                            score = self.score_news_prediction(p, actual_move_pct)
                            p.actual_move_pct = actual_move_pct
                            p.prediction_score = score
                            p.outcome_checked = True

                            await self.credibility_engine.update_trust_score(p.source_domain, score)

                    await session.commit()

            except Exception as e:
                logger.error("prediction_outcome_check_error", error=str(e))

    async def run(self) -> None:
        logger.info("llm_news_agent_starting")
        try:
            await asyncio.gather(
                self.run_pipeline_processor(),
                self.check_prediction_outcomes(),
                self.heartbeat_loop()
            )
        except Exception as e:
            logger.error("llm_news_agent_crash", error=str(e))
