import asyncio
import json
import re
import aiohttp
from datetime import datetime, timezone, timedelta
import structlog

from sqlalchemy.ext.asyncio import async_sessionmaker
from sqlalchemy import select

from backend.data.news_feed import NewsIngestionPipeline, NewsArticle
from backend.agents.credibility import CredibilityEngine
from backend.memory.redis_client import PriorityNewsQueue, NewsImpact, HeartbeatClient
from backend.memory.database import NewsPrediction, Severity, Direction, KeywordWeight, SeverityCalibration
from backend.training.backbone import map_asset_to_symbol, extract_symbol_relevance, CRYPTO_KEYWORD_BANK

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
        market_feed=None,
        min_trust_to_analyse: float = 0.40
    ):
        self.news_pipeline = news_pipeline
        self.credibility_engine = credibility_engine
        self.news_queue = news_queue
        self.llm_service = llm_service
        self.db_session_factory = db_session_factory
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
            combined_text = f"{article.headline}\n{article.body[:2000]}"
            kw_weights = await self._load_keyword_weights()
            symbol_relevance, matched_keywords = extract_symbol_relevance(combined_text, weights=kw_weights)
            mapped_symbol = map_asset_to_symbol(data.get("asset"))
            if mapped_symbol:
                # Ensure model-declared asset is always represented in relevance map
                symbol_relevance[mapped_symbol] = max(symbol_relevance.get(mapped_symbol, 0.0), 0.75)
                matched_keywords.setdefault(mapped_symbol, [])
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
                    created_at=datetime.fromisoformat(impact.created_at),
                    symbol_relevance=symbol_relevance if symbol_relevance else None,
                matched_keywords=matched_keywords if matched_keywords else None,
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
            # Re-ping well within the 10s heartbeat TTL (HeartbeatClient.ping uses setex 10s)
            # so the dashboard/health check doesn't falsely flag the news agent as offline.
            await asyncio.sleep(5)

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

    async def _load_keyword_weights(self) -> dict:
        """Load learned (symbol, keyword)->weight; seed from CRYPTO_KEYWORD_BANK on first use."""
        try:
            async with self.db_session_factory() as session:
                rows = (await session.execute(select(KeywordWeight))).scalars().all()
                if not rows:
                    for sym, kws in CRYPTO_KEYWORD_BANK.items():
                        for kw in kws:
                            session.add(KeywordWeight(symbol=sym, keyword=kw.lower(), weight=1.0))
                    await session.commit()
                    rows = (await session.execute(select(KeywordWeight))).scalars().all()
                return {(r.symbol, r.keyword.lower()): float(r.weight) for r in rows}
        except Exception as e:
            logger.warning("keyword_weight_load_failed", error=str(e))
            return {}

    async def _fetch_realized_move(self, symbol: str, start: datetime, end: datetime) -> float | None:
        """Realized % move for a crypto symbol over [start, end] via Binance klines."""
        try:
            start_ms = int(start.replace(tzinfo=timezone.utc).timestamp() * 1000)
            end_ms = int(end.replace(tzinfo=timezone.utc).timestamp() * 1000)
            url = ("https://api.binance.us/api/v3/klines"
                   f"?symbol={symbol}&interval=5m&startTime={start_ms}&endTime={end_ms}&limit=1000")
            timeout = aiohttp.ClientTimeout(total=8)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url) as resp:
                    data = await resp.json()
            if not data or not isinstance(data, list):
                return None
            first_open = float(data[0][1])
            last_close = float(data[-1][4])
            if first_open <= 0:
                return None
            return (last_close - first_open) / first_open * 100.0
        except Exception as e:
            logger.warning("realized_move_fetch_failed", symbol=symbol, error=str(e))
            return None

    async def _update_keyword_feedback(self, session, prediction, score: float) -> None:
        """Negative/positive automated feedback on keywords: good predictions raise the
        keyword's weight, poor ones lower it (toward a 0.05 prune floor)."""
        matched = prediction.matched_keywords or {}
        for sym, kws in matched.items():
            for kw in (kws or []):
                kwl = str(kw).lower()
                row = (await session.execute(
                    select(KeywordWeight).where(KeywordWeight.symbol == sym, KeywordWeight.keyword == kwl)
                )).scalar_one_or_none()
                if row is None:
                    row = KeywordWeight(symbol=sym, keyword=kwl, weight=1.0)
                    session.add(row)
                row.hits = (row.hits or 0) + 1
                if score > 0.5:
                    row.correct = (row.correct or 0) + 1
                new_w = float(row.weight) + 0.1 * ((score - 0.5) * 2.0)
                row.weight = float(max(0.05, min(new_w, 3.0)))

    async def _record_severity_calibration(self, session, prediction, actual_move_pct: float) -> None:
        bucket = prediction.severity.value if hasattr(prediction.severity, "value") else str(prediction.severity)
        row = (await session.execute(
            select(SeverityCalibration).where(SeverityCalibration.severity_bucket == bucket)
        )).scalar_one_or_none()
        if row is None:
            row = SeverityCalibration(severity_bucket=bucket)
            session.add(row)
        predicted_mid = (float(prediction.magnitude_pct_low) + float(prediction.magnitude_pct_high)) / 2.0
        row.sum_predicted = float(row.sum_predicted or 0.0) + predicted_mid
        row.sum_realized = float(row.sum_realized or 0.0) + abs(actual_move_pct)
        row.count = (row.count or 0) + 1

    async def check_prediction_outcomes(self):
        while True:
            try:
                await asyncio.sleep(300)
                scored = []  # (source_domain, score) — trust updates run after commit (separate sessions)
                async with self.db_session_factory() as session:
                    now = datetime.utcnow()

                    stmt = select(NewsPrediction).where(NewsPrediction.outcome_checked == False)
                    result = await session.execute(stmt)
                    predictions = result.scalars().all()

                    for p in predictions:
                        expires_at = p.created_at + timedelta(minutes=max(int(p.t_max_minutes), 1))
                        if now <= expires_at:
                            continue

                        mapped = map_asset_to_symbol(p.asset)
                        actual_move_pct = None
                        if mapped:
                            actual_move_pct = await self._fetch_realized_move(mapped, p.created_at, expires_at)

                        if actual_move_pct is None:
                            # Can't evaluate (non-crypto, or fetch failed) — mark checked to avoid pile-up.
                            p.outcome_checked = True
                            continue

                        score = self.score_news_prediction(p, actual_move_pct)
                        p.actual_move_pct = actual_move_pct
                        p.prediction_score = score
                        p.outcome_checked = True

                        await self._update_keyword_feedback(session, p, score)
                        await self._record_severity_calibration(session, p, actual_move_pct)
                        scored.append((p.source_domain, score))

                    await session.commit()

                for domain, score in scored:
                    try:
                        await self.credibility_engine.update_trust_score(domain, score)
                    except Exception as ce:
                        logger.warning("trust_update_failed", domain=domain, error=str(ce))

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
