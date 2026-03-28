import asyncio
import ctypes
import multiprocessing
import signal
import sys
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import structlog

from backend.core.config import settings
from backend.core.logger import get_logger
from backend.memory.database import init_db, AsyncSessionLocal
from backend.memory.redis_client import PriorityNewsQueue
from backend.api.routes import router, ws_live_updater
from backend.agents.nn_model import PersistentTradingModel

# Agent imports
from backend.agents.nn_agent import NNTradingAgent
from backend.agents.news_agent import LLMNewsAgent
from backend.agents.credibility import CredibilityEngine
from backend.data.market_feed import BinanceMarketFeed
from backend.data.news_feed import NewsIngestionPipeline, DEFAULT_RSS_FEEDS
from backend.signals.features import FeatureVectorBuilder
from backend.signals.regime import RegimeDetector
from backend.risk.manager import RiskManager
from backend.execution.engine import ExecutionEngine
from backend.execution.kite_chain import KiteChainClient
from backend.agents.llm import LLMService
import ccxt

logger = get_logger("main")

# We instantiate models at global scope loosely mapped for signal catching 
_model_instance = None

def run_nn_agent(severe_flag):
    # Setup asyncio event loop for this process
    async def _run():
        import ccxt.async_support as ccxt_async
        
        market_feed = BinanceMarketFeed(symbols=["BTCUSDT"])
        await market_feed.start()
        
        feature_builder = FeatureVectorBuilder(None, None, None)
        regime_detector = RegimeDetector()
        
        global _model_instance
        _model_instance = PersistentTradingModel()
        
        risk_manager = RiskManager()
        kite_chain = KiteChainClient(
            rpc_url=settings.KITE_CHAIN_RPC_URL,
            private_key=settings.KITE_CHAIN_PRIVATE_KEY,
            agent_address=settings.KITE_AGENT_ADDRESS,
            db_session_factory=AsyncSessionLocal
        )
        
        exchange = ccxt.binance({
            'apiKey': settings.BINANCE_API_KEY,
            'secret': settings.BINANCE_SECRET,
        })
        
        exec_engine = ExecutionEngine(
            exchange=exchange,
            kite_chain=kite_chain,
            db_session_factory=AsyncSessionLocal,
            paper_mode=True
        )
        
        from backend.memory.redis_client import get_redis
        redis_session = await get_redis()
        news_queue = PriorityNewsQueue(redis_session)
        
        agent = NNTradingAgent(
            market_feed=market_feed,
            feature_builder=feature_builder,
            regime_detector=regime_detector,
            model=_model_instance,
            risk_manager=risk_manager,
            execution_engine=exec_engine,
            news_queue=news_queue,
            severe_flag=severe_flag,
            symbols=["BTCUSDT"]
        )
        await agent.run()

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        if _model_instance:
            _model_instance.safe_checkpoint(label="shutdown")

def run_news_agent(severe_flag):
    async def _run():
        from backend.memory.redis_client import get_redis
        redis_session = await get_redis()
        news_queue = PriorityNewsQueue(redis_session)
        
        news_pipeline = NewsIngestionPipeline(rss_urls=DEFAULT_RSS_FEEDS)
        
        llm_service = LLMService(
            provider=settings.AI_PROVIDER,
            anthropic_key=settings.ANTHROPIC_API_KEY,
            gemini_key=settings.GEMINI_API_KEY
        )
        
        credibility_engine = CredibilityEngine(db_session_factory=AsyncSessionLocal, llm_service=llm_service)

        agent = LLMNewsAgent(
            news_pipeline=news_pipeline,
            credibility_engine=credibility_engine,
            news_queue=news_queue,
            llm_service=llm_service,
            db_session_factory=AsyncSessionLocal
        )
        await agent.run()
        
    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        pass

def create_app() -> FastAPI:
    app = FastAPI(title="Trading Agent API")
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"], # Should be constrained in prod
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    app.include_router(router)
    
    @app.on_event("startup")
    async def startup_event():
        await init_db()
        asyncio.create_task(ws_live_updater())
        logger.info("fastapi_startup_complete")
        
    return app

if __name__ == "__main__":
    logger.info("starting_trading_agent_processes")
    
    multiprocessing.set_start_method('spawn')
    
    shared_severe_flag = multiprocessing.Value(ctypes.c_bool, False)
    
    p1 = multiprocessing.Process(target=run_nn_agent, args=(shared_severe_flag,), name="NNTradingAgent")
    p2 = multiprocessing.Process(target=run_news_agent, args=(shared_severe_flag,), name="LLMNewsAgent")
    
    needs_setup = False
    try:
        if settings.needs_setup():
            needs_setup = True
            logger.warning("Setup required -> Suppressing agent processes until .env is configured")
    except Exception:
        needs_setup = True
        
    if not needs_setup:
        p1.start()
        p2.start()

    def handle_sigterm(signum, frame):
        logger.info("graceful_shutdown_initiated")
        if not needs_setup or p1.is_alive():
            p1.terminate()
            p1.join()
        if not needs_setup or p2.is_alive():
            p2.terminate()
            p2.join()
    signal.signal(signal.SIGTERM, handle_sigterm)
    
    # Run FastAPI in the main process
    try:
        app = create_app()
        uvicorn.run(app, host="0.0.0.0", port=8000, log_config=None)
    except KeyboardInterrupt:
        handle_sigterm(None, None)
