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
        
        from backend.execution.defi_engine import UniswapV3Executor, DefiPortfolioTracker, DefiExecutionEngine
        from web3 import Web3
        web3_client = Web3(Web3.HTTPProvider(settings.ARBITRUM_RPC_URL or "https://arb1.arbitrum.io/rpc"))
        
        from backend.memory.redis_client import get_redis
        redis_session = await get_redis()
        
        uniswap = UniswapV3Executor(
            web3=web3_client,
            wallet_address=settings.AGENT_WALLET_ADDRESS or "0x0000000000000000000000000000000000000000",
            private_key=settings.AGENT_PRIVATE_KEY or "0" * 64,
            slippage_tolerance=0.005
        )
        
        portfolio = DefiPortfolioTracker(
            web3=web3_client,
            wallet_address=settings.AGENT_WALLET_ADDRESS or "0x0000000000000000000000000000000000000000",
            redis_client=redis_session
        )
        
        exec_engine = DefiExecutionEngine(
            uniswap=uniswap,
            portfolio=portfolio,
            kite_chain=kite_chain,
            db_session_factory=AsyncSessionLocal,
            paper_mode=False if os.environ.get("PAPER_MODE", "true").lower() == "false" else True
        )
        
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
            gemini_key=settings.GEMINI_API_KEY,
            ollama_model=settings.OLLAMA_MODEL
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

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await init_db()
    asyncio.create_task(ws_live_updater())
    logger.info("fastapi_startup_complete")
    yield
    # Shutdown
    logger.info("fastapi_shutdown_complete")

def create_app() -> FastAPI:
    app = FastAPI(title="Trading Agent API", lifespan=lifespan)
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"], # Should be constrained in prod
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    app.include_router(router)
        
    return app

if __name__ == "__main__":
    logger.info("starting_trading_agent_processes")
    
    # Auto-start Redis if needed and not running on Windows
    try:
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1)
            s.connect(("127.0.0.1", 6379))
    except Exception:
        import subprocess, os
        if os.name == 'nt':
            logger.info("Real Redis not found on port 6379. Attempting to download/run portable Windows Redis...")
            import urllib.request, zipfile
            redis_dir = os.path.join(os.getcwd(), ".redis_win")
            redis_exe = os.path.join(redis_dir, "redis-server.exe")
            
            if not os.path.exists(redis_exe):
                try:
                    os.makedirs(redis_dir, exist_ok=True)
                    redis_url = "https://github.com/microsoftarchive/redis/releases/download/win-3.0.504/Redis-x64-3.0.504.zip"
                    zip_path = os.path.join(redis_dir, "redis.zip")
                    urllib.request.urlretrieve(redis_url, zip_path)
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(redis_dir)
                    os.remove(zip_path)
                except Exception as e:
                    logger.warning(f"Failed to auto-install Redis: {e}. Will continue falling back to FakeRedis.")
            
            if os.path.exists(redis_exe):
                try:
                    creation_flags = getattr(subprocess, 'CREATE_NO_WINDOW', 0x08000000)
                    subprocess.Popen([redis_exe], creationflags=creation_flags, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL)
                    import time
                    time.sleep(1) # Let Redis bind to the port
                except Exception as e:
                    pass

    # Auto-start Ollama if needed and not running
    if "ollama" in settings.AI_PROVIDER.lower() or "hybrid" in settings.AI_PROVIDER.lower():
        try:
            import urllib.request
            urllib.request.urlopen("http://127.0.0.1:11434/api/version", timeout=1)
        except Exception:
            logger.info("Ollama not running. Starting in background...")
            import subprocess, os
            if os.name == 'nt':
                creation_flags = getattr(subprocess, 'CREATE_NO_WINDOW', 0x08000000)
                try:
                    subprocess.Popen(["ollama", "serve"], creationflags=creation_flags, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL)
                except FileNotFoundError:
                    pass
            else:
                try:
                    subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                except FileNotFoundError:
                    pass
            import time
            time.sleep(3)

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
