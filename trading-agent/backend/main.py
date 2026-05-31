import os
# Workstream B: hardware-aware thread budget, set BEFORE torch is imported
# anywhere. Replaces the old hard-pin to 1 thread (a conservative choice for a
# "small CPU") with an auto-tuned count that uses spare physical cores while
# leaving OS/I/O headroom. `HW_AUTO_TUNE=false` or an explicit OMP_NUM_THREADS
# restores the legacy single-thread behavior. Falls back to 1 on any error.
try:
    from backend.core.hardware import apply_startup_threads, summary_line
    _HW_BUDGET = apply_startup_threads()
    print("[Backend] " + summary_line(_HW_BUDGET), flush=True)
except Exception as _hw_e:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    print(f"[Backend] hardware auto-tune skipped ({_hw_e}); using 1 thread.", flush=True)

import asyncio
import ctypes
import multiprocessing
import signal
import sys

# DNS pre-flight + DoH fallback must run BEFORE any aiohttp / urllib sessions
# spin up, otherwise they cache the broken system resolver. Cheap when the
# system DNS works (single parallel probe with a 2.5s budget per host).
try:
    from backend.core.network_check import pre_flight as _dns_pre_flight
    _DNS_SUMMARY = _dns_pre_flight(verbose=True)
    if _DNS_SUMMARY["system_ok"]:
        print("[Backend] DNS pre-flight: all hosts OK via system resolver.", flush=True)
    elif _DNS_SUMMARY["doh_installed"]:
        print(f"[Backend] DNS pre-flight: system resolver FAILED for "
              f"{_DNS_SUMMARY['failed_hosts']}; Cloudflare DoH fallback (1.1.1.1) "
              f"installed for this process.", flush=True)
    else:
        print(f"[Backend] WARNING: system DNS AND Cloudflare DoH both unreachable. "
              f"Failed hosts: {_DNS_SUMMARY['failed_hosts']}. Stock data + LLM calls "
              f"will not work until this is resolved.", flush=True)
except Exception as _e:
    print(f"[Backend] DNS pre-flight skipped due to: {_e}", flush=True)

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
from backend.agents.llm import LLMService
import ccxt
import subprocess

try:
    # Start the watchdog to monitor terminal shutdown in an entirely detached process
    watchdog_script = os.path.join(os.path.dirname(__file__), "watchdog.py")
    subprocess.Popen(
        [sys.executable, watchdog_script],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | 0x08000000 # CREATE_NO_WINDOW
    )
except Exception as e:
    pass

logger = get_logger("main")

# We instantiate models at global scope loosely mapped for signal catching 
_model_instance = None

def run_nn_agent(severe_flag):
    # Setup asyncio event loop for this process
    async def _run():
        import ccxt.async_support as ccxt_async
        
        # Phase 1: the crypto agent now considers the FULL crypto vocabulary the
        # model was trained on (8 symbols, ids 0..7) rather than a 2-symbol
        # subset. Sourced from core.universe so there's a single source of truth.
        # NOTE: the asset chosen for VIEWING on the UI never reaches this list —
        # viewing is purely a frontend concern, so it can never narrow the
        # agent's trading/attention scope.
        from backend.core import universe as _universe
        trading_symbols = list(_universe.CRYPTO_SYMBOLS)
        market_feed = BinanceMarketFeed(symbols=trading_symbols)
        await market_feed.start()
        
        from backend.memory.redis_client import get_redis
        redis_session = await get_redis()
        
        from backend.signals import technical, orderbook
        feature_builder = FeatureVectorBuilder(redis_session, technical, orderbook)
        regime_detector = RegimeDetector()
        
        global _model_instance
        _model_instance = PersistentTradingModel()
        
        # The breaker measures drawdown against ``peak_portfolio_value``, which
        # seeds from this initial value. It MUST match the agent's actual starting
        # cash (INITIAL_USDC_AMOUNT) — otherwise a default 10k peak vs a 1k real
        # book reads as an instant ~90% phantom drawdown and latches the agent
        # into HALTED on cycle one (every trade then rejected "manual reset
        # required"). This was the root cause of "the backend isn't doing much".
        risk_manager = RiskManager(initial_portfolio_value=float(settings.INITIAL_USDC_AMOUNT))

        from backend.execution.defi_engine import UniswapV3Executor, DefiPortfolioTracker, DefiExecutionEngine
        from web3 import Web3
        web3_client = Web3(Web3.HTTPProvider(settings.ARBITRUM_RPC_URL or "https://arb1.arbitrum.io/rpc"))
        
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
            db_session_factory=AsyncSessionLocal,
            paper_mode=settings.PAPER_MODE.lower() != "false",
            risk_manager=risk_manager,
        )

        # Phase 7b: multi-broker registry. Crypto stays on Uniswap; Alpaca (US stocks)
        # and IBKR (LSE leveraged ETPs) register if their credentials / Gateway are
        # available. The crypto agent still routes via exec_engine until the parallel
        # multi-asset loop lands; the registry surfaces what's connectable for /api/brokers.
        from backend.execution.broker_registry import BrokerRegistry
        from backend.execution.alpaca_broker import AlpacaBroker
        from backend.execution.ibkr_broker import IBKRBroker
        broker_registry = BrokerRegistry()
        broker_registry.register("crypto", exec_engine)
        try:
            broker_registry.register("us_stock", AlpacaBroker(
                paper=settings.PAPER_MODE.lower() != "false",
                db_session_factory=AsyncSessionLocal,
            ))
        except Exception as e:
            logger.warning("alpaca_broker_init_failed", error=str(e))
        ibkr_broker = None
        try:
            ibkr_broker = IBKRBroker(db_session_factory=AsyncSessionLocal)
            broker_registry.register("lse_etp", ibkr_broker)
            # Best-effort connect to a running Gateway/TWS. Failure is non-fatal —
            # is_available() will stay False and the registry routes elsewhere.
            try:
                await ibkr_broker.connect()
            except Exception as e:
                logger.warning("ibkr_connect_failed_at_startup", error=str(e))
        except Exception as e:
            logger.warning("ibkr_broker_init_failed", error=str(e))

        avail = {k: bool(b.is_available()) for k, b in broker_registry.all().items()}
        logger.info("broker_registry_initialised", brokers=avail)
        # Publish to Redis so the FastAPI process (which doesn't hold the IB Gateway
        # connection — only one client per clientId) can surface accurate status.
        try:
            import json as _json
            await redis_session.set("brokers:availability", _json.dumps(avail))
        except Exception:
            pass

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
            symbols=trading_symbols
        )

        # Phase 14: macro/derivatives feed populates the macro feature slots
        # (fear_greed, btc_dominance, funding_rate, oi_change) that previously
        # always read neutral defaults — see backend/signals/features.py:217-226.
        from backend.data.macro_feed import MacroFeed
        macro_feed = MacroFeed(redis_session, symbols=trading_symbols)
        asyncio.create_task(macro_feed.run())

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
            
        import subprocess, os
        if os.name == 'nt':
            subprocess.run(["taskkill", "/F", "/IM", "ollama.exe"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            subprocess.run(["taskkill", "/F", "/IM", "ollama app.exe"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            subprocess.run(["pkill", "-f", "ollama"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    signal.signal(signal.SIGTERM, handle_sigterm)
    
    # Run FastAPI in the main process
    try:
        app = create_app()
        uvicorn.run(app, host="0.0.0.0", port=8000, log_config=None)
    except KeyboardInterrupt:
        handle_sigterm(None, None)
