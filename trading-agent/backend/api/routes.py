import asyncio
import json
from typing import List, Optional, Dict, Any
from datetime import datetime

from fastapi import APIRouter, Depends, Header, HTTPException, WebSocket, WebSocketDisconnect, Body
from pydantic import BaseModel
from sqlalchemy import select, desc, func

from backend.memory.database import AsyncSessionLocal as async_session_maker, Trade, NewsPrediction, AgentEvent, TradeStatus
from backend.memory.redis_client import FeatureCache, HeartbeatClient, get_redis
from backend.core.config import settings
from backend.api.market_routes import router as market_router
from backend.api.risk_routes import router as risk_router
from backend.api.stats_routes import router as stats_router
import structlog
import os
import signal
import subprocess
import threading
import time

logger = structlog.get_logger(__name__)
router = APIRouter()
router.include_router(market_router)
router.include_router(risk_router)
router.include_router(stats_router)

SENSITIVE_SETUP_KEYS = {
    "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY",
    "AGENT_PRIVATE_KEY",
    "ALPACA_API_KEY",
    "ALPACA_SECRET_KEY",
    "X_API_KEY",
    "X_API_SECRET",
    "X_ACCESS_TOKEN",
    "X_ACCESS_TOKEN_SECRET",
    "TELEGRAM_API_ID",
    "TELEGRAM_API_HASH",
    "FINNHUB_API_KEY",
    "TWELVEDATA_API_KEY",
}

SETUP_ALLOWED_KEYS = {
    "AI_PROVIDER",
    "OLLAMA_MODEL",
    "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY",
    "PAPER_MODE",
    "ARBITRUM_RPC_URL",
    "AGENT_WALLET_ADDRESS",
    "AGENT_PRIVATE_KEY",
    "ALPACA_API_KEY",
    "ALPACA_SECRET_KEY",
    "X_API_KEY",
    "X_API_SECRET",
    "X_ACCESS_TOKEN",
    "X_ACCESS_TOKEN_SECRET",
    "TELEGRAM_API_ID",
    "TELEGRAM_API_HASH",
    # Advanced risk options
    "RISK_MAX_DRAWDOWN_PCT",
    "RISK_MAX_DAILY_LOSS_PCT",
    "RISK_MAX_POSITION_PCT",
    "RISK_MIN_CONFIDENCE",
    "RISK_CVAR_LIMIT_PCT",
    "NN_KELLY_FRACTION",
    # Stock brokers + data providers
    "IBKR_HOST",
    "IBKR_PORT",
    "IBKR_CLIENT_ID",
    "IBKR_ACCOUNT_ID",
    "FINNHUB_API_KEY",
    "TWELVEDATA_API_KEY",
    # Funding: wallet connect + Google-Pay on-ramp (Workstreams D/E). These are
    # publishable values (safe in the frontend), served by /api/wallet/config.
    "WALLETCONNECT_PROJECT_ID",
    "ONRAMP_PROVIDER",
    "RAMP_HOST_API_KEY",
    "ONRAMP_DEFAULT_PAYMENT_METHOD",
}


def _require_admin(x_admin_key: str = Header(default=None)) -> str:
    if x_admin_key != settings.ADMIN_KEY:
        raise HTTPException(status_code=401, detail="Invalid admin key")
    return x_admin_key


def _sanitize_env_value(value: Any) -> str:
    txt = str(value)
    return txt.replace("\n", "").replace("\r", "")


def _redact_secret(value: str) -> str:
    if not value:
        return ""
    if len(value) <= 8:
        return "*" * len(value)
    return f"{value[:2]}{'*' * (len(value) - 6)}{value[-4:]}"

pull_state = {
    "status": "idle",
    "model": "",
    "pct": 0.0,
    "total_mb": 0.0,
    "comp_mb": 0.0,
    "speed_mb": 0.0,
    "rem_time": 0.0,
    "error_msg": "",
    "ollama_status": ""
}

def _ollama_model_name_matches(installed: str, target: str) -> bool:
    i = (installed or "").strip().lower()
    t = (target or "").strip().lower()
    if not i or not t:
        return False
    if i == t:
        return True
    if ":" not in t and i == f"{t}:latest":
        return True
    if t.endswith(":latest") and i == t[:-7]:
        return True
    # Handles richer variants such as llama3.2:1b-instruct-* for target llama3.2:1b
    if i.startswith(f"{t}-") or i.startswith(f"{t}:"):
        return True
    return False

def ollama_background_task(target_ollama_model, install_ollama=False):
    import urllib.request
    import shutil
    try:
        ollama_cmd = "ollama"
        if install_ollama:
            pull_state["status"] = "installing_ollama"
            if os.name == 'nt':
                setup_path = "OllamaSetup.exe"
                url = "https://ollama.com/download/OllamaSetup.exe"
                
                # Fetch remote total size
                try:
                    req_head = urllib.request.Request(url, method="HEAD")
                    with urllib.request.urlopen(req_head) as response:
                        total_size = int(response.headers.get("Content-Length", 0))
                except Exception:
                    total_size = 0
                
                downloaded_size = 0
                if os.path.exists(setup_path):
                    downloaded_size = os.path.getsize(setup_path)

                if downloaded_size != total_size or total_size == 0:
                    if downloaded_size > total_size and total_size > 0:
                        os.remove(setup_path)
                        downloaded_size = 0
                    
                    req_dl = urllib.request.Request(url)
                    if downloaded_size > 0:
                        req_dl.add_header("Range", f"bytes={downloaded_size}-")
                    
                    start_time = time.time()
                    mode = "ab" if downloaded_size > 0 else "wb"
                    pull_state["model"] = "OllamaSetup.exe"
                    
                    try:
                        with urllib.request.urlopen(req_dl) as response, open(setup_path, mode) as out_file:
                            block_size = 32768
                            start_down = downloaded_size
                            while True:
                                buffer = response.read(block_size)
                                if not buffer:
                                    break
                                downloaded_size += len(buffer)
                                out_file.write(buffer)
                                
                                # Update Stats
                                if total_size > 0:
                                    pct = downloaded_size / total_size
                                    now = time.time()
                                    elapsed = now - start_time
                                    down_session = downloaded_size - start_down
                                    speed = down_session / elapsed if elapsed > 0 else 0
                                    rem_time = (total_size - downloaded_size) / speed if speed > 0 else 0
                                    
                                    pull_state["pct"] = pct
                                    pull_state["total_mb"] = total_size / (1024*1024)
                                    pull_state["comp_mb"] = downloaded_size / (1024*1024)
                                    pull_state["speed_mb"] = speed / (1024*1024)
                                    pull_state["rem_time"] = rem_time
                    except Exception as e:
                        pull_state["status"] = "error"
                        pull_state["error_msg"] = f"Download failed: {e}"
                        return
                
                # After download is done
                pull_state["status"] = "installing_ollama"
                pull_state["comp_mb"] = pull_state["total_mb"]
                pull_state["pct"] = 1.0
                
                install_res = subprocess.run(f"{setup_path} /VERYSILENT /NORESTART /SUPPRESSMSGBOXES", shell=True)
                
                # If installation failed, it likely got corrupted. Delete & abort.
                if install_res.returncode != 0:
                    pull_state["status"] = "error"
                    pull_state["error_msg"] = "Corrupted download detected. Please try again."
                    if os.path.exists(setup_path):
                        os.remove(setup_path)
                    return
                
                if os.path.exists(setup_path):
                    try:
                        os.remove(setup_path)
                    except:
                        pass
                
                local_app_data = os.environ.get("LOCALAPPDATA", "")
                ollama_dir = os.path.join(local_app_data, "Programs", "Ollama")
                os.environ["PATH"] += os.pathsep + ollama_dir
                
                # Directly specify the binary to bypass "command not found" PATH sync problems
                ollama_bin = os.path.join(ollama_dir, "ollama.exe")
                if os.path.exists(ollama_bin):
                    ollama_cmd = ollama_bin

                # Kill the system tray app if the installer started it (Optional, but highly helpful if user doesn't want the visual app)
                try:
                    subprocess.run(["taskkill", "/F", "/IM", "ollama app.exe"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                except Exception:
                    pass
            else:
                subprocess.run("curl -fsSL https://ollama.com/install.sh | sh", shell=True)
            
            pull_state["status"] = "starting_ollama"
            
            # Start the background CLI server
            if os.name == 'nt':
                creation_flags = getattr(subprocess, 'CREATE_NO_WINDOW', 0x08000000)
                # To prevent child process from opening console windows, we set CREATE_NO_WINDOW
                try:
                    subprocess.Popen([ollama_cmd, "serve"], creationflags=creation_flags, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL)
                except FileNotFoundError:
                    subprocess.Popen(["ollama", "serve"], creationflags=creation_flags, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL)
            else:
                subprocess.Popen([ollama_cmd, "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(5)
        
        # Ensure the server is actually running even if we didn't just install it natively
        if not install_ollama:
            try:
                urllib.request.urlopen("http://127.0.0.1:11434/api/version", timeout=1)
            except Exception:
                if os.name == 'nt':
                    local_app_data = os.environ.get("LOCALAPPDATA", "")
                    ollama_dir = os.path.join(local_app_data, "Programs", "Ollama")
                    os.environ["PATH"] += os.pathsep + ollama_dir
                    ollama_bin = os.path.join(ollama_dir, "ollama.exe")
                    if os.path.exists(ollama_bin):
                        ollama_cmd = ollama_bin
                    
                    creation_flags = getattr(subprocess, 'CREATE_NO_WINDOW', 0x08000000)
                    try:
                        subprocess.Popen([ollama_cmd, "serve"], creationflags=creation_flags, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL)
                    except FileNotFoundError:
                        subprocess.Popen(["ollama", "serve"], creationflags=creation_flags, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL)
                else:
                    subprocess.Popen([ollama_cmd, "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                time.sleep(3)

        pull_state["status"] = "pulling_model"
        pull_state["model"] = target_ollama_model
        
        url = "http://127.0.0.1:11434/api/pull"
        data = json.dumps({"name": target_ollama_model}).encode('utf-8')
        req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
        
        start_time = time.time()
        with urllib.request.urlopen(req) as response:
            for line in response:
                if line:
                    info = json.loads(line.decode('utf-8'))
                    
                    if "status" in info:
                        pull_state["ollama_status"] = info["status"]

                    if "total" in info and "completed" in info:
                        total = info["total"]
                        completed = info["completed"]
                        if total > 0:
                            pct = completed / total
                            now = time.time()
                            elapsed = now - start_time
                            speed = completed / elapsed if elapsed > 0 else 0
                            rem_time = (total - completed) / speed if speed > 0 else 0
                            
                            pull_state["pct"] = pct
                            pull_state["total_mb"] = total / (1024*1024)
                            pull_state["comp_mb"] = completed / (1024*1024)
                            pull_state["speed_mb"] = speed / (1024*1024)
                            pull_state["rem_time"] = rem_time
    except Exception as e:
        pull_state["status"] = "error"
        pull_state["error_msg"] = str(e)
        logger.error(f"Error pulling model: {e}")
        return

    pull_state["status"] = "done"
    time.sleep(2)
    os.kill(os.getpid(), signal.SIGTERM)

@router.get("/api/setup/ollama-progress")
async def get_ollama_progress(_: str = Depends(_require_admin)):
    return pull_state


class HealthResponse(BaseModel):
    status: str
    nn_alive: bool
    news_alive: bool
    model_trade_count: int

@router.get("/api/agent/status")
async def get_agent_status():
    redis_client = await get_redis()
    status_str = await redis_client.get("agent_frontend_status")
    
    if not status_str:
        return {
            "is_halted": False,
            "buffer_current": 0,
            "buffer_required": 60,
            "cycle_interval": 5.0,
            "status_text": "Starting up agent core..."
        }
        
    try:
        data = json.loads(status_str)
        curr = data.get("buffer_current", 0)
        req = data.get("buffer_required", 60)
        halted = data.get("is_halted", False)
        
        status_text = "Analyzing Live Action..."
        if halted:
            status_text = "Trading Force Halted (Manual Stop)"
        elif not data.get("has_market_data", False):
            status_text = f"Awaiting 5m Kline closure (Fetching initial market data)..."
        elif curr < req:
            status_text = f"Warming up neural pathways ({curr}/{req} sequences)..."
            
        return {
            "is_halted": halted,
            "buffer_current": curr,
            "buffer_required": req,
            "cycle_interval": data.get("cycle_interval", 5.0),
            "started_at": data.get("started_at", 0),
            "has_market_data": data.get("has_market_data", False),
            "status_text": status_text
        }
    except Exception as e:
        logger.error("error_parsing_agent_status", error=str(e))
        return {
            "is_halted": False,
            "buffer_current": 0,
            "buffer_required": 60,
            "cycle_interval": 5.0,
            "status_text": "Signal Interrupted..."
        }

class StopResumeRequest(BaseModel):
    halt: bool

@router.post("/api/agent/stop")
async def toggle_agent_stop(req: StopResumeRequest):
    redis_client = await get_redis()
    val = "true" if req.halt else "false"
    await redis_client.set("agent_force_stopped", val)
    return {"status": "success", "is_halted": req.halt}

@router.get("/api/portfolio")
async def get_portfolio(symbol: Optional[str] = None):
    """Phase 1 bug fix: accept ?symbol= so the agent thought reflects the
    asset currently being VIEWED on the dashboard. Falls back to BTCUSDT
    for backwards-compatibility with callers that don't pass a symbol."""
    redis_client = await get_redis()
    portfolio_live_state = {"unrealized_pnl": 0.0, "total_value_locked": 0.0, "positions": []}

    # Show simulated paper positions only in paper mode.
    # When PAPER_MODE is false (live trading), hide any historical paper positions.
    if settings.PAPER_MODE.lower() == "true":
        live_state_str = await redis_client.get("portfolio:live_state")
        if live_state_str:
            try:
                portfolio_live_state = json.loads(live_state_str)
            except Exception as e:
                logger.error("error_parsing_live_state", error=str(e))

    agent_thought = "Initializing CNN market analysis..."
    thought_symbol = (symbol or "BTCUSDT").upper()
    predictions_str = await redis_client.get(f"agent_visual_predictions:{thought_symbol}")
    if not predictions_str:
        predictions_str = await redis_client.get("agent_visual_predictions")
    if predictions_str:
        try:
            preds_data = json.loads(predictions_str)
            agent_thought = preds_data.get("thought", agent_thought)
        except:
            pass

    async with async_session_maker() as session:
        initial_usdc = settings.INITIAL_USDC_AMOUNT

        result = await session.execute(select(func.sum(Trade.pnl_usd)).where(Trade.status == TradeStatus.closed))
        realized_pnl = result.scalar() or 0.0

        result_open = await session.execute(select(func.sum(Trade.size_usd)).where(Trade.status == TradeStatus.open))
        locked_cash = result_open.scalar() or 0.0

        available_cash = initial_usdc + realized_pnl - locked_cash
        if available_cash < 0:
            available_cash = 0.0

    # Workstream F: in LIVE mode reflect the REAL on-chain Arbitrum balance so
    # deposited funds (Google-Pay on-ramp / MetaMask) are visible. Best-effort +
    # cached; paper mode keeps the simulated INITIAL_USDC_AMOUNT untouched.
    if str(settings.PAPER_MODE).lower() != "true":
        onchain_usd = await _read_onchain_balance_usd(redis_client)
        if onchain_usd is not None:
            initial_usdc = onchain_usd
            available_cash = max(0.0, onchain_usd - locked_cash)

    total_portfolio_value = available_cash + locked_cash + portfolio_live_state.get("unrealized_pnl", 0.0)

    return {
        "initial_usdc": initial_usdc,
        "available_cash": available_cash,
        "locked_cash": locked_cash,
        "realized_pnl": realized_pnl,
        "unrealized_pnl": portfolio_live_state.get("unrealized_pnl", 0.0),
        "total_value": total_portfolio_value,
        "live_positions": portfolio_live_state.get("positions", []),
        "agent_thought": agent_thought
    }


@router.get("/api/predictions/{symbol}")
async def get_predictions(symbol: str):
    """Phase 1 bug fix: per-symbol NN visual-prediction payload (purple band,
    median line, buy/sell targets, agent thought) so the frontend can load
    the correct asset INSTANTLY on a symbol switch instead of waiting up to
    ~5s for the next WS broadcast tick.

    Returns the per-symbol Redis key written by nn_agent._render_prediction_chart,
    with the legacy global key as a fallback."""
    redis_client = await get_redis()
    sym = (symbol or "").upper()
    payload_str = await redis_client.get(f"agent_visual_predictions:{sym}")
    if not payload_str:
        payload_str = await redis_client.get("agent_visual_predictions")
    if not payload_str:
        return {"symbol": sym, "predictions": [], "thought": None, "warming_up": True}
    try:
        return json.loads(payload_str)
    except Exception as e:
        logger.error("predictions_payload_parse_error", error=str(e), symbol=sym)
        return {"symbol": sym, "predictions": [], "thought": None, "error": "parse_failed"}

@router.get("/api/setup/status")
async def get_setup_status(_: str = Depends(_require_admin)):
    return {
        "needs_setup": settings.needs_setup(),
        "missing_integrations": settings.missing_integration_status(),
        "ai_provider": settings.AI_PROVIDER,
        "anthropic": not (not settings.ANTHROPIC_API_KEY or "your_" in settings.ANTHROPIC_API_KEY.lower()),
        "gemini": not (not settings.GEMINI_API_KEY or "your_" in settings.GEMINI_API_KEY.lower()),
        "arbitrum": not (not settings.ARBITRUM_RPC_URL or "your_" in settings.ARBITRUM_RPC_URL.lower()),
        "agent_wallet": not (not settings.AGENT_WALLET_ADDRESS or "0x000" in settings.AGENT_WALLET_ADDRESS),
        "agent_pk": not (not settings.AGENT_PRIVATE_KEY or "your_" in settings.AGENT_PRIVATE_KEY.lower() or "0" * 64 in settings.AGENT_PRIVATE_KEY),
        "alpaca": not (not settings.ALPACA_API_KEY or "your_" in settings.ALPACA_API_KEY.lower()),
        "x_api_key": not (not settings.X_API_KEY or "your_" in settings.X_API_KEY.lower()),
        "telegram": not (not settings.TELEGRAM_API_ID or "your_" in settings.TELEGRAM_API_ID.lower())
    }


def _clean(v: str) -> str:
    v = (v or "").strip()
    return "" if (not v or "your_" in v.lower() or v.startswith("0x000")) else v


async def _read_onchain_balance_usd(redis_client) -> Optional[float]:
    """Workstream F: cached real on-chain portfolio value (Arbitrum). Returns None
    (caller falls back to the simulated number) when no wallet is configured or
    the RPC read fails. Cached 30s in Redis to avoid per-request RPC spam."""
    try:
        cached = await redis_client.get("wallet:onchain_usd")
        if cached is not None:
            return float(cached)
    except Exception:
        pass
    addr = _clean(getattr(settings, "AGENT_WALLET_ADDRESS", ""))
    if not addr:
        return None
    try:
        from web3 import Web3
        from backend.execution.defi_engine import DefiPortfolioTracker
        web3 = Web3(Web3.HTTPProvider(settings.ARBITRUM_RPC_URL or "https://arb1.arbitrum.io/rpc"))
        tracker = DefiPortfolioTracker(web3=web3, wallet_address=addr, redis_client=redis_client)
        val = float(await tracker.get_portfolio_value_usd())
        try:
            await redis_client.setex("wallet:onchain_usd", 30, str(val))
        except Exception:
            pass
        return val
    except Exception as e:
        logger.warning("onchain_balance_read_failed", error=str(e))
        return None


@router.get("/api/wallet/config")
async def get_wallet_config():
    """Public, non-sensitive wallet/funding config for the frontend (Workstreams
    D/E). Drives: the WalletConnect connect-QR, the deposit-address QR, and the
    Google-Pay on-ramp widget. Only publishable values are exposed here."""
    addr = _clean(getattr(settings, "AGENT_WALLET_ADDRESS", ""))
    chain_id = int(getattr(settings, "DEPOSIT_CHAIN_ID", 42161) or 42161)
    asset = str(getattr(settings, "ONRAMP_CRYPTO_ASSET", "ARBITRUM_USDC"))
    # EIP-681 URI so scanning the deposit QR in a mobile wallet prefills a send.
    deposit_uri = f"ethereum:{addr}@{chain_id}" if addr else ""
    return {
        "deposit_address": addr,
        "chain_id": chain_id,
        "deposit_uri": deposit_uri,
        "onramp_asset": asset,
        "walletconnect_project_id": _clean(getattr(settings, "WALLETCONNECT_PROJECT_ID", "")),
        "onramp": {
            "provider": str(getattr(settings, "ONRAMP_PROVIDER", "ramp")),
            "ramp_host_api_key": _clean(getattr(settings, "RAMP_HOST_API_KEY", "")),
            "default_payment_method": str(getattr(settings, "ONRAMP_DEFAULT_PAYMENT_METHOD", "")),
            "enabled": bool(_clean(getattr(settings, "RAMP_HOST_API_KEY", "")) and addr),
        },
        "connect_enabled": bool(_clean(getattr(settings, "WALLETCONNECT_PROJECT_ID", ""))),
        "paper_mode": str(getattr(settings, "PAPER_MODE", "true")).lower() == "true",
    }

@router.get("/api/llm/status")
async def get_llm_status():
    import os, json, time
    state_file = os.path.join(os.getcwd(), "llm_state.json")
    if os.path.exists(state_file):
        try:
            with open(state_file, "r") as f:
                state = json.load(f)
                if state.get("is_overloaded", False):
                    elapsed = time.time() - state.get("downgrade_time", 0)
                    rem = int(max(0, 120 - elapsed))
                    state["time_remaining"] = rem
                else:
                    state["time_remaining"] = 0
                return state
        except: pass
    return {"is_overloaded": False, "time_remaining": 0}

@router.post("/api/llm/force-revert")
async def force_llm_revert(_: str = Depends(_require_admin)):
    import os, json
    state_file = os.path.join(os.getcwd(), "llm_state.json")
    if os.path.exists(state_file):
        try:
            with open(state_file, "r") as f:
                state = json.load(f)
            state["force_revert"] = True
            with open(state_file, "w") as f:
                json.dump(state, f)
            return {"status": "success", "message": "Manual revert triggered."}
        except: pass
    return {"status": "error"}

@router.get("/api/setup/config")
async def get_setup_config(reveal: bool = False, _: str = Depends(_require_admin)):
    """G5: with ``?reveal=1`` (admin-gated) return UNREDACTED secret values so the
    local Settings UI can show them behind a password-style show/hide toggle.
    Without it, secrets stay redacted as before. Also returns every raw key found
    in the .env (``_extra``) so nothing is hidden from the owner."""
    import os
    import dotenv
    from backend.core.config import ENV_PATH

    # Read fresh from the .env file directly so we don't rely on cached settings
    env_vars = dotenv.dotenv_values(str(ENV_PATH)) if os.path.exists(str(ENV_PATH)) else {}

    payload = {
        "AI_PROVIDER": env_vars.get("AI_PROVIDER", "gemini"),
        "OLLAMA_MODEL": env_vars.get("OLLAMA_MODEL", "llama3"),
        "ANTHROPIC_API_KEY": env_vars.get("ANTHROPIC_API_KEY", ""),
        "GEMINI_API_KEY": env_vars.get("GEMINI_API_KEY", ""),
        "PAPER_MODE": env_vars.get("PAPER_MODE", "true"),
        "ARBITRUM_RPC_URL": env_vars.get("ARBITRUM_RPC_URL", ""),
        "AGENT_WALLET_ADDRESS": env_vars.get("AGENT_WALLET_ADDRESS", ""),
        "AGENT_PRIVATE_KEY": env_vars.get("AGENT_PRIVATE_KEY", ""),
        "ALPACA_API_KEY": env_vars.get("ALPACA_API_KEY", ""),
        "ALPACA_SECRET_KEY": env_vars.get("ALPACA_SECRET_KEY", ""),
        "X_API_KEY": env_vars.get("X_API_KEY", ""),
        "X_API_SECRET": env_vars.get("X_API_SECRET", ""),
        "X_ACCESS_TOKEN": env_vars.get("X_ACCESS_TOKEN", ""),
        "X_ACCESS_TOKEN_SECRET": env_vars.get("X_ACCESS_TOKEN_SECRET", ""),
        "TELEGRAM_API_ID": env_vars.get("TELEGRAM_API_ID", ""),
        "TELEGRAM_API_HASH": env_vars.get("TELEGRAM_API_HASH", ""),
        "RISK_MAX_DRAWDOWN_PCT": env_vars.get("RISK_MAX_DRAWDOWN_PCT", "15.0"),
        "RISK_MAX_DAILY_LOSS_PCT": env_vars.get("RISK_MAX_DAILY_LOSS_PCT", "5.0"),
        "RISK_MAX_POSITION_PCT": env_vars.get("RISK_MAX_POSITION_PCT", "20.0"),
        "RISK_MIN_CONFIDENCE": env_vars.get("RISK_MIN_CONFIDENCE", "0.0"),
        "RISK_CVAR_LIMIT_PCT": env_vars.get("RISK_CVAR_LIMIT_PCT", "10.0"),
        "NN_KELLY_FRACTION": env_vars.get("NN_KELLY_FRACTION", "0.5"),
        "IBKR_HOST": env_vars.get("IBKR_HOST", "127.0.0.1"),
        "IBKR_PORT": env_vars.get("IBKR_PORT", "4002"),
        "IBKR_CLIENT_ID": env_vars.get("IBKR_CLIENT_ID", "11"),
        "IBKR_ACCOUNT_ID": env_vars.get("IBKR_ACCOUNT_ID", ""),
        "FINNHUB_API_KEY": env_vars.get("FINNHUB_API_KEY", ""),
        "TWELVEDATA_API_KEY": env_vars.get("TWELVEDATA_API_KEY", ""),
    }
    for key in SENSITIVE_SETUP_KEYS:
        payload[f"{key}_IS_SET"] = bool(payload.get(key, ""))
        if not reveal:
            payload[key] = _redact_secret(str(payload.get(key, "")))
    # Surface any other keys present in the .env so the owner sees everything
    # (e.g. the new on-ramp / WalletConnect keys). Redacted unless reveal=1.
    extra = {}
    for k, v in (env_vars or {}).items():
        if k in payload:
            continue
        is_secret = any(tok in k.upper() for tok in ("KEY", "SECRET", "TOKEN", "PASSWORD", "PRIVATE"))
        extra[k] = str(v) if (reveal or not is_secret) else _redact_secret(str(v))
    payload["_extra"] = extra
    payload["_revealed"] = bool(reveal)
    return payload

class SetupRequest(BaseModel):
    ai_provider: str = ""
    anthropic_api_key: str = ""
    gemini_api_key: str = ""
    arbitrum_rpc_url: str = ""
    agent_wallet_address: str = ""
    agent_private_key: str = ""
    alpaca_api_key: str = ""
    alpaca_secret: str = ""
    x_api_key: str = ""
    telegram_api_id: str = ""
    telegram_api_hash: str = ""

@router.post("/api/setup/save")
async def save_setup(req: Dict[str, Any] = Body(...), _: str = Depends(_require_admin)):
    import os
    import signal
    import urllib.request
    from backend.core.config import ENV_PATH
    env_path = str(ENV_PATH)
    
    req_dict: Dict[str, str] = {}
    for k, v in req.items():
        if v is None:
            continue
        key = str(k).upper()
        if key not in SETUP_ALLOWED_KEYS:
            continue
        req_dict[key] = _sanitize_env_value(v)

    if not req_dict:
        return {"status": "ignored", "message": "No allowed settings provided", "installing_model": False}

    if os.path.exists(env_path):
        with open(env_path, "r") as f:
            lines = f.readlines()
            
        settings_seen = set()
        
        with open(env_path, "w") as f:
            for line in lines:
                written = False
                for k, v in req_dict.items():
                    if line.startswith(f"{k}="):
                        f.write(f"{k}={v}\n")
                        settings_seen.add(k)
                        written = True
                        break
                if not written:
                    f.write(line)

            # Append missing lines
            for k, v in req_dict.items():
                if k not in settings_seen:
                    f.write(f"{k}={v}\n")
    else:
        # Create new env file if it doesn't exist
        with open(env_path, "w") as f:
            for k, v in req_dict.items():
                f.write(f"{k}={v}\n")

    # Check if we need to pull Ollama model
    import shutil
    target_ollama_model = req_dict.get("OLLAMA_MODEL")
    provider = req_dict.get("AI_PROVIDER", "gemini").lower()
    
    installing_model = False
    msg = "Applying configuration and restarting..."

    if target_ollama_model and ("ollama" in provider or "hybrid" in provider):
        ollama_installed = shutil.which("ollama") is not None
        
        if not ollama_installed and os.name == 'nt':
            # Check if it was secretly installed but just not in PATH
            ollama_dir = os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs", "Ollama")
            if os.path.exists(os.path.join(ollama_dir, "ollama.exe")):
                os.environ["PATH"] += os.pathsep + ollama_dir
                ollama_installed = True
        
        if not ollama_installed:
            installing_model = True
            msg = f"Applying config. Installing Ollama & {target_ollama_model} in the background..."
            threading.Thread(target=ollama_background_task, args=(target_ollama_model, True)).start()
        else:
            try:
                # Check directly via the local API instead of CLI to prevent the Windows GUI tray app from triggering
                req = urllib.request.Request("http://127.0.0.1:11434/api/tags")
                model_exists = False

                tags_reachable = False
            
                try:
                    with urllib.request.urlopen(req, timeout=5) as response:
                        tags_reachable = True
                        tags_data = json.loads(response.read().decode('utf-8'))
                        for model_obj in tags_data.get("models", []):
                            model_name = model_obj.get("name", "")
                            if _ollama_model_name_matches(model_name, target_ollama_model):
                                model_exists = True
                                break
                except Exception:
                    # If API is unreachable, don't assume model missing to avoid accidental re-download.
                    pass

                if tags_reachable and not model_exists:
                    installing_model = True
                    msg = f"Applying config. Downloading {target_ollama_model} in the background..."
                    threading.Thread(target=ollama_background_task, args=(target_ollama_model, False)).start()
                elif not tags_reachable:
                    logger.warning("Ollama tags endpoint unreachable during setup save; skipping eager model pull check.")
                else:
                    logger.info(f"Ollama model {target_ollama_model} is already installed locally. Skipping download.")
            except Exception as e:
                logger.error(f"Failed to check ollama model: {e}")

    logger.info("setup_config_updated", updated_keys=sorted(list(req_dict.keys())))

    # Trigger an orchestrated restart only if we're not waiting for a download background task
    if not installing_model:
        def restart():
            import sys
            os.execv(sys.executable, ['python'] + sys.argv)
        import asyncio
        asyncio.get_event_loop().call_later(1.0, restart)
        
    return {"status": "saved", "message": msg, "installing_model": installing_model}

@router.get("/api/health", response_model=HealthResponse)
async def get_health():
    redis = await get_redis()
    hb = HeartbeatClient(redis)
    nn_alive = await hb.check_alive("nn_trading_agent")
    news_alive = await hb.check_alive("llm_news_agent")
    
    trade_count = 0
    # In a real app we might fetch model_trade_count from DB model_checkpoints
    
    return HealthResponse(
        status="ok",
        nn_alive=nn_alive,
        news_alive=news_alive,
        model_trade_count=trade_count
    )

@router.get("/api/positions")
async def get_positions():
    async with async_session_maker() as session:
        stmt = select(Trade).where(Trade.status == TradeStatus.open)
        result = await session.execute(stmt)
        trades = result.scalars().all()
        # In a real app you'd compute unrealised PnL against current price
        return trades

@router.get("/api/trades")
async def get_trades(limit: int = 50, offset: int = 0):
    async with async_session_maker() as session:
        stmt = select(Trade).order_by(desc(Trade.opened_at)).limit(limit).offset(offset)
        result = await session.execute(stmt)
        trades = result.scalars().all()
        return trades

@router.get("/api/signals/latest")
async def get_latest_signals():
    redis = await get_redis()
    cache = FeatureCache(redis)
    
    # Ideally iterate over watched symbols. Hardcoding BTCUSDT for dashboard.
    data = await cache.get_features("BTCUSDT")
    if data:
        return data
    return {}

@router.get("/api/news/recent")
async def get_news_recent(limit: int = 20):
    async with async_session_maker() as session:
        stmt = select(NewsPrediction).order_by(desc(NewsPrediction.created_at)).limit(limit)
        result = await session.execute(stmt)
        return result.scalars().all()

@router.get("/api/news/raw")
async def get_raw_news():
    redis = await get_redis()
    raw = await redis.lrange("recent_raw_news", 0, 19)
    
    is_fake_redis = "FakeRedis" in str(type(redis))
    
    valid_news = []
    if raw and not is_fake_redis:
        for r in raw:
            try:
                valid_news.append(json.loads(r))
            except:
                pass
    else:
        # Fallback to physical cross-process JSON cache if FakeRedis is active
        import os
        cache_file = os.path.join(os.getcwd(), "raw_news_cache.json")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "r") as f:
                    valid_news = json.load(f)
            except Exception:
                pass
                
    return valid_news

@router.get("/api/universe")
async def get_universe():
    """Tradeable universe for the dashboard asset tabs (crypto + restricted stocks + ETP map)."""
    from backend.core.universe import as_dict
    return as_dict()


@router.get("/api/brokers")
async def get_brokers():
    """Broker connectivity status for the dashboard. IBKR availability is read from
    Redis (the agent process holds the Gateway connection — IB only allows one
    client per clientId, so the FastAPI process can't probe it directly)."""
    from backend.execution.alpaca_broker import AlpacaBroker
    a = AlpacaBroker()

    ibkr_av = False
    try:
        from backend.memory.redis_client import get_redis
        r = await get_redis()
        raw = await r.get("brokers:availability")
        if raw:
            import json as _json
            data = _json.loads(raw if isinstance(raw, str) else raw.decode())
            ibkr_av = bool(data.get("lse_etp"))
    except Exception:
        pass

    return {
        "crypto_uniswap":  {"asset_class": "crypto",   "available": True,                  "venue": "uniswap"},
        "alpaca_us_stock": {"asset_class": "us_stock", "available": bool(a.is_available()),"venue": "nasdaq/nyse"},
        "ibkr_lse_etp":    {"asset_class": "lse_etp",  "available": ibkr_av,               "venue": "lse"},
    }


@router.get("/api/attention")
async def get_attention():
    """Current per-symbol attention state (published by the agent) + manual overrides."""
    from backend.memory.redis_client import get_redis
    r = await get_redis()
    state: Dict[str, Any] = {}
    overrides: Dict[str, Any] = {}
    try:
        s = await r.get("attention:state")
        if s:
            state = json.loads(s if isinstance(s, str) else s.decode())
    except Exception:
        pass
    try:
        o = await r.get("attention:overrides")
        if o:
            overrides = json.loads(o if isinstance(o, str) else o.decode())
    except Exception:
        pass
    return {"state": state, "overrides": overrides}


class AttentionOverrideRequest(BaseModel):
    symbol: str
    attention: str | None = None  # "high" | "low" | None / "auto" -> clear


@router.post("/api/attention")
async def set_attention(req: AttentionOverrideRequest):
    """Set / clear a manual attention override for one symbol (auto = clear)."""
    from backend.memory.redis_client import get_redis
    r = await get_redis()
    try:
        raw = await r.get("attention:overrides")
        overrides = json.loads(raw if isinstance(raw, str) else raw.decode()) if raw else {}
    except Exception:
        overrides = {}
    att = (req.attention or "").lower() or None
    if att in (None, "auto", ""):
        overrides.pop(req.symbol, None)
    elif att in ("high", "low"):
        overrides[req.symbol] = att
    else:
        raise HTTPException(status_code=400, detail="attention must be 'high', 'low', or null/auto")
    await r.set("attention:overrides", json.dumps(overrides))
    return {"ok": True, "overrides": overrides}


@router.get("/api/audit")
async def get_audit(limit: int = 20):
    async with async_session_maker() as session:
        stmt = select(Trade).where(Trade.closed_at.isnot(None)).order_by(desc(Trade.opened_at)).limit(limit)
        result = await session.execute(stmt)
        return result.scalars().all()

# WebSocket connections
connected_clients: List[WebSocket] = []

@router.websocket("/ws/live")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_clients.append(websocket)
    try:
        while True:
            # We'll just wait for messages or keep the connection open.
            # In main.py or a background task we can push updates to `connected_clients`.
            data = await websocket.receive_text()
    except WebSocketDisconnect:
        connected_clients.remove(websocket)

async def broadcast_ws_message(message_type: str, data: dict):
    payload = json.dumps({"type": message_type, "data": data})
    for client in list(connected_clients):
        try:
            await client.send_text(payload)
        except Exception:
            connected_clients.remove(client)

# Background task to send live updates via WS
async def ws_live_updater():
    redis = await get_redis()
    cache = FeatureCache(redis)
    while True:
        try:
            # Broadcast cycle update every 5s
            features = await cache.get_features("BTCUSDT")
            if features:
                await broadcast_ws_message("cycle_update", features)
                
            # Broadcast NN predictions for UI overlay
            prediction_keys = await redis.keys("agent_visual_predictions:*")
            if prediction_keys:
                for key in prediction_keys:
                    predictions_str = await redis.get(key)
                    if not predictions_str:
                        continue
                    predictions = json.loads(predictions_str)
                    await broadcast_ws_message("prediction_update", predictions)
            else:
                # Backward-compatible fallback to legacy single-payload key
                predictions_str = await redis.get("agent_visual_predictions")
                if predictions_str:
                    predictions = json.loads(predictions_str)
                    await broadcast_ws_message("prediction_update", predictions)

        except Exception as e:
            logger.error("ws_broadcast_error", error=str(e))
        await asyncio.sleep(5)
