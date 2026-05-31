from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path
import os

# Project root: .../trading-agent
BASE_DIR = Path(__file__).resolve().parents[2]
ENV_PATH = BASE_DIR / ".env"

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=str(ENV_PATH), env_file_encoding='utf-8', extra='ignore')

    DATABASE_URL: str = "sqlite+aiosqlite:///./trading-agent.db"
    REDIS_URL: str = "redis://localhost:6379/0"

    ANTHROPIC_API_KEY: str = ""
    GEMINI_API_KEY: str = ""
    AI_PROVIDER: str = "anthropic" 
    OLLAMA_MODEL: str = "llama3"
    OLLAMA_FALLBACK_MODEL: str = ""
    AUTO_PULL_OLLAMA_MODELS: bool = True

    ARBITRUM_RPC_URL: str = ""
    AGENT_WALLET_ADDRESS: str = ""
    AGENT_PRIVATE_KEY: str = ""
    INITIAL_USDC_AMOUNT: float = 1000.0

    ALPACA_API_KEY: str = ""
    ALPACA_SECRET: str = ""
    ALPACA_SECRET_KEY: str = ""

    # Stock brokers besides Alpaca + optional extra (free) stock-data providers
    IBKR_HOST: str = "127.0.0.1"        # IB Gateway / TWS host (LSE leveraged-ETP execution)
    IBKR_PORT: str = "4002"             # 4002 paper gateway / 7497 paper TWS (string-tolerant for blank .env)
    IBKR_CLIENT_ID: str = "11"          # string so a blank .env value can't break settings loading
    IBKR_ACCOUNT_ID: str = ""
    FINNHUB_API_KEY: str = ""           # optional free stock data provider
    TWELVEDATA_API_KEY: str = ""        # optional free stock data provider

    X_API_KEY: str = ""
    X_API_SECRET: str = ""
    X_ACCESS_TOKEN: str = ""
    X_ACCESS_TOKEN_SECRET: str = ""
    TELEGRAM_API_ID: str = ""
    TELEGRAM_API_HASH: str = ""
    
    PAPER_MODE: str = "true"
    
    ADMIN_KEY: str = "supersecretadmin"
    LOG_LEVEL: str = "INFO"
    NN_HOLD_PROB_MULTIPLIER: float = 1.0
    NN_MIN_ACTION_CONFIDENCE: float = 0.0
    NN_LONG_CONFIDENCE_THRESHOLD: float = 0.0
    NN_SHORT_CONFIDENCE_THRESHOLD: float = 0.0
    NN_ONLINE_PNL_NOISE_BAND: float = 0.0005
    NN_ONLINE_PNL_WEIGHT_SCALE: float = 0.01
    NN_ONLINE_WEIGHT_CAP: float = 3.0
    FEATURE_SCHEMA_VERSION: str = "v2.0"

    # API rate limiting (keep outbound calls under exchange limits)
    BINANCE_RATE_LIMIT_PER_SEC: float = 8.0

    # RL (Advantage-Weighted Regression) + reward shaping
    NN_REWARD_K: float = 10.0              # tanh steepness on fee-net pnl
    NN_HOLD_PENALTY: float = 0.05          # penalty per horizon held beyond the primary horizon
    NN_AWR_BETA: float = 1.0               # AWR temperature (higher = softer weighting)
    NN_AWR_WEIGHT_CAP: float = 20.0        # max per-sample advantage weight
    # Sortino-style risk adjustment: dampen the per-trade reward by the recent
    # downside deviation of realized returns, so smooth/consistent gains are
    # rewarded more than the same PnL earned in a volatile, drawdown-prone streak.
    NN_SORTINO_WINDOW: int = 50            # # of recent realized returns used for downside deviation
    NN_DOWNSIDE_WEIGHT: float = 25.0       # strength of the downside-deviation dampening (0 disables)
    # News-text embedding (Phase 3): feeds the semantic NEWS_EMBED block [70:86].
    NN_NEWS_EMBED_BACKEND: str = "hashing"        # "hashing" (no deps) | "transformer" (sentence-transformers)
    NN_NEWS_EMBED_MODEL: str = "all-MiniLM-L6-v2" # used only when backend == "transformer"
    NN_NEWS_LABEL_LOG: str = "training_data/news_labels.jsonl"  # online self-labeling accumulator
    NN_NEWS_LABEL_HORIZON_MIN: int = 60    # forward window (minutes) to realize a news label
    # Extended-hours (Phase 4): additive pre/after-hours trading + data. Never
    # replaces Alpaca/IBKR — augments them. Disable to confine activity to RTH.
    EXTENDED_HOURS_TRADING_ENABLED: bool = True   # allow Alpaca extended_hours limit orders pre/after
    EXTENDED_HOURS_DATA_ENABLED: bool = True       # surface yfinance/Finnhub pre/after quotes
    # Hardware-aware compute budget (Workstream B). Auto-tunes the BLAS/torch
    # thread pools to the host instead of the legacy hard-pinned 1 thread.
    HW_AUTO_TUNE: bool = True               # false => keep the legacy single-thread behavior
    HW_RESERVED_CORES: int = 2              # physical cores left for OS/I/O headroom
    HW_THREAD_CAP: int = 8                  # max intra-op threads (guards E-core thrash on the i5)
    HW_USE_IGPU: bool = False               # opt-in Intel iGPU inference (shared RAM — off by default)
    # Uncertainty-gated trading (Workstream C): require edge to dominate MC noise.
    NN_MIN_EDGE_TO_UNCERTAINTY: float = 1.0  # min |edge_mean|/(edge_std+eps) to act; 0 disables
    # A4 / anti-overfitting architecture knobs (model is untrained → retrain is
    # free). These must match between live (improved_model.py) and offline
    # (scripts/pretrain.py) for checkpoints to load — both read these settings.
    NN_RNN_TYPE: str = "lstm"               # "lstm" | "gru" (gru = fewer params, faster, less overfit)
    NN_DROPOUT: float = 0.3                 # recurrent + trunk dropout (regularization)
    NN_WEIGHT_DECAY: float = 1e-4           # L2 regularization (was 1e-5; modest anti-overfit bump)
    NN_LABEL_SMOOTHING: float = 0.05        # softens targets → less overconfident, less overfit
    NN_AUGMENT_NOISE_STD: float = 0.0       # Gaussian noise added to replay sequences in online AWR (0 = off)
    # Funding / wallet (Workstreams D/E): MetaMask connect (WalletConnect QR) +
    # Google-Pay fiat on-ramp. Publishable keys are safe in the frontend; they're
    # served by /api/wallet/config and set via Settings -> .env.
    WALLETCONNECT_PROJECT_ID: str = ""      # free from cloud.reown.com — enables the connect-QR
    ONRAMP_PROVIDER: str = "ramp"           # "ramp" (lowest fee) | "moonpay" | "transak"
    RAMP_HOST_API_KEY: str = ""             # publishable host key from ramp.network
    ONRAMP_DEFAULT_PAYMENT_METHOD: str = "" # e.g. "google_pay" — biases the widget where supported
    ONRAMP_CRYPTO_ASSET: str = "ARBITRUM_USDC"  # asset delivered to the agent wallet
    DEPOSIT_CHAIN_ID: int = 42161           # Arbitrum One — for the EIP-681 deposit QR
    # Anti-inaction: discourage the agent from holding forever
    NN_IDLE_HOLD_DECAY: float = 0.5        # at full idle pressure, hold prob scaled by (1 - this)
    NN_IDLE_PATIENCE: int = 60             # cycles of no-trade before idle pressure reaches 1.0
    NN_MIN_EDGE_OVER_FEE: float = 1.0      # require |edge| > this * round-trip fee to open a trade

    # Bayesian position sizing
    NN_KELLY_FRACTION: float = 0.5         # fraction of Kelly to bet (0 disables -> use model size)
    NN_MC_SAMPLES: int = 16                # MC-dropout passes for edge uncertainty
    NN_USE_IPEX: bool = True               # apply Intel Extension for PyTorch (CPU inference speedup) when installed

    # Risk limits (editable via Advanced Options in the UI; applied on restart)
    RISK_MAX_DRAWDOWN_PCT: float = 15.0
    RISK_MAX_DAILY_LOSS_PCT: float = 5.0
    RISK_MAX_POSITION_PCT: float = 20.0
    RISK_MIN_CONFIDENCE: float = 0.0
    RISK_MAX_TRADES_PER_HOUR: int = 2000
    RISK_MIN_POSITION_USD: float = 12.0
    RISK_MAX_POSITION_USD: float = 5000.0
    RISK_CVAR_LIMIT_PCT: float = 10.0      # block new trades if projected 10-trade CVaR exceeds this

    # Variable Attention Engine (Phase 11) — dynamic compute cadence
    ATTENTION_HIGH_SECONDS: float = 1.0        # poll/inference interval under high attention
    ATTENTION_LOW_SECONDS: float = 300.0       # interval under low attention (flat/quiet)
    ATTENTION_VOL_THRESHOLD: float = 0.004     # recent return std above this -> high attention
    ATTENTION_R2_THRESHOLD: float = 0.6        # linear-fit R^2 below this -> non-linear -> high attention
    ATTENTION_VOLUME_THRESHOLD: float = 1.8    # volume ratio above this -> high attention

    # Statistical R:R boundary floors (Phase 9) — vol-aware SL + minimum reward:risk
    NN_BOUNDARY_K_SIGMA: float = 1.0           # SL floor = max(0.3%, k * recent_return_std)
    NN_MIN_RR_RATIO: float = 1.5               # TP must be >= sl * this  (e.g. 1.5x)

    # Cycle 19.3: momentum-reversal exit thresholds used by PositionMonitor.
    NN_REVERSAL_MACD_DROP: float = 0.5         # prev_macd_hist - curr_macd_hist must exceed this
    NN_REVERSAL_VOL_MULTIPLE: float = 2.0      # curr volume must be > this × 20-bar vol MA

    # Cycle 19.4: DeFi gas-fee ceilings (Arbitrum-tuned defaults). Above these
    # the swap aborts rather than burn ETH during congestion.
    DEFI_MAX_GAS_UNITS: int = 500_000          # hard cap on the gas-units field
    DEFI_MAX_GAS_PRICE_GWEI: float = 5.0       # gas_price above this -> abort the swap

    def needs_setup(self) -> bool:
        def is_valid_key(k: str, prefix: str = None, min_len: int = 15) -> bool:
            if not k:
                return False
            k_lower = k.lower()
            if "your_" in k_lower or "fake" in k_lower or "placeholder" in k_lower:
                return False
            if len(k) < min_len:
                return False
            if prefix and not k.startswith(prefix):
                return False
            return True

        if self.AI_PROVIDER == "gemini":
            has_ai_key = is_valid_key(self.GEMINI_API_KEY, prefix="AIzaSy", min_len=30)
        elif self.AI_PROVIDER == "anthropic":
            has_ai_key = is_valid_key(self.ANTHROPIC_API_KEY, prefix="sk-ant-", min_len=30)
        elif self.AI_PROVIDER == "hybrid_gemini":
            has_ai_key = is_valid_key(self.GEMINI_API_KEY, prefix="AIzaSy", min_len=30)
        elif self.AI_PROVIDER == "hybrid_claude":
            has_ai_key = is_valid_key(self.ANTHROPIC_API_KEY, prefix="sk-ant-", min_len=30)
        else: # ollama only
            has_ai_key = True

        if not has_ai_key:
            return True

        essential_keys = [self.AGENT_PRIVATE_KEY]
        for val in essential_keys:
            if not is_valid_key(val, min_len=30):
                return False  # Not failing hard if wallet not supplied since there's paper mode
        return False

    def missing_integration_status(self) -> dict:
        """Determines which integrations are missing API keys and provides human-readable context on what features the user will lose"""
        missing = []
        is_real = lambda k: bool(k and len(k) > 5 and "your_" not in k.lower())

        if not is_real(self.X_API_KEY):
            missing.append({
                "service": "X (Twitter) Firehose",
                "impact": "Agent cannot ingest real-time sentiment from crypto Twitter. It will function purely on delayed RSS feeds."
            })
            
        if not is_real(self.TELEGRAM_API_ID):
            missing.append({
                "service": "Telegram Scraper",
                "impact": "Agent loses access to alpha groups and fast-breaking Telegram announcements."
            })

        if not is_real(self.ALPACA_API_KEY):
            missing.append({
                "service": "Alpaca Data/Broker",
                "impact": "Agent cannot access Alpaca stock integrations or paid crypto news feeds, settling for generic free feeds."
            })
            
        return missing

settings = None
try:
    settings = Settings()
except Exception as e:
    settings = Settings(_env_file=None)
