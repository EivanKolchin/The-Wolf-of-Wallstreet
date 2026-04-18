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

    ARBITRUM_RPC_URL: str = ""
    AGENT_WALLET_ADDRESS: str = ""
    AGENT_PRIVATE_KEY: str = ""
    INITIAL_USDC_AMOUNT: float = 1000.0

    ALPACA_API_KEY: str = ""
    ALPACA_SECRET: str = ""
    ALPACA_SECRET_KEY: str = ""

    KITE_CHAIN_RPC_URL: str = ""
    KITE_CHAIN_PRIVATE_KEY: str = ""
    KITE_AGENT_ADDRESS: str = ""

    X_API_KEY: str = ""
    X_API_SECRET: str = ""
    X_ACCESS_TOKEN: str = ""
    X_ACCESS_TOKEN_SECRET: str = ""
    TELEGRAM_API_ID: str = ""
    TELEGRAM_API_HASH: str = ""
    
    PAPER_MODE: str = "true"
    
    ADMIN_KEY: str = "supersecretadmin"
    LOG_LEVEL: str = "INFO"

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
