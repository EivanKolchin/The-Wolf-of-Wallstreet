from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path
import os

# Project root: .../trading-agent
BASE_DIR = Path(__file__).resolve().parents[2]
ENV_PATH = BASE_DIR / ".env"

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=str(ENV_PATH), env_file_encoding='utf-8')

    DATABASE_URL: str = "sqlite+aiosqlite:///./trading-agent.db"
    REDIS_URL: str = "redis://localhost:6379/0"

    ANTHROPIC_API_KEY: str = ""
    GEMINI_API_KEY: str = ""
    AI_PROVIDER: str = "anthropic" 

    ARBITRUM_RPC_URL: str = ""
    AGENT_WALLET_ADDRESS: str = ""
    AGENT_PRIVATE_KEY: str = ""
    INITIAL_USDC_AMOUNT: float = 1000.0

    ALPACA_API_KEY: str = ""
    ALPACA_SECRET: str = ""

    KITE_CHAIN_RPC_URL: str = ""
    KITE_CHAIN_PRIVATE_KEY: str = ""
    KITE_AGENT_ADDRESS: str = ""

    X_API_KEY: str = ""
    TELEGRAM_API_ID: str = ""
    TELEGRAM_API_HASH: str = ""
    
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
        else:
            has_ai_key = is_valid_key(self.ANTHROPIC_API_KEY, prefix="sk-ant-", min_len=30)

        if not has_ai_key:
            return True

        essential_keys = [self.AGENT_PRIVATE_KEY]
        for val in essential_keys:
            if not is_valid_key(val, min_len=30):
                return False  # Not failing hard if wallet not supplied since there's paper mode
        return False

settings = None
try:
    settings = Settings()
except Exception as e:
    settings = Settings(_env_file=None)
