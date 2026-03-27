from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8')

    DATABASE_URL: str
    REDIS_URL: str
    
    ANTHROPIC_API_KEY: str
    
    BINANCE_API_KEY: str
    BINANCE_SECRET: str
    
    ALPACA_API_KEY: str
    ALPACA_SECRET: str
    
    KITE_CHAIN_RPC_URL: str
    KITE_CHAIN_PRIVATE_KEY: str
    KITE_AGENT_ADDRESS: str
    
    X_API_KEY: str
    TELEGRAM_API_ID: str
    TELEGRAM_API_HASH: str
    
    ADMIN_KEY: str = "supersecretadmin"

