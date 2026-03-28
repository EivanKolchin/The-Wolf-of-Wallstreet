import json
import asyncio
import structlog
from web3 import Web3, AsyncWeb3
from eth_account import Account

from backend.memory.database import Trade
from backend.agents.nn_agent import TradeDecision # Explicit dependency needed per spec typing

logger = structlog.get_logger(__name__)

class KiteChainClient:
    def __init__(self, rpc_url: str, private_key: str, agent_address: str, db_session_factory=None):
        self.rpc_url = rpc_url
        self.private_key = private_key
        self.agent_address = agent_address
        self.db_session_factory = db_session_factory
        
        # Determine async capability from web3, use robust provider
        # Assuming AsyncWeb3 implementation
        self.w3 = AsyncWeb3(AsyncWeb3.AsyncHTTPProvider(self.rpc_url))
        
        self.account = None
        if private_key:
            try:
                self.account = Account.from_key(private_key)
            except Exception as e:
                logger.warning("kite_chain_invalid_private_key", error=str(e), msg="Private key could not be parsed. Kite Chain logging is disabled.")

    async def log_trade_decision(self, trade: Trade, decision: TradeDecision) -> str | None:
        if not self.account:
            logger.warning("kite_chain_missing_private_key")
            return None

        try:
            summary = {
                "trade_id": str(trade.id),
                "symbol": trade.asset,
                "direction": trade.direction.value if trade.direction else "unknown",
                "nn_confidence": trade.nn_confidence,
                "regime": trade.regime_at_entry,
                "news_impact": decision.active_news.to_json() if decision.active_news and hasattr(decision.active_news, 'to_json') else (decision.active_news if decision.active_news else None),
                "timestamp": trade.opened_at.isoformat() if trade.opened_at else ""
            }
            
            payload_json = json.dumps(summary)
            data_hex = self.w3.to_hex(text=payload_json)
            
            # Nonce management
            nonce = await self.w3.eth.get_transaction_count(self.account.address)
            
            # Gas estimation
            gas_price = await self.w3.eth.gas_price
            adjusted_gas_price = int(gas_price * 1.1)
            
            tx = {
                "nonce": nonce,
                "to": self.agent_address,
                "value": 0,
                "gas": 100_000, 
                "gasPrice": adjusted_gas_price,
                "data": data_hex,
                "chainId": await self.w3.eth.chain_id
            }
            
            signed_tx = self.w3.eth.account.sign_transaction(tx, self.private_key)
            tx_hash = await self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
            
            hex_hash = self.w3.to_hex(tx_hash)
            
            # Update DB
            if self.db_session_factory:
                async with self.db_session_factory() as session:
                    # we must fetch because `trade` is detached in asyncio.create_task context
                    t = await session.get(Trade, trade.id)
                    if t:
                        t.kite_tx_hash = hex_hash
                        await session.commit()
            
            logger.info("trade_decision_logged_on_chain", tx_hash=hex_hash, trade_id=str(trade.id))
            return hex_hash
            
        except Exception as e:
            logger.error("kite_chain_logging_failed", error=str(e), trade_id=str(trade.id))
            return None

    async def log_prediction(self, prediction_id: str, summary: dict) -> str | None:
        if not self.account:
            return None
            
        try:
            summary["prediction_id"] = prediction_id
            payload_json = json.dumps(summary)
            data_hex = self.w3.to_hex(text=payload_json)
            
            nonce = await self.w3.eth.get_transaction_count(self.account.address)
            gas_price = await self.w3.eth.gas_price
            adjusted_gas_price = int(gas_price * 1.1)
            
            tx = {
                "nonce": nonce,
                "to": self.agent_address,
                "value": 0,
                "gas": 100_000, 
                "gasPrice": adjusted_gas_price,
                "data": data_hex,
                "chainId": await self.w3.eth.chain_id
            }
            
            signed_tx = self.w3.eth.account.sign_transaction(tx, self.private_key)
            tx_hash = await self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
            
            hex_hash = self.w3.to_hex(tx_hash)
            logger.info("prediction_logged_on_chain", tx_hash=hex_hash, prediction_id=prediction_id)
            return hex_hash
            
        except Exception as e:
            logger.error("prediction_logging_failed", error=str(e), prediction_id=prediction_id)
            return None

    async def get_agent_reputation(self) -> dict:
        # Placeholder for real DB fetching. 
        # For full implementation, requires complex group_by queries on Trade / NewsPrediction tables
        return {
            "trade_count": 0,
            "win_rate": 0.0,
            "avg_prediction_score": 0.0,
            "total_pnl_usd": 0.0
        }