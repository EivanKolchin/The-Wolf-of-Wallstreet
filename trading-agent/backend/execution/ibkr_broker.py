"""IBKR broker adapter — LSE leveraged-ETP execution via ib_insync (TWS / IB Gateway).

Streaming market data via `reqMktData` (not REST polling — avoids IBKR pacing /
Error 100). `is_available()` is False until `connect()` has succeeded against a
running Gateway, so the registry falls back gracefully when the Gateway is offline.
Persists real `Trade` rows on open/close (mirrors the crypto/Alpaca path) so the
position monitor, statements ledger, and AWR learner all work uniformly.
"""
from __future__ import annotations

import time
from dataclasses import asdict as _asdict
from datetime import datetime as _dt
from typing import Dict, Optional

import structlog

from backend.execution.base import BrokerInterface, AssetClass
from backend.core import ledger
from backend.core.config import settings
from backend.memory.database import Trade, TradeDirection, TradeStatus, OrderType

logger = structlog.get_logger(__name__)

try:
    from ib_insync import IB, Stock, MarketOrder  # type: ignore
    HAS_IB = True
except Exception:
    HAS_IB = False


class IBKRBroker(BrokerInterface):
    asset_class = AssetClass.lse_etp.value

    def __init__(self, host: Optional[str] = None, port=None, client_id=None, db_session_factory=None):
        self.host = host or getattr(settings, "IBKR_HOST", "127.0.0.1")
        try:
            self.port = int(port if port is not None else settings.IBKR_PORT)
        except Exception:
            self.port = 4002
        try:
            self.client_id = int(client_id if client_id is not None else settings.IBKR_CLIENT_ID)
        except Exception:
            self.client_id = 11
        self.account_id = getattr(settings, "IBKR_ACCOUNT_ID", "") or ""
        self.db_session_factory = db_session_factory

        self.ib = None
        self._open: Dict[str, Trade] = {}
        self._closing: set = set()
        self._tickers: dict = {}
        self._contracts: dict = {}
        self._trade_closed_callback = None

    # ----------------------------------------------------------------- lifecycle
    def is_available(self) -> bool:
        if not HAS_IB or self.ib is None:
            return False
        try:
            return bool(self.ib.isConnected())
        except Exception:
            return False

    def quote_asset(self, symbol: str) -> str:
        return "GBP"

    async def connect(self) -> bool:
        if not HAS_IB:
            logger.warning("ibkr_unavailable_no_ib_insync")
            return False
        if self.ib and self.ib.isConnected():
            return True
        try:
            self.ib = IB()
            await self.ib.connectAsync(self.host, self.port, clientId=self.client_id)
            logger.info("ibkr_connected", host=self.host, port=self.port,
                        client_id=self.client_id, account=self.account_id or None)
            return True
        except Exception as e:
            logger.warning("ibkr_connect_failed", host=self.host, port=self.port, error=str(e))
            self.ib = None
            return False

    async def disconnect(self) -> None:
        if self.ib and self.ib.isConnected():
            try:
                self.ib.disconnect()
            except Exception:
                pass
        self.ib = None

    # ----------------------------------------------------------------- contracts
    async def _qualified_contract(self, symbol: str):
        if symbol in self._contracts:
            return self._contracts[symbol]
        c = Stock(symbol, "LSE", "GBP")
        try:
            qualified = await self.ib.qualifyContractsAsync(c)
            if qualified:
                c = qualified[0]
        except Exception as e:
            logger.warning("ibkr_qualify_failed", symbol=symbol, error=str(e))
        self._contracts[symbol] = c
        return c

    async def _subscribe(self, symbol: str):
        if symbol in self._tickers:
            return self._tickers[symbol]
        c = await self._qualified_contract(symbol)
        try:
            t = self.ib.reqMktData(c, "", False, False)  # streaming
            self._tickers[symbol] = t
            return t
        except Exception as e:
            logger.warning("ibkr_reqMktData_failed", symbol=symbol, error=str(e))
            return None

    async def get_price(self, symbol: str) -> Optional[float]:
        if not self.is_available():
            return None
        try:
            t = await self._subscribe(symbol)
            if t is None:
                return None
            price = None
            try:
                mp = t.marketPrice()
                if mp is not None and mp == mp:                # NaN check
                    price = float(mp)
            except Exception:
                pass
            if price is None:
                for attr in ("last", "close", "bid", "ask"):
                    v = getattr(t, attr, None)
                    if v is not None and v == v:               # NaN check
                        price = float(v); break
            return price
        except Exception as e:
            logger.warning("ibkr_get_price_failed", symbol=symbol, error=str(e))
            return None

    def get_open_trades(self) -> dict:
        return self._open

    # ----------------------------------------------------------------- execution
    async def execute(self, decision, portfolio_state: dict):
        if not self.is_available():
            logger.warning("ibkr_unavailable_skip_execute")
            return None
        if getattr(decision, "direction", "hold") == "hold":
            return None

        symbol = decision.symbol
        direction = decision.direction
        size_usd = float(getattr(decision, "size_pct", 0.1)) * float(portfolio_state.get("available_cash", 0.0) or 0.0)
        if size_usd <= 0:
            return None
        price = await self.get_price(symbol)
        if not price or price <= 0:
            logger.warning("ibkr_no_price_skip_execute", symbol=symbol)
            return None

        qty = max(1, int(size_usd / price))                   # IBKR LSE -> whole shares
        side = "BUY" if direction == "long" else "SELL"
        try:
            c = await self._qualified_contract(symbol)
            order = MarketOrder(side, qty)
            self.ib.placeOrder(c, order)
            ledger.log_api_call("ibkr", "ORDER", f"placeOrder/{symbol}", ok=True, note=f"side={side} qty={qty}")
        except Exception as e:
            logger.error("ibkr_place_order_failed", symbol=symbol, error=str(e))
            ledger.log_api_call("ibkr", "ORDER", f"placeOrder/{symbol}", ok=False, note=str(e)[:120])
            return None

        trade = self._build_trade(decision, symbol, direction, size_usd, price)
        if self.db_session_factory:
            try:
                async with self.db_session_factory() as session:
                    session.add(trade)
                    await session.commit()
                    await session.refresh(trade)
            except Exception as e:
                logger.error("ibkr_persist_failed", error=str(e))

        self._open[symbol] = trade
        return trade

    def _build_trade(self, decision, symbol, direction, size_usd, entry_price) -> Trade:
        sl_frac = float(getattr(decision, "sl", 0.0) or 0.0)
        tp_frac = float(getattr(decision, "tp", 0.0) or 0.0)
        trail_frac = float(getattr(decision, "trail", 0.0) or 0.0)
        if direction == "long":
            sl_abs = entry_price * (1 - sl_frac) if sl_frac > 0 else 0.0
            tp_abs = entry_price * (1 + tp_frac) if tp_frac > 0 else 0.0
        else:
            sl_abs = entry_price * (1 + sl_frac) if sl_frac > 0 else 0.0
            tp_abs = entry_price * (1 - tp_frac) if tp_frac > 0 else 0.0
        try:
            news = getattr(decision, "active_news", None)
            news_dict = _asdict(news) if (news is not None and hasattr(news, "__dataclass_fields__")) else None
        except Exception:
            news_dict = None
        return Trade(
            asset=symbol,
            direction=TradeDirection.long if direction == "long" else TradeDirection.short,
            size_usd=float(size_usd), entry_price=float(entry_price),
            status=TradeStatus.open, order_type=OrderType.market,
            nn_confidence=float(getattr(decision, "nn_confidence", 0.0)),
            nn_direction_probs=getattr(decision, "nn_probs", {}) or {},
            active_news_impact=news_dict,
            regime_at_entry=str(getattr(decision, "regime", "")),
            stop_loss=sl_abs, take_profit=tp_abs,
            quote_asset="GBP", fee_paid=0.0,
            trailing_stop=trail_frac, highest_price_seen=float(entry_price),
            broker="ibkr",
            account_id=self.account_id or None,
            asset_class="lse_etp",
            target_price=float(getattr(decision, "target_price", 0.0) or 0.0),
            expected_execution_ts=float(getattr(decision, "expected_execution_ts", 0.0) or 0.0),
            rationale=getattr(decision, "rationale", None),
        )

    async def close_position(self, symbol: str, reason: str = "signal"):
        if symbol in self._closing:
            return None
        self._closing.add(symbol)
        try:
            trade = self._open.get(symbol)
            if trade is None:
                return None
            exit_price = await self.get_price(symbol) or float(trade.entry_price)
            direction = trade.direction.value if hasattr(trade.direction, "value") else str(trade.direction)
            qty = max(1, int(float(trade.size_usd) / float(trade.entry_price)))
            close_side = "SELL" if direction == "long" else "BUY"
            try:
                c = await self._qualified_contract(symbol)
                self.ib.placeOrder(c, MarketOrder(close_side, qty))
                ledger.log_api_call("ibkr", "ORDER", f"closeOrder/{symbol}",
                                    ok=True, note=f"side={close_side} qty={qty}")
            except Exception as e:
                logger.error("ibkr_close_order_failed", symbol=symbol, error=str(e))
                ledger.log_api_call("ibkr", "ORDER", f"closeOrder/{symbol}", ok=False, note=str(e)[:120])

            entry = float(trade.entry_price); size_usd = float(trade.size_usd)
            pnl_pct = ((exit_price - entry) / entry) if entry > 0 else 0.0
            if direction == "short":
                pnl_pct = -pnl_pct
            gross = size_usd * pnl_pct
            fee_close = 0.0           # IBKR commissions vary by exchange/plan; track per-fill later
            fee_total = float(trade.fee_paid or 0.0) + fee_close
            net_pnl_usd = gross - fee_close
            net_pnl_pct = (net_pnl_usd - float(trade.fee_paid or 0.0)) / size_usd if size_usd > 0 else 0.0

            updated = await self._finalize_row(
                trade.id, exit_price=exit_price, pnl_usd=net_pnl_usd, pnl_pct=net_pnl_pct,
                fee_total=fee_total, reason=reason,
                highest_price_seen=getattr(trade, "highest_price_seen", entry),
            )

            ledger.record_transaction({
                "opened_at": str(getattr(trade, "opened_at", "")),
                "trade_id": str(trade.id),
                "symbol": symbol, "asset_class": "lse_etp", "broker": "ibkr",
                "direction": direction, "size_usd": round(size_usd, 4),
                "entry_price": round(entry, 6), "exit_price": round(float(exit_price), 6),
                "pnl_usd": round(net_pnl_usd, 4), "pnl_pct": round(net_pnl_pct, 6),
                "fee_paid": round(fee_total, 4), "quote_asset": "GBP",
                "exit_reason": reason, "paper": True,
                "target_price": float(getattr(trade, "target_price", 0.0) or 0.0),
            })

            self._open.pop(symbol, None)
            logger.info("ibkr_position_closed", symbol=symbol, reason=reason, pnl_pct=round(net_pnl_pct, 5))
            if self._trade_closed_callback:
                await self._trade_closed_callback(updated or trade, net_pnl_pct)
            return updated or trade
        finally:
            self._closing.discard(symbol)

    async def _finalize_row(self, trade_id, *, exit_price, pnl_usd, pnl_pct, fee_total, reason, highest_price_seen):
        if not self.db_session_factory:
            return None
        try:
            async with self.db_session_factory() as session:
                row = await session.get(Trade, trade_id)
                if row is None:
                    return None
                row.exit_price = float(exit_price)
                row.status = TradeStatus.closed
                row.closed_at = _dt.utcnow()
                row.pnl_usd = float(pnl_usd)
                row.pnl_pct = float(pnl_pct)
                row.fee_paid = float(fee_total)
                row.exit_reason = str(reason)
                row.highest_price_seen = float(highest_price_seen) if highest_price_seen else row.entry_price
                await session.commit()
                await session.refresh(row)
                return row
        except Exception as e:
            logger.error("ibkr_finalize_trade_failed", trade_id=str(trade_id), error=str(e))
            return None
