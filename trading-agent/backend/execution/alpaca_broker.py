"""Alpaca broker adapter — US stocks (NASDAQ/NYSE) data + paper execution.

Uses the Alpaca REST API directly via aiohttp (no extra SDK dependency). Persists
real `Trade` rows on open/close, writes the same fee-aware transaction statements
the crypto path does, calls the trade-closed callback so the AWR learner reacts
to closes, and routes through the shared API-call ledger.

`is_available()` is False without credentials so the registry falls back gracefully.
Defaults to the paper endpoint. Order submission is real; switching to live needs
`paper=False` plus a funded live account.
"""
from __future__ import annotations

import time
from dataclasses import asdict as _asdict
from datetime import datetime as _dt
from typing import Optional

import aiohttp
import structlog
from sqlalchemy import select

from backend.execution.base import BrokerInterface, AssetClass
from backend.core import ledger
from backend.core.config import settings
from backend.memory.database import Trade, TradeDirection, TradeStatus, OrderType

logger = structlog.get_logger(__name__)

ALPACA_DATA = "https://data.alpaca.markets/v2"
ALPACA_PAPER = "https://paper-api.alpaca.markets/v2"
ALPACA_LIVE = "https://api.alpaca.markets/v2"


class AlpacaBroker(BrokerInterface):
    asset_class = AssetClass.us_stock.value

    def __init__(self, paper: bool = True, db_session_factory=None):
        self.key = (getattr(settings, "ALPACA_API_KEY", "") or "")
        self.secret = (getattr(settings, "ALPACA_SECRET_KEY", "") or getattr(settings, "ALPACA_SECRET", "") or "")
        self.base = ALPACA_PAPER if paper else ALPACA_LIVE
        self.paper = paper
        self.db_session_factory = db_session_factory
        self._open: dict[str, Trade] = {}
        self._closing: set[str] = set()
        self._trade_closed_callback = None

    def is_available(self) -> bool:
        return bool(self.key and self.secret and "your_" not in self.key.lower())

    def quote_asset(self, symbol: str) -> str:
        return "USD"

    def _headers(self) -> dict:
        return {"APCA-API-KEY-ID": self.key, "APCA-API-SECRET-KEY": self.secret}

    # -------------------------------------------------------------- market data
    async def get_price(self, symbol: str) -> Optional[float]:
        if not self.is_available():
            return None
        url = f"{ALPACA_DATA}/stocks/{symbol}/trades/latest"
        t0 = time.monotonic()
        try:
            timeout = aiohttp.ClientTimeout(total=5)
            async with aiohttp.ClientSession(timeout=timeout) as s:
                async with s.get(url, headers=self._headers()) as r:
                    data = await r.json()
                    ledger.log_api_call("alpaca", "GET", url, status=r.status, ok=True,
                                        latency_ms=(time.monotonic() - t0) * 1000)
                    return float(data["trade"]["p"])
        except Exception as e:
            ledger.log_api_call("alpaca", "GET", url, ok=False, note=str(e)[:120])
            return None

    def get_open_trades(self) -> dict:
        return self._open

    # ---------------------------------------------------------------- execution
    async def execute(self, decision, portfolio_state: dict):
        if not self.is_available():
            logger.warning("alpaca_unavailable_skip_execute")
            return None
        if getattr(decision, "direction", "hold") == "hold":
            return None

        symbol = decision.symbol
        direction = decision.direction
        size_pct = float(getattr(decision, "size_pct", 0.1))
        size_usd = size_pct * float(portfolio_state.get("available_cash", 0.0) or 0.0)
        if size_usd <= 0:
            return None

        current_price = await self.get_price(symbol)
        if not current_price or current_price <= 0:
            logger.warning("alpaca_no_price_skip_execute", symbol=symbol)
            return None

        # Submit an order. Inside regular hours we use a market order with
        # notional sizing (fractional shares). In extended/overnight Alpaca
        # rejects market orders, so we issue a marketable LIMIT with
        # extended_hours=true (caps the price at +/-0.5% from last so it still
        # fills aggressively but is accepted).
        from backend.core.market_hours import us_session_state
        side = "buy" if direction == "long" else "sell"
        url = f"{self.base}/orders"
        state = us_session_state()
        if state == "regular":
            payload = {
                "symbol": symbol, "notional": f"{size_usd:.2f}",
                "side": side, "type": "market", "time_in_force": "day",
            }
        else:
            # Phase 4: extended-hours is an OPT-IN augmentation. When disabled,
            # confine trading to the regular session. Also skip 'closed'
            # (weekends) where even extended-hours limit orders are rejected.
            if not bool(getattr(settings, "EXTENDED_HOURS_TRADING_ENABLED", True)):
                logger.info("alpaca_extended_hours_disabled_skip", symbol=symbol, state=state)
                return None
            if state == "closed":
                logger.info("alpaca_market_closed_skip", symbol=symbol)
                return None
            # Extended-hours / overnight: limit order required.
            # Notional sizing is not supported with limit — convert to share qty.
            qty = max(1.0, size_usd / current_price)
            slack = 0.005   # 0.5% — aggressive enough to fill in thin books
            limit_price = current_price * (1 + slack) if side == "buy" else current_price * (1 - slack)
            payload = {
                "symbol": symbol, "qty": f"{qty:.4f}",
                "side": side, "type": "limit",
                "limit_price": f"{limit_price:.4f}",
                "time_in_force": "day",
                "extended_hours": True,
            }
            logger.info("alpaca_extended_hours_order", symbol=symbol, state=state,
                         qty=qty, limit_price=limit_price)
        ok = False
        t0 = time.monotonic()
        try:
            timeout = aiohttp.ClientTimeout(total=10)
            async with aiohttp.ClientSession(timeout=timeout) as s:
                async with s.post(url, headers=self._headers(), json=payload) as r:
                    body = await r.text()
                    ok = r.status in (200, 201)
                    ledger.log_api_call("alpaca", "POST", url, status=r.status, ok=ok,
                                        latency_ms=(time.monotonic() - t0) * 1000)
                    if not ok:
                        logger.error("alpaca_order_rejected", status=r.status, body=body[:300])
                        return None
        except Exception as e:
            logger.error("alpaca_order_post_failed", symbol=symbol, error=str(e))
            ledger.log_api_call("alpaca", "POST", url, ok=False, note=str(e)[:120])
            return None

        # Persist a real Trade row (or build a transient one if no DB wired).
        trade = self._build_trade(decision, symbol, direction, size_usd, current_price)
        if self.db_session_factory:
            try:
                async with self.db_session_factory() as session:
                    session.add(trade)
                    await session.commit()
                    await session.refresh(trade)
            except Exception as e:
                logger.error("alpaca_trade_persist_failed", error=str(e))

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
            quote_asset="USD", fee_paid=0.0,                 # Alpaca: commission-free for stocks
            trailing_stop=trail_frac, highest_price_seen=float(entry_price),
            broker="alpaca",
            account_id=(getattr(settings, "ALPACA_API_KEY", "")[:8] or None),
            asset_class="us_stock",
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

            # DELETE the position via Alpaca (liquidates at market).
            url = f"{self.base}/positions/{symbol}"
            t0 = time.monotonic()
            try:
                timeout = aiohttp.ClientTimeout(total=10)
                async with aiohttp.ClientSession(timeout=timeout) as s:
                    async with s.delete(url, headers=self._headers()) as r:
                        await r.text()
                        ledger.log_api_call("alpaca", "DELETE", url, status=r.status,
                                            ok=r.status in (200, 207, 204),
                                            latency_ms=(time.monotonic() - t0) * 1000)
            except Exception as e:
                logger.error("alpaca_position_close_failed", symbol=symbol, error=str(e))
                ledger.log_api_call("alpaca", "DELETE", url, ok=False, note=str(e)[:120])

            # Compute fee-aware PnL (Alpaca = commission-free stocks).
            exit_price = await self.get_price(symbol) or float(trade.entry_price)
            direction = trade.direction.value if hasattr(trade.direction, "value") else str(trade.direction)
            entry = float(trade.entry_price); size_usd = float(trade.size_usd)
            pnl_pct = ((exit_price - entry) / entry) if entry > 0 else 0.0
            if direction == "short":
                pnl_pct = -pnl_pct
            gross = size_usd * pnl_pct
            fee_close = 0.0
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
                "symbol": symbol, "asset_class": "us_stock", "broker": "alpaca",
                "direction": direction, "size_usd": round(size_usd, 4),
                "entry_price": round(entry, 6), "exit_price": round(float(exit_price), 6),
                "pnl_usd": round(net_pnl_usd, 4), "pnl_pct": round(net_pnl_pct, 6),
                "fee_paid": round(fee_total, 4), "quote_asset": "USD",
                "exit_reason": reason, "paper": self.paper,
                "target_price": float(getattr(trade, "target_price", 0.0) or 0.0),
            })

            self._open.pop(symbol, None)
            logger.info("alpaca_position_closed", symbol=symbol, reason=reason, pnl_pct=round(net_pnl_pct, 5))
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
            logger.error("alpaca_finalize_trade_failed", trade_id=str(trade_id), error=str(e))
            return None
