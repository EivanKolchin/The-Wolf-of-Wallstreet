"""Binance USD-M futures adapter — the crypto-perp / short-capable venue for the TS-momentum sleeve.

Minimal signed-REST client (HMAC-SHA256) over aiohttp — no extra SDK. It exposes exactly the four
primitives the live router needs: get_price, get_equity, set_leverage, submit_notional. Market orders
support BOTH directions and reduce-only, so the sleeve's short leg works as designed.

SAFETY (mirrors the project's never-auto-wire-money posture):
  * Defaults to the Binance futures TESTNET endpoint. Mainnet is used ONLY when settings say
    PAPER_TRADING=false AND BINANCE_FUTURES_TESTNET=false AND real keys are present
    (settings.binance_futures_live_enabled()).
  * ``submit_notional`` refuses to place an order unless live is enabled — otherwise it logs the
    intended order and returns a simulated ack. So importing/constructing this never risks a fill.
  * Per-symbol leverage is capped by settings.BINANCE_FUTURES_LEVERAGE.

This file is real-order-capable but UNTESTED against the live API here (no keys); the operator must
validate on testnet first. The deterministic bits (signing, quantity math) are unit-tested with fakes.
"""
from __future__ import annotations

import hashlib
import hmac
import time
import urllib.parse
from typing import Optional

try:
    import aiohttp
except Exception:  # pragma: no cover
    aiohttp = None

try:
    import structlog
    logger = structlog.get_logger("binance_futures")
except Exception:  # pragma: no cover
    import logging
    logger = logging.getLogger("binance_futures")

from backend.core.config import settings

MAINNET = "https://fapi.binance.com"
TESTNET = "https://testnet.binancefuture.com"


def sign_query(params: dict, secret: str) -> str:
    """Return the urlencoded query string with an appended HMAC-SHA256 ``signature`` (Binance spec)."""
    qs = urllib.parse.urlencode(params)
    sig = hmac.new(secret.encode(), qs.encode(), hashlib.sha256).hexdigest()
    return f"{qs}&signature={sig}"


def _f(x) -> Optional[float]:
    """Tolerant float: Binance returns numerics as strings; '' / None / garbage → None."""
    try:
        v = float(x)
        return v if v > 0 else None
    except (TypeError, ValueError):
        return None


def quantity_from_notional(notional_usd: float, price: float, step: float = 0.001) -> float:
    """Convert a USD notional to a contract quantity, floored to the lot ``step`` (≥0)."""
    if price <= 0 or notional_usd <= 0:
        return 0.0
    raw = notional_usd / price
    return float(int(raw / step) * step)


class BinanceFuturesBroker:
    def __init__(self, *, paper: Optional[bool] = None, leverage: Optional[int] = None):
        self.key = (getattr(settings, "BINANCE_FUTURES_API_KEY", "") or "")
        self.secret = (getattr(settings, "BINANCE_FUTURES_SECRET", "") or "")
        self._live = settings.binance_futures_live_enabled() if paper is None else (not paper)
        self.base = MAINNET if self._live else TESTNET
        self.leverage = int(leverage if leverage is not None
                            else getattr(settings, "BINANCE_FUTURES_LEVERAGE", 2))
        self._levered: set[str] = set()

    def is_available(self) -> bool:
        return bool(self.key and self.secret and "your_" not in self.key.lower())

    @property
    def live(self) -> bool:
        return self._live and self.is_available()

    def _headers(self) -> dict:
        return {"X-MBX-APIKEY": self.key}

    async def _signed(self, method: str, path: str, params: dict):
        if aiohttp is None:
            raise RuntimeError("aiohttp required for Binance futures")
        params = {**params, "timestamp": int(time.time() * 1000), "recvWindow": 5000}
        url = f"{self.base}{path}?{sign_query(params, self.secret)}"
        timeout = aiohttp.ClientTimeout(total=10)
        async with aiohttp.ClientSession(timeout=timeout) as s:
            async with s.request(method, url, headers=self._headers()) as r:
                return await r.json()

    async def get_price(self, symbol: str) -> Optional[float]:
        if aiohttp is None:
            return None
        url = f"{self.base}/fapi/v1/ticker/price?symbol={symbol}"
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as s:
                async with s.get(url) as r:
                    return float((await r.json())["price"])
        except Exception as e:
            logger.warning("binance_price_failed", symbol=symbol, error=str(e)[:80])
            return None

    async def get_book_ticker(self, symbol: str) -> Optional[dict]:
        """Best bid/ask as {'bid': str, 'ask': str} — STRINGS from the API on purpose:
        they already conform to the symbol's tick size, so quoting them back as the
        limit price can never be rejected for price precision."""
        if aiohttp is None:
            return None
        url = f"{self.base}/fapi/v1/ticker/bookTicker?symbol={symbol}"
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as s:
                async with s.get(url) as r:
                    d = await r.json()
                    return {"bid": str(d["bidPrice"]), "ask": str(d["askPrice"])}
        except Exception as e:
            logger.warning("binance_book_ticker_failed", symbol=symbol, error=str(e)[:80])
            return None

    async def get_order(self, symbol: str, order_id) -> Optional[dict]:
        try:
            return await self._signed("GET", "/fapi/v1/order",
                                      {"symbol": symbol, "orderId": order_id})
        except Exception as e:
            logger.warning("binance_get_order_failed", symbol=symbol, error=str(e)[:80])
            return None

    async def cancel_order(self, symbol: str, order_id) -> Optional[dict]:
        try:
            return await self._signed("DELETE", "/fapi/v1/order",
                                      {"symbol": symbol, "orderId": order_id})
        except Exception as e:
            logger.warning("binance_cancel_failed", symbol=symbol, error=str(e)[:80])
            return None

    async def get_equity(self) -> Optional[float]:
        if not self.is_available():
            return None
        try:
            acc = await self._signed("GET", "/fapi/v2/account", {})
            return float(acc.get("totalWalletBalance", 0.0))
        except Exception as e:
            logger.warning("binance_equity_failed", error=str(e)[:80])
            return None

    async def set_leverage(self, symbol: str) -> None:
        if not self.live or symbol in self._levered:
            return
        try:
            await self._signed("POST", "/fapi/v1/leverage",
                               {"symbol": symbol, "leverage": self.leverage})
            self._levered.add(symbol)
        except Exception as e:
            logger.warning("binance_set_leverage_failed", symbol=symbol, error=str(e)[:80])

    async def _market_order(self, symbol: str, side: str, qty: float, reduce_only: bool) -> dict:
        params = {"symbol": symbol, "side": side, "type": "MARKET", "quantity": qty,
                  "newOrderRespType": "RESULT"}      # RESULT → response carries avgPrice
        if reduce_only:
            params["reduceOnly"] = "true"
        resp = await self._signed("POST", "/fapi/v1/order", params)
        return resp or {}

    async def submit_notional(self, symbol: str, notional_usd: float, *, reduce_only: bool = False,
                              style: str = "market", limit_timeout_s: float = 30.0,
                              poll_s: float = 2.0):
        """Order of |notional_usd| (BUY if >0, SELL if <0). Refuses to route a REAL order
        unless live is enabled — otherwise logs intent + returns a simulated ack.

        ``style="passive"`` (live only): post-only LIMIT joined to the near touch (bid for
        BUY, ask for SELL) — earns the spread instead of paying it, which matters exactly
        when leverage multiplies turnover. If unfilled after ``limit_timeout_s`` the
        remainder is cancelled and swept with a MARKET order, so the rebalance always
        completes; a post-only rejection (would-cross) falls straight through to MARKET.

        The result includes ``decision_price`` (mid at decision time) and, when available,
        ``fill_price``/``executed_qty`` so the caller can log realized slippage."""
        if abs(notional_usd) < 1e-9:
            return {"status": "skipped", "reason": "zero notional"}
        side = "BUY" if notional_usd > 0 else "SELL"
        price = await self.get_price(symbol)
        if not price:
            return {"status": "error", "reason": "no price"}
        qty = quantity_from_notional(abs(notional_usd), price)
        if qty <= 0:
            return {"status": "skipped", "reason": "below lot step"}
        if not self.live:
            logger.info("binance_paper_order", symbol=symbol, side=side, qty=qty,
                        notional=round(notional_usd, 2), reduce_only=reduce_only, style=style)
            return {"status": "paper", "symbol": symbol, "side": side, "qty": qty,
                    "style": style, "decision_price": price, "fill_price": price}
        await self.set_leverage(symbol)

        if style != "passive":
            try:
                resp = await self._market_order(symbol, side, qty, reduce_only)
                logger.info("binance_live_order", symbol=symbol, side=side, qty=qty,
                            resp_id=resp.get("orderId"))
                return {"status": "live", "style": "market", "decision_price": price,
                        "fill_price": _f(resp.get("avgPrice")),
                        "executed_qty": _f(resp.get("executedQty")), "resp": resp}
            except Exception as e:
                logger.error("binance_order_failed", symbol=symbol, error=str(e)[:120])
                return {"status": "error", "reason": str(e)[:120]}

        # ── passive: post-only limit at the near touch, timeout → market sweep ──
        book = await self.get_book_ticker(symbol)
        if not book:
            return await self.submit_notional(symbol, notional_usd, reduce_only=reduce_only,
                                              style="market")
        limit_px = book["bid"] if side == "BUY" else book["ask"]     # join, don't cross
        params = {"symbol": symbol, "side": side, "type": "LIMIT", "timeInForce": "GTX",
                  "quantity": qty, "price": limit_px}
        if reduce_only:
            params["reduceOnly"] = "true"
        try:
            resp = await self._signed("POST", "/fapi/v1/order", params)
        except Exception as e:
            logger.warning("binance_passive_submit_failed_fallback_market",
                           symbol=symbol, error=str(e)[:120])
            return await self.submit_notional(symbol, notional_usd, reduce_only=reduce_only,
                                              style="market")
        order_id = (resp or {}).get("orderId")
        if order_id is None or (resp or {}).get("status") == "EXPIRED":
            # GTX rejected (would cross) → the market has moved through us; take it.
            return await self.submit_notional(symbol, notional_usd, reduce_only=reduce_only,
                                              style="market")

        import asyncio
        deadline = time.time() + max(1.0, float(limit_timeout_s))
        last = resp
        while time.time() < deadline:
            await asyncio.sleep(max(0.2, float(poll_s)))
            od = await self.get_order(symbol, order_id)
            if od:
                last = od
                if od.get("status") == "FILLED":
                    logger.info("binance_passive_filled", symbol=symbol, side=side, qty=qty)
                    return {"status": "live", "style": "passive", "decision_price": price,
                            "fill_price": _f(od.get("avgPrice")) or _f(limit_px),
                            "executed_qty": _f(od.get("executedQty")), "resp": od}
                if od.get("status") in ("CANCELED", "EXPIRED", "REJECTED"):
                    break
        # timeout / dead order: cancel the remainder and sweep it with a market order
        await self.cancel_order(symbol, order_id)
        done_qty = _f((last or {}).get("executedQty")) or 0.0
        remain = max(0.0, qty - done_qty)
        if remain <= 0:
            return {"status": "live", "style": "passive", "decision_price": price,
                    "fill_price": _f((last or {}).get("avgPrice")) or _f(limit_px),
                    "executed_qty": done_qty, "resp": last}
        try:
            sweep = await self._market_order(symbol, side, remain, reduce_only)
            logger.info("binance_passive_timeout_market_sweep", symbol=symbol,
                        filled_passive=done_qty, swept=remain)
            return {"status": "live", "style": "passive+sweep", "decision_price": price,
                    "fill_price": _f(sweep.get("avgPrice")),
                    "executed_qty": done_qty + (_f(sweep.get("executedQty")) or 0.0),
                    "resp": sweep}
        except Exception as e:
            logger.error("binance_sweep_failed", symbol=symbol, error=str(e)[:120])
            return {"status": "error", "reason": f"sweep failed: {str(e)[:100]}",
                    "executed_qty": done_qty}
