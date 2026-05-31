"""Extended-hours (pre-market / after-hours) data add-on (Phase 4).

A FREE, no-subscription augmentation for US stocks: the Alpaca free IEX feed
under-serves the 04:00–09:30 and 16:00–20:00 ET windows, so this module fills the
gap using, in order of preference:

  1. yfinance with ``prepost=True`` — free, no API key, returns pre/post bars.
     Unofficial (can break if Yahoo changes their site) so it's a lazy, optional
     import behind a try/except.
  2. Finnhub ``/quote`` (free tier, uses the existing FINNHUB_API_KEY).

It NEVER replaces Alpaca/IBKR — it's purely additive. All functions degrade
gracefully to ``None`` when no provider is available, so callers can treat the
extended-hours price as best-effort.
"""
from __future__ import annotations

import asyncio
import time
from typing import Optional

import structlog

from backend.core.config import settings
from backend.core.market_hours import us_session_state

logger = structlog.get_logger(__name__)

FINNHUB_BASE = "https://finnhub.io/api/v1"


def _finnhub_key() -> str:
    key = getattr(settings, "FINNHUB_API_KEY", "") or ""
    return "" if "your_" in key.lower() else key


def _yf_prepost_price(symbol: str) -> Optional[float]:
    """Latest pre/post-market price via yfinance (sync; run in a thread).

    Returns None if yfinance isn't installed or returns nothing."""
    try:
        import yfinance as yf  # type: ignore
    except Exception:
        return None
    try:
        hist = yf.Ticker(symbol).history(period="1d", interval="1m", prepost=True)
        if hist is None or hist.empty:
            return None
        return float(hist["Close"].iloc[-1])
    except Exception:
        return None


async def _finnhub_quote_price(symbol: str) -> Optional[float]:
    key = _finnhub_key()
    if not key:
        return None
    import aiohttp
    url = f"{FINNHUB_BASE}/quote"
    try:
        timeout = aiohttp.ClientTimeout(total=5)
        async with aiohttp.ClientSession(timeout=timeout) as s:
            async with s.get(url, params={"symbol": symbol.upper(), "token": key}) as r:
                if r.status != 200:
                    return None
                data = await r.json(content_type=None)
                # 'c' = current price (Finnhub reflects extended-hours prints on free tier)
                px = float((data or {}).get("c", 0.0) or 0.0)
                return px if px > 0 else None
    except Exception:
        return None


async def get_extended_hours_quote(symbol: str) -> Optional[dict]:
    """Best-effort extended-hours quote for a US stock.

    Returns ``{"symbol", "price", "session", "is_extended", "source"}`` or None.
    Crypto trades 24/7 so there's no "extended hours" concept — returns None.
    """
    if not bool(getattr(settings, "EXTENDED_HOURS_DATA_ENABLED", True)):
        return None
    session = us_session_state()
    price: Optional[float] = None
    source: Optional[str] = None

    # Prefer yfinance prepost (free, key-less) — run the sync call off the loop.
    try:
        price = await asyncio.to_thread(_yf_prepost_price, symbol)
        if price is not None:
            source = "yfinance"
    except Exception:
        price = None

    if price is None:
        price = await _finnhub_quote_price(symbol)
        if price is not None:
            source = "finnhub"

    if price is None:
        return None
    return {
        "symbol": symbol.upper(),
        "price": price,
        "session": session,
        "is_extended": session in ("pre", "after", "overnight"),
        "source": source,
        "ts": time.time(),
    }
