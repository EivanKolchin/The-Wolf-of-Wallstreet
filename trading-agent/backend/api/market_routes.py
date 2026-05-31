"""Market-data proxy endpoints used by the frontend chart.

Routes crypto symbols to Binance and US-stock symbols to Alpaca Market Data so
the frontend can fetch klines / depth / trades with the same interface regardless
of asset class. Returns Binance-shaped klines arrays for both backends so the
chart doesn't need branching logic.
"""
from __future__ import annotations

import asyncio
import json
import time
import urllib.request
import urllib.parse

import aiohttp
import structlog
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from backend.core import ledger
from backend.core.config import settings
from backend.core import universe as _universe
from backend.core.resilient_http import make_resilient_session

logger = structlog.get_logger(__name__)
router = APIRouter()

ALPACA_DATA = "https://data.alpaca.markets/v2"
ALPACA_WS = "wss://stream.data.alpaca.markets/v2/iex"
FINNHUB_BASE = "https://finnhub.io/api/v1"

# Cycle 21: Finnhub fallback for the historical stock bars endpoint. Fires
# ONLY when the Alpaca request failed at the transport level (connection,
# DNS, HTTP 5xx). Never on 401 — that's a credential problem and a fallback
# would mask the real cause. Silent no-op when FINNHUB_API_KEY is absent.
_FINNHUB_RESOLUTION = {
    "1m": "1", "5m": "5", "15m": "15", "30m": "30",
    "1h": "60", "4h": "240", "1d": "D", "1w": "W", "1M": "M",
    "1Min": "1", "5Min": "5", "15Min": "15", "30Min": "30",
    "1Hour": "60", "4Hour": "240", "1Day": "D",
}

# Alpaca's timeframe vocabulary differs from Binance's. Map both to Alpaca's.
_ALPACA_TF_MAP = {
    "1m": "1Min", "5m": "5Min", "15m": "15Min", "30m": "30Min",
    "1h": "1Hour", "2h": "2Hour", "4h": "4Hour",
    "1d": "1Day", "1w": "1Week", "1M": "1Month",
    # Pass through if user already gave the Alpaca form
    "1Min": "1Min", "5Min": "5Min", "15Min": "15Min", "30Min": "30Min",
    "1Hour": "1Hour", "4Hour": "4Hour", "1Day": "1Day",
}


def _is_us_stock(symbol: str) -> bool:
    """Treat anything not in CRYPTO_SYMBOLS as a US stock — covers the configured
    universe and any tickers that may join later without code edits."""
    try:
        return _universe.asset_class_of(symbol) == "us_stock"
    except Exception:
        s = (symbol or "").upper()
        return s in set(_universe.STOCK_UNDERLYINGS)


def _alpaca_headers() -> dict:
    key = getattr(settings, "ALPACA_API_KEY", "") or ""
    secret = getattr(settings, "ALPACA_SECRET_KEY", "") or getattr(settings, "ALPACA_SECRET", "") or ""
    return {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}


def _alpaca_available() -> bool:
    h = _alpaca_headers()
    return bool(h["APCA-API-KEY-ID"] and h["APCA-API-SECRET-KEY"] and "your_" not in h["APCA-API-KEY-ID"].lower())


# How far back to ask Alpaca for, per timeframe — must cover `limit` bars + IEX
# 15-min delay slack. Tuned so 1m doesn't blow the 10k-bar cap and 1d covers years.
_ALPACA_LOOKBACK_DAYS = {
    "1Min": 7, "5Min": 30, "15Min": 90, "30Min": 180,
    "1Hour": 365, "2Hour": 730, "4Hour": 1095,
    "1Day": 3650, "1Week": 7300, "1Month": 7300,
}


def _alpaca_start_iso(tf: str, end_ms: int | None) -> tuple[str, str | None]:
    """Compute (start_iso, end_iso) for an Alpaca bars request.

    Alpaca's free IEX feed has a ~15-min delay; setting `end = now - 16min`
    avoids "empty most-recent" surprises. Caller may override `end_ms` to fetch
    older history (used by the chart's load-more scroll).
    """
    import datetime as _dt
    lookback = _ALPACA_LOOKBACK_DAYS.get(tf, 30)
    if end_ms:
        end_dt = _dt.datetime.utcfromtimestamp(end_ms / 1000.0)
    else:
        end_dt = _dt.datetime.utcnow() - _dt.timedelta(minutes=16)
    start_dt = end_dt - _dt.timedelta(days=lookback)
    end_iso = end_dt.replace(microsecond=0).isoformat() + "Z" if end_ms else None
    start_iso = start_dt.replace(microsecond=0).isoformat() + "Z"
    return start_iso, end_iso


def _finnhub_available() -> bool:
    key = getattr(settings, "FINNHUB_API_KEY", "") or ""
    return bool(key and "your_" not in key.lower())


async def _finnhub_bars_to_klines(symbol: str, interval: str, limit: int,
                                    end_ms: int | None = None) -> list:
    """Cycle 21: free-tier Finnhub /stock/candle as a transport-error fallback
    for Alpaca. Same Binance-shaped output so the frontend doesn't branch.
    Returns ``[]`` on any failure (silent) — never raises."""
    if not _finnhub_available():
        return []
    resolution = _FINNHUB_RESOLUTION.get(interval, "5")
    # Finnhub takes unix seconds. Mirror the Alpaca lookback window so the
    # resulting dataset size is comparable.
    import datetime as _dt
    tf = _ALPACA_TF_MAP.get(interval, "5Min")
    lookback_days = _ALPACA_LOOKBACK_DAYS.get(tf, 30)
    end_dt = _dt.datetime.utcfromtimestamp(end_ms / 1000.0) if end_ms else _dt.datetime.utcnow()
    start_dt = end_dt - _dt.timedelta(days=lookback_days)
    params = {
        "symbol": symbol.upper(),
        "resolution": resolution,
        "from": int(start_dt.timestamp()),
        "to": int(end_dt.timestamp()),
        "token": getattr(settings, "FINNHUB_API_KEY", "") or "",
    }
    url = f"{FINNHUB_BASE}/stock/candle"
    t0 = time.monotonic()
    try:
        async with make_resilient_session(timeout_total=10.0) as s:
            async with s.get(url, params=params) as r:
                data = await r.json(content_type=None)
                ok = (r.status == 200) and (data or {}).get("s") == "ok"
                ledger.log_api_call("finnhub", "GET", url, status=r.status, ok=ok,
                                    latency_ms=(time.monotonic() - t0) * 1000)
                if not ok:
                    return []
                ts = data.get("t", []) or []
                opens = data.get("o", []) or []
                highs = data.get("h", []) or []
                lows = data.get("l", []) or []
                closes = data.get("c", []) or []
                vols = data.get("v", []) or []
                out = []
                for i in range(min(len(ts), int(limit))):
                    ms = int(ts[i]) * 1000
                    out.append([
                        ms, str(opens[i]), str(highs[i]), str(lows[i]), str(closes[i]),
                        str(vols[i]), ms, "0", 0, "0", "0", "0",
                    ])
                return out
    except Exception as e:
        ledger.log_api_call("finnhub", "GET", url, ok=False,
                            latency_ms=(time.monotonic() - t0) * 1000, note=str(e)[:120])
        return []


async def _alpaca_bars_to_klines(symbol: str, interval: str, limit: int,
                                  end_ms: int | None = None) -> list | dict:
    """Fetch bars from Alpaca and shape them like Binance klines so the
    frontend renderer doesn't have to branch.
    Binance row: [openTime, open, high, low, close, volume, closeTime, ...]
    On Alpaca error returns a dict (``{error,status,detail,bars:[]}``) so the
    frontend can surface the cause; on success returns a plain list.
    """
    tf = _ALPACA_TF_MAP.get(interval, "5Min")
    start_iso, end_iso = _alpaca_start_iso(tf, end_ms)
    url = f"{ALPACA_DATA}/stocks/{symbol.upper()}/bars"
    params = {
        "timeframe": tf,
        "limit": str(min(int(limit), 10000)),
        "adjustment": "raw",
        "feed": "iex",
        "sort": "asc",
        "start": start_iso,
    }
    if end_iso:
        params["end"] = end_iso

    t0 = time.monotonic()
    try:
        async with make_resilient_session(timeout_total=10.0) as s:
            async with s.get(url, headers=_alpaca_headers(), params=params) as r:
                data = await r.json(content_type=None)
                ok = (r.status == 200)
                ledger.log_api_call("alpaca", "GET", url, status=r.status, ok=ok,
                                    latency_ms=(time.monotonic() - t0) * 1000)
                if not ok:
                    return {
                        "error": "alpaca_http_error",
                        "status": r.status,
                        "detail": (data if isinstance(data, dict) else {"raw": str(data)[:300]}),
                        "bars": [],
                    }
                bars = (data or {}).get("bars") or []
                out = []
                import datetime as _dt
                for b in bars:
                    # b: {"t":"2025-01-01T13:30:00Z","o":..,"h":..,"l":..,"c":..,"v":..}
                    try:
                        ts_ms = int(_dt.datetime.fromisoformat(b["t"].replace("Z", "+00:00")).timestamp() * 1000)
                    except Exception:
                        ts_ms = 0
                    out.append([
                        ts_ms,
                        str(b.get("o", 0)), str(b.get("h", 0)), str(b.get("l", 0)), str(b.get("c", 0)),
                        str(b.get("v", 0)),
                        ts_ms,  # closeTime ≈ same; frontend only reads index 0..5
                        "0", 0, "0", "0", "0",
                    ])
                return out
    except Exception as e:
        ledger.log_api_call("alpaca", "GET", url, ok=False,
                            latency_ms=(time.monotonic() - t0) * 1000, note=str(e)[:120])
        # Cycle 21: transport failure → try Finnhub silently. Only fires when
        # FINNHUB_API_KEY is set; otherwise we surface the original error.
        if _finnhub_available():
            try:
                bars = await _finnhub_bars_to_klines(symbol, interval, limit, end_ms=end_ms)
                if bars:
                    logger.info("alpaca_failed_finnhub_fallback_ok",
                                  symbol=symbol, bars=len(bars), alpaca_error=str(e)[:120])
                    return bars
            except Exception as fe:
                logger.warning("finnhub_fallback_failed", error=str(fe)[:120])
        return {"error": "alpaca_request_failed", "detail": str(e)[:300], "bars": []}


def _binance_klines(symbol: str, interval: str, limit: int,
                     start_ms: int | None = None, end_ms: int | None = None) -> list:
    try:
        q = {"symbol": symbol.upper(), "interval": interval, "limit": int(limit)}
        if start_ms is not None:
            q["startTime"] = int(start_ms)
        if end_ms is not None:
            q["endTime"] = int(end_ms)
        qs = urllib.parse.urlencode(q)
        url = f"https://api.binance.com/api/v3/klines?{qs}"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=8) as response:
            return json.loads(response.read().decode())
    except Exception:
        return []


@router.get("/api/market/klines")
async def get_market_klines(symbol: str, interval: str, limit: int = 100,
                             startTime: int | None = None, endTime: int | None = None):
    if _is_us_stock(symbol):
        if not _alpaca_available():
            return {"error": "alpaca_credentials_missing", "bars": []}
        return await _alpaca_bars_to_klines(symbol, interval, limit, end_ms=endTime)
    return _binance_klines(symbol, interval, limit, start_ms=startTime, end_ms=endTime)


# Quote endpoint — single most-recent price (used for ticker refreshes).
@router.get("/api/market/quote")
async def get_market_quote(symbol: str):
    if _is_us_stock(symbol):
        if not _alpaca_available():
            return {"price": None, "error": "alpaca_credentials_missing"}
        url = f"{ALPACA_DATA}/stocks/{symbol.upper()}/trades/latest"
        t0 = time.monotonic()
        try:
            async with make_resilient_session(timeout_total=5.0) as s:
                async with s.get(url, headers=_alpaca_headers()) as r:
                    data = await r.json(content_type=None)
                    ledger.log_api_call("alpaca", "GET", url, status=r.status, ok=(r.status == 200),
                                        latency_ms=(time.monotonic() - t0) * 1000)
                    return {"price": float(((data or {}).get("trade") or {}).get("p", 0.0) or 0.0)}
        except Exception as e:
            ledger.log_api_call("alpaca", "GET", url, ok=False,
                                latency_ms=(time.monotonic() - t0) * 1000, note=str(e)[:120])
            return {"price": None, "error": str(e)[:120]}
    # Crypto: ask Binance for the latest trade
    try:
        qs = urllib.parse.urlencode({"symbol": symbol.upper()})
        url = f"https://api.binance.com/api/v3/ticker/price?{qs}"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read().decode())
            return {"price": float(data.get("price", 0.0))}
    except Exception as e:
        return {"price": None, "error": str(e)[:120]}


# Universe — used by the frontend symbol dropdown. The model registry knows the
# full set of cryptos (8) while universe.CRYPTO_SYMBOLS is just the trading subset;
# the chart should expose anything the user might want to view.
@router.get("/api/market/universe")
async def get_market_universe():
    stocks = list(_universe.STOCK_UNDERLYINGS)
    try:
        from backend.agents.improved_model import SYMBOLS as MODEL_SYMBOLS
        # MODEL_SYMBOLS holds crypto (ids 0..7) AND the 5 US stocks (ids 8..12);
        # the stocks must NOT leak into the "crypto" bucket, so filter them out.
        cryptos = [s for s in MODEL_SYMBOLS if s not in stocks]
    except Exception:
        cryptos = list(_universe.CRYPTO_SYMBOLS)
    return {
        "crypto": cryptos,
        "stocks": stocks,
        "us_exchange": getattr(_universe, "US_EXCHANGE", {}),
    }


@router.get("/api/market/depth")
async def get_market_depth(symbol: str, limit: int = 50):
    # Alpaca has no public L2 endpoint without a paid feed; return empty for stocks.
    if _is_us_stock(symbol):
        return {"bids": [], "asks": []}
    try:
        qs = urllib.parse.urlencode({"symbol": symbol.upper(), "limit": int(limit)})
        url = f"https://api.binance.com/api/v3/depth?{qs}"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=8) as response:
            return json.loads(response.read().decode())
    except Exception:
        return {"bids": [], "asks": []}


@router.get("/api/market/trades")
async def get_market_trades(symbol: str, limit: int = 50):
    if _is_us_stock(symbol):
        return []   # stocks: not needed for the chart's recent-trades widget
    try:
        qs = urllib.parse.urlencode({"symbol": symbol.upper(), "limit": int(limit)})
        url = f"https://api.binance.com/api/v3/trades?{qs}"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=8) as response:
            return json.loads(response.read().decode())
    except Exception:
        return []


def _compute_kline_stats(klines: list) -> dict:
    """Volatility / volume / price-change from Binance-shaped klines.
    Each row: [openTime, open, high, low, close, volume, ...]."""
    out = {"last_price": None, "volume": 0.0, "volatility_pct": None,
           "price_change_pct": None, "bars": 0, "spark": []}
    try:
        closes = [float(r[4]) for r in klines if r and len(r) > 5]
        vols = [float(r[5]) for r in klines if r and len(r) > 5]
        if not closes:
            return out
        out["bars"] = len(closes)
        out["last_price"] = closes[-1]
        out["volume"] = float(sum(vols))
        # Downsample to ~32 points for a lightweight frontend sparkline.
        if len(closes) > 32:
            step = len(closes) / 32.0
            out["spark"] = [round(closes[int(i * step)], 6) for i in range(32)]
        else:
            out["spark"] = [round(c, 6) for c in closes]
        if len(closes) >= 2:
            rets = [(closes[i] - closes[i - 1]) / closes[i - 1]
                    for i in range(1, len(closes)) if closes[i - 1] > 0]
            if rets:
                mean = sum(rets) / len(rets)
                var = sum((x - mean) ** 2 for x in rets) / len(rets)
                out["volatility_pct"] = (var ** 0.5) * 100.0
            if closes[0] > 0:
                out["price_change_pct"] = (closes[-1] - closes[0]) / closes[0] * 100.0
    except Exception:
        pass
    return out


async def _overview_for_symbol(symbol: str, interval: str, limit: int) -> dict:
    """Per-asset overview row: stats + session + (stock) extended-hours quote."""
    asset_class = _universe.asset_class_of(symbol)
    is_stock = asset_class == "us_stock"
    try:
        if is_stock:
            kl = await _alpaca_bars_to_klines(symbol, interval, limit) if _alpaca_available() else []
            if not kl:
                kl = await _finnhub_bars_to_klines(symbol, interval, limit)
        else:
            kl = _binance_klines(symbol, interval, limit)
    except Exception:
        kl = []
    stats = _compute_kline_stats(kl if isinstance(kl, list) else (kl.get("bars") if isinstance(kl, dict) else []))

    from backend.core.market_hours import us_session_state
    session = us_session_state() if is_stock else "open"

    extended = None
    if is_stock and session in ("pre", "after", "overnight"):
        try:
            from backend.data.extended_hours_feed import get_extended_hours_quote
            extended = await get_extended_hours_quote(symbol)
        except Exception:
            extended = None

    return {
        "symbol": symbol.upper(),
        "asset_class": asset_class,
        "session": session,
        **stats,
        "extended_hours": extended,
    }


@router.get("/api/assets/overview")
async def get_assets_overview(interval: str = "15m", limit: int = 96):
    """Cross-asset snapshot powering the All Assets page: per-symbol last price,
    volume, volatility, price change, market session, any open position, and
    (for stocks in extended hours) a best-effort pre/after-hours quote."""
    symbols: list[str] = []
    try:
        from backend.agents.improved_model import SYMBOLS as MODEL_SYMBOLS
        stocks = set(_universe.STOCK_UNDERLYINGS)
        symbols = [s for s in MODEL_SYMBOLS if s not in stocks] + list(_universe.STOCK_UNDERLYINGS)
    except Exception:
        symbols = list(_universe.CRYPTO_SYMBOLS) + list(_universe.STOCK_UNDERLYINGS)

    rows = await asyncio.gather(
        *[_overview_for_symbol(s, interval, limit) for s in symbols],
        return_exceptions=True,
    )
    assets = [r for r in rows if isinstance(r, dict)]

    # Attach open positions from the live portfolio state (same source the
    # dashboard ledger uses), keyed by symbol.
    positions_by_symbol: dict[str, dict] = {}
    try:
        from backend.memory.redis_client import get_redis
        redis_client = await get_redis()
        live_state_str = await redis_client.get("portfolio:live_state")
        if live_state_str:
            live_state = json.loads(live_state_str)
            for pos in (live_state.get("positions") or []):
                sym = str(pos.get("symbol", "")).upper()
                if sym:
                    positions_by_symbol[sym] = pos
    except Exception as e:
        logger.warning("assets_overview_positions_failed", error=str(e))

    for a in assets:
        a["position"] = positions_by_symbol.get(a["symbol"])

    return {"assets": assets, "count": len(assets), "interval": interval}


# =============================================================================
# Alpaca live-trade WebSocket proxy
# =============================================================================
#
# Alpaca's free IEX streaming feed allows ONE concurrent auth'd connection per
# account. We hold that single connection on the backend, multiplex many
# subscribers, and forward trade ticks to whichever browser tab is listening.
#
# Browser connects to:  ws://127.0.0.1:8000/ws/stocks?symbol=AMD
# We push JSON messages of shape:
#   {"type":"trade","symbol":"AMD","price":123.45,"size":100,"ts_ms":1716960000000}
# The frontend ticks the last candle's high/low/close exactly like the Binance
# aggTrade path does for crypto.

class _AlpacaStreamHub:
    """Single backend-side connection to Alpaca; broadcasts trades to listeners."""
    def __init__(self) -> None:
        self._listeners: dict[str, set[asyncio.Queue]] = {}  # symbol -> queues
        self._task: asyncio.Task | None = None
        self._lock = asyncio.Lock()
        self._subscribed_symbols: set[str] = set()
        self._stop = False
        self._ws_send: asyncio.Queue[dict] | None = None   # outbound auth/subscribe msgs

    async def subscribe(self, symbol: str) -> asyncio.Queue:
        """Register a listener for `symbol`. Returns a Queue that yields dicts.

        Subscribes the upstream connection to BOTH trades AND quotes:
        quotes update many times per second (bid/ask), so they drive smooth
        chart movement between sparse trade prints.
        """
        symbol = symbol.upper()
        q: asyncio.Queue = asyncio.Queue(maxsize=2000)   # quotes are chatty
        async with self._lock:
            self._listeners.setdefault(symbol, set()).add(q)
            need_subscribe = symbol not in self._subscribed_symbols
            self._subscribed_symbols.add(symbol)
            if self._task is None or self._task.done():
                self._stop = False
                self._ws_send = asyncio.Queue()
                self._task = asyncio.create_task(self._run())
            if need_subscribe and self._ws_send is not None:
                try:
                    self._ws_send.put_nowait({"action": "subscribe",
                                                "trades": [symbol], "quotes": [symbol]})
                except Exception:
                    pass
        return q

    async def unsubscribe(self, symbol: str, q: asyncio.Queue) -> None:
        symbol = symbol.upper()
        async with self._lock:
            listeners = self._listeners.get(symbol)
            if listeners and q in listeners:
                listeners.discard(q)
                if not listeners:
                    self._listeners.pop(symbol, None)
                    self._subscribed_symbols.discard(symbol)
                    if self._ws_send is not None:
                        try:
                            self._ws_send.put_nowait({"action": "unsubscribe",
                                                        "trades": [symbol], "quotes": [symbol]})
                        except Exception:
                            pass
            # No subscribers anywhere → tear the upstream connection down.
            if not self._listeners and self._task and not self._task.done():
                self._stop = True
                self._task.cancel()

    async def _broadcast(self, symbol: str, msg: dict) -> None:
        listeners = list(self._listeners.get(symbol, ()))
        for q in listeners:
            try:
                q.put_nowait(msg)
            except asyncio.QueueFull:
                pass  # slow consumer — drop

    async def _run(self) -> None:
        if not _alpaca_available():
            logger.warning("alpaca_stream_skip_no_credentials")
            return
        backoff = 1.0
        while not self._stop:
            try:
                async with make_resilient_session(timeout_total=30.0) as sess:
                    async with sess.ws_connect(ALPACA_WS, heartbeat=20) as ws:
                        # Alpaca v2 IEX expects key/secret immediately after connect.
                        await ws.send_json({
                            "action": "auth",
                            "key": getattr(settings, "ALPACA_API_KEY", ""),
                            "secret": (getattr(settings, "ALPACA_SECRET_KEY", "") or
                                       getattr(settings, "ALPACA_SECRET", "") or ""),
                        })
                        # Subscribe to whatever symbols are currently registered
                        # for BOTH trades and quotes — quotes provide the dense
                        # bid/ask stream that smooths chart movement.
                        async with self._lock:
                            syms = list(self._subscribed_symbols)
                        if syms:
                            await ws.send_json({"action": "subscribe",
                                                  "trades": syms, "quotes": syms})

                        backoff = 1.0
                        # Pump outgoing control messages + incoming data concurrently.
                        sender_task = asyncio.create_task(self._sender(ws))
                        try:
                            async for msg in ws:
                                if msg.type == aiohttp.WSMsgType.TEXT:
                                    payload = json.loads(msg.data)
                                    for ev in (payload if isinstance(payload, list) else [payload]):
                                        await self._handle_event(ev)
                                elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                                    break
                        finally:
                            sender_task.cancel()
            except asyncio.CancelledError:
                return
            except Exception as e:
                logger.warning("alpaca_stream_disconnect", error=str(e)[:200])
            if self._stop:
                break
            await asyncio.sleep(min(backoff, 30.0))
            backoff *= 2

    async def _sender(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        assert self._ws_send is not None
        try:
            while True:
                out = await self._ws_send.get()
                await ws.send_json(out)
        except asyncio.CancelledError:
            return
        except Exception:
            return

    async def _handle_event(self, ev: dict) -> None:
        t = ev.get("T")
        if t == "t":
            # trade: {"T":"t","S":"AMD","p":..,"s":..,"t":"2025-01-01T..."}
            sym = (ev.get("S") or "").upper()
            price = ev.get("p")
            size = ev.get("s")
            ts_iso = ev.get("t") or ""
            try:
                import datetime as _dt
                ts_ms = int(_dt.datetime.fromisoformat(ts_iso.replace("Z", "+00:00")).timestamp() * 1000)
            except Exception:
                ts_ms = int(time.time() * 1000)
            await self._broadcast(sym, {
                "type": "trade", "symbol": sym, "price": float(price or 0.0),
                "size": float(size or 0.0), "ts_ms": ts_ms,
            })
        elif t == "q":
            # quote: {"T":"q","S":"AMD","bp":..,"ap":..,"bs":..,"as":..,"t":"..."}
            # Quotes fire many times per second — they drive smooth chart
            # movement between sparse trade prints. Frontend uses mid-price.
            sym = (ev.get("S") or "").upper()
            bid = ev.get("bp"); ask = ev.get("ap")
            try:
                bid_f = float(bid or 0.0); ask_f = float(ask or 0.0)
            except Exception:
                return
            if bid_f <= 0 or ask_f <= 0:
                return
            mid = (bid_f + ask_f) / 2.0
            ts_iso = ev.get("t") or ""
            try:
                import datetime as _dt
                ts_ms = int(_dt.datetime.fromisoformat(ts_iso.replace("Z", "+00:00")).timestamp() * 1000)
            except Exception:
                ts_ms = int(time.time() * 1000)
            await self._broadcast(sym, {
                "type": "quote", "symbol": sym,
                "bid": bid_f, "ask": ask_f, "mid": mid, "ts_ms": ts_ms,
            })
        elif t == "subscription":
            logger.info("alpaca_stream_subscribed",
                         trades=ev.get("trades"), quotes=ev.get("quotes"))
        elif t == "error":
            logger.warning("alpaca_stream_error", code=ev.get("code"), msg=ev.get("msg"))


_hub = _AlpacaStreamHub()


@router.websocket("/ws/stocks")
async def alpaca_stocks_ws(websocket: WebSocket, symbol: str):
    """Per-tab subscriber socket. Browser opens this with ?symbol=AMD and gets a
    JSON stream of trade ticks until disconnect."""
    await websocket.accept()
    symbol = (symbol or "").upper()
    if not symbol or not _is_us_stock(symbol):
        await websocket.send_json({"type": "error", "detail": "unknown_or_non_stock_symbol", "symbol": symbol})
        await websocket.close()
        return
    if not _alpaca_available():
        await websocket.send_json({"type": "error", "detail": "alpaca_credentials_missing"})
        await websocket.close()
        return

    q = await _hub.subscribe(symbol)
    sender_done = asyncio.Event()

    async def pump():
        try:
            while not sender_done.is_set():
                msg = await q.get()
                await websocket.send_json(msg)
        except WebSocketDisconnect:
            sender_done.set()
        except Exception:
            sender_done.set()

    pump_task = asyncio.create_task(pump())
    try:
        # Drain incoming control frames (browser may send subscribe/unsubscribe later).
        while True:
            data = await websocket.receive_text()
            try:
                ctl = json.loads(data)
            except Exception:
                continue
            # Allow the browser to switch the streamed symbol on the fly.
            if ctl.get("action") == "switch":
                new_sym = (ctl.get("symbol") or "").upper()
                if new_sym and _is_us_stock(new_sym) and new_sym != symbol:
                    await _hub.unsubscribe(symbol, q)
                    symbol = new_sym
                    q = await _hub.subscribe(symbol)
    except WebSocketDisconnect:
        pass
    finally:
        sender_done.set()
        pump_task.cancel()
        await _hub.unsubscribe(symbol, q)
