import asyncio
import json
import time
from collections import deque
from typing import Callable, Coroutine, Optional, Dict, List

import pandas as pd
import websockets
from structlog import get_logger

log = get_logger("market_feed")

class OHLCVBuffer:
    def __init__(self, max_len: int = 500):
        self.max_len = max_len
        self.buffers: Dict[str, deque] = {}
        self.locks: Dict[str, asyncio.Lock] = {}

    async def add_kline(self, symbol: str, kline: dict):
        if symbol not in self.buffers:
            self.buffers[symbol] = deque(maxlen=self.max_len)
            self.locks[symbol] = asyncio.Lock()
        
        async with self.locks[symbol]:
            self.buffers[symbol].append({
                "timestamp": kline["timestamp"],
                "open": float(kline["open"]),
                "high": float(kline["high"]),
                "low": float(kline["low"]),
                "close": float(kline["close"]),
                "volume": float(kline["volume"]),
            })

    async def get_dataframe(self, symbol: str) -> pd.DataFrame:
        if symbol not in self.locks:
            return pd.DataFrame()

        async with self.locks[symbol]:
            data = list(self.buffers[symbol])
            
        if not data:
            return pd.DataFrame()
            
        df = pd.DataFrame(data)
        # convert timestamp back to datetime for easy usage
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("datetime", inplace=True)
        return df


class BinanceMarketFeed:
    BINANCE_WS_URL = "wss://stream.binance.com:9443/ws"

    def __init__(
        self,
        symbols: List[str],
        on_kline: Optional[Callable[[str, dict], Coroutine]] = None,
        on_orderbook: Optional[Callable[[str, dict], Coroutine]] = None,
        on_trade: Optional[Callable[[str, dict], Coroutine]] = None,
    ):
        self.symbols = [s.lower() for s in symbols]
        
        async def noop(*args, **kwargs): pass
        
        self.on_kline = on_kline or noop
        self.on_orderbook = on_orderbook or noop
        self.on_trade = on_trade or noop

        self.running = False
        self._ws = None
        self._task = None
        self._watchdog_task = None
        
        self.last_message_time = time.time()

        # Data stores
        self._closed_klines: Dict[str, deque] = {s.upper(): deque(maxlen=300) for s in self.symbols}
        self._orderbooks: Dict[str, dict] = {s.upper(): None for s in self.symbols}
        self._recent_trades: Dict[str, deque] = {s.upper(): deque(maxlen=100) for s in self.symbols}
        
    async def _fetch_historical(self) -> None:
        try:
            import requests
            import asyncio
            def fetch():
                for symbol in self.symbols:
                    url = f"https://api.binance.us/api/v3/klines?symbol={symbol.upper()}&interval=5m&limit=100"
                    resp = requests.get(url, timeout=10)
                    if resp.status_code == 200:
                        klines = resp.json()
                        for k in klines:
                            self._closed_klines[symbol.upper()].append({
                                "timestamp": int(k[0]),
                                "open": float(k[1]),
                                "high": float(k[2]),
                                "low": float(k[3]),
                                "close": float(k[4]),
                                "volume": float(k[5])
                            })
            await asyncio.to_thread(fetch)
            log.info("Historical klines prefetched")
        except Exception as e:
            log.error("error_fetching_historical", error=str(e))

    async def start(self) -> None:
        self.running = True
        await self._fetch_historical()
        self._task = asyncio.create_task(self._run_loop())
        self._watchdog_task = asyncio.create_task(self._watchdog())
        log.info("Market feed started", symbols=self.symbols)

    async def stop(self) -> None:
        self.running = False
        if self._task:
            self._task.cancel()
        if self._watchdog_task:
            self._watchdog_task.cancel()
        if self._ws:
            await self._ws.close()
        log.info("Market feed stopped")

    async def get_klines(self, symbol: str) -> List[dict]:
        return list(self._closed_klines.get(symbol.upper(), []))

    async def get_dataframe(self, symbol: str) -> pd.DataFrame:
        data = list(self._closed_klines.get(symbol.upper(), []))
        if not data:
            return pd.DataFrame()
        df = pd.DataFrame(data)
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("datetime", inplace=True)
        return df

    async def get_orderbook(self, symbol: str) -> Optional[dict]:
        return self._orderbooks.get(symbol.upper())

    async def get_recent_trades(self, symbol: str, n: int = 100) -> List[dict]:
        trades = self._recent_trades.get(symbol.upper(), [])
        return list(trades)[-n:]

    def _get_subscription_payload(self) -> str:
        streams = []
        for sym in self.symbols:
            streams.append(f"{sym}@kline_5m")
            streams.append(f"{sym}@depth20@100ms")
            streams.append(f"{sym}@aggTrade")
        
        req = {
            "method": "SUBSCRIBE",
            "params": streams,
            "id": int(time.time() * 1000)
        }
        return json.dumps(req)

    async def _handle_message(self, message: str):
        self.last_message_time = time.time()
        
        try:
            data = json.loads(message)
            if "data" in data and "stream" in data:
                data = data["data"]
        except Exception as e:
            log.error("Failed to parse message", error=str(e))
            return

        # Ignore subscription confirmations
        if "result" in data and "id" in data:
            return

        stream: str = data.get("e", "")
        
        # Orderbook uses lastUpdateId natively, without 'e' event
        if "lastUpdateId" in data and "bids" in data:
            # We must parse symbol from stream in combined mode, or infer from topic 
            # In single stream mode without combined connection, it doesn't give a symbol.
            # We'll just listen combined or look in the message if it has 's'
            symbol = data.get("s", "").upper() # if available
            if not symbol:
                return # Can't handle this correctly without knowing the symbol natively
                
            book = {
                "bids": [[float(p), float(q)] for p, q in data.get("bids", [])],
                "asks": [[float(p), float(q)] for p, q in data.get("asks", [])],
                "timestamp": int(time.time() * 1000) # Fallback timestamp
            }
            if symbol in self._orderbooks:
                self._orderbooks[symbol] = book
                await self.on_orderbook(symbol, book)
            return

        symbol = data.get("s", "").upper()
        if not symbol:
            return

        if stream == "kline":
            k = data["k"]
            kline = {
                "open": float(k["o"]),
                "high": float(k["h"]),
                "low": float(k["l"]),
                "close": float(k["c"]),
                "volume": float(k["v"]),
                "is_closed": bool(k["x"]),
                "timestamp": int(k["t"])
            }
            if kline["is_closed"]:
                if symbol in self._closed_klines:
                    self._closed_klines[symbol].append(kline)
            await self.on_kline(symbol, kline)
            
        elif stream == "aggTrade":
            trade = {
                "price": float(data["p"]),
                "qty": float(data["q"]),
                "is_buyer_maker": bool(data["m"]),
                "timestamp": int(data["T"])
            }
            if symbol in self._recent_trades:
                self._recent_trades[symbol].append(trade)
            await self.on_trade(symbol, trade)

        # Handle depth updates if stream = depthUpdate (not partial book)
        elif stream == "depthUpdate":
             book = {
                "bids": [[float(p), float(q)] for p, q in data.get("b", [])],
                "asks": [[float(p), float(q)] for p, q in data.get("a", [])],
                "timestamp": data.get("E", int(time.time() * 1000))
            }
             if symbol in self._orderbooks:
                self._orderbooks[symbol] = book
                await self.on_orderbook(symbol, book)

    async def _run_loop(self):
        backoff = 1.0
        
        # Connect to combined stream
        streams_query = "/".join([f"{s}@kline_5m" for s in self.symbols] + 
                                [f"{s}@depth20@100ms" for s in self.symbols] +
                                [f"{s}@aggTrade" for s in self.symbols])
        url = f"wss://stream.binance.com:9443/stream?streams={streams_query}"
        
        while self.running:
            try:
                log.info("Connecting to Binance WS...", url=url)
                async with websockets.connect(url) as ws:
                    self._ws = ws
                    backoff = 1.0  # Reset backoff on successful connect
                    self.last_message_time = time.time()
                    
                    async for message in ws:
                        if not self.running:
                            break
                        await self._handle_message(message)
                        
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.warning("WebSocket disconnected", error=str(e), next_retry=backoff)
                if not self.running:
                    break
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2.0, 60.0)
                
    async def _watchdog(self):
        while self.running:
            await asyncio.sleep(5)
            # If no message for 30s, force reconnect
            if time.time() - self.last_message_time > 30:
                log.warning("Market feed stalled (no messages > 30s), forcing reconnect")
                if self._ws:
                    await self._ws.close()
                self.last_message_time = time.time() # Give some grace time to reconnect
