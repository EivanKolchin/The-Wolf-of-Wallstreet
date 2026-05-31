"""Phase 15 — Higher-Timeframe (1h / 4h) features for the LIVE agent.

The offline pretrainer (`scripts/pretrain.py:build_htf_features`) already computes
HTF features for training. Until now the LIVE feature builder put zeros into
`feature_spec.HTF[62:70]` — a train/serve skew. This module is the live
counterpart: same 4+4 features per timeframe, computed with `talib` (already used
by `backend/signals/technical.py` and `regime.py`) so we don't add a new dep.

Layout produced (8 floats, matching `feature_spec.HTF_START..INPUT`):
  [62] 1h rsi_norm        (RSI(14) -> (rsi-50)/50, clipped [-1,1])
  [63] 1h ema21_dist      ((close - ema21)/close, clipped [-0.1, 0.1])
  [64] 1h macd_hist_norm  (MACD hist / close * 100, clipped [-1, 1])
  [65] 1h atr_norm        (ATR(14)/close, clipped [0, 0.1] * 10)
  [66] 4h rsi_norm
  [67] 4h ema21_dist
  [68] 4h trend_dir       (sign(ema21 - ema50): +1 bull, -1 bear)
  [69] 4h atr_norm

`HTFFeatureProvider` caches per-symbol/per-TF klines in Redis (`features:htf:{sym}:{tf}`)
with TTL ≈ 30 min and refetches in the background. Rate-limited via the shared
Binance limiter so 1h+4h × N symbols stays well under the limit.
"""
from __future__ import annotations

import asyncio
import json
import time
from typing import Iterable, Optional

import aiohttp
import numpy as np
import pandas as pd
import structlog

try:
    import talib                                  # type: ignore
    HAS_TALIB = True
except Exception:
    HAS_TALIB = False

from backend.core import ledger
from backend.core.rate_limiter import binance_rest_limiter

logger = structlog.get_logger(__name__)

BINANCE_KLINES = "https://api.binance.us/api/v3/klines"

# Per-timeframe slot offsets inside the 8-vec (relative to HTF_START)
_TF_SLOTS_1H = (0, 1, 2, 3)  # rsi_norm, ema21_dist, macd_hist_norm, atr_norm
_TF_SLOTS_4H = (4, 5, 6, 7)  # rsi_norm, ema21_dist, trend_dir,      atr_norm

EPS = 1e-9


def _safe_last(arr) -> float:
    """Return the last finite value of a talib output, or 0.0."""
    if arr is None:
        return 0.0
    try:
        a = np.asarray(arr, dtype=float)
        finite = a[np.isfinite(a)]
        return float(finite[-1]) if finite.size else 0.0
    except Exception:
        return 0.0


def compute_htf_features_for_tf(df: pd.DataFrame, tf: str) -> np.ndarray:
    """Compute the 4-feature HTF vector for one timeframe ('1h' or '4h')."""
    out = np.zeros(4, dtype=np.float32)
    if not HAS_TALIB or df is None or len(df) < 50:
        return out
    try:
        close = df["close"].astype(float).values
        high = df["high"].astype(float).values
        low = df["low"].astype(float).values
        last_close = float(close[-1]) if close.size else 0.0
        denom_close = (last_close + EPS) if last_close else 1.0

        rsi = _safe_last(talib.RSI(close, timeperiod=14))
        rsi_norm = max(-1.0, min(1.0, (rsi - 50.0) / 50.0))

        ema21 = _safe_last(talib.EMA(close, timeperiod=21))
        ema21_dist = max(-0.1, min(0.1, (last_close - ema21) / denom_close)) if ema21 else 0.0

        atr = _safe_last(talib.ATR(high, low, close, timeperiod=14))
        atr_norm = max(0.0, min(0.1, atr / denom_close)) * 10.0   # -> ~[0, 1]

        if tf == "1h":
            m, _s, h = talib.MACD(close)
            macd_hist = _safe_last(h)
            macd_norm = max(-1.0, min(1.0, (macd_hist / denom_close) * 100.0))
            out[:] = (rsi_norm, ema21_dist, macd_norm, atr_norm)
        else:  # "4h" -> trend instead of MACD hist
            ema50 = _safe_last(talib.EMA(close, timeperiod=50))
            trend = 1.0 if (ema21 and ema50 and ema21 > ema50) else (-1.0 if (ema21 and ema50) else 0.0)
            out[:] = (rsi_norm, ema21_dist, trend, atr_norm)
    except Exception as e:
        logger.warning("htf_compute_failed", tf=tf, error=str(e))
    return out


async def fetch_binance_klines(symbol: str, interval: str, limit: int = 200) -> pd.DataFrame:
    """Single REST fetch of Binance klines (rate-limited + API-logged)."""
    await binance_rest_limiter().acquire()
    t0 = time.monotonic()
    params = {"symbol": symbol, "interval": interval, "limit": str(limit)}
    try:
        timeout = aiohttp.ClientTimeout(total=8)
        async with aiohttp.ClientSession(timeout=timeout) as s:
            async with s.get(BINANCE_KLINES, params=params) as r:
                data = await r.json(content_type=None)
                ledger.log_api_call("binance", "GET", BINANCE_KLINES, status=r.status, ok=True,
                                    latency_ms=(time.monotonic() - t0) * 1000)
    except Exception as e:
        ledger.log_api_call("binance", "GET", BINANCE_KLINES, ok=False,
                            latency_ms=(time.monotonic() - t0) * 1000, note=str(e)[:120])
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    if not isinstance(data, list) or not data:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    df = pd.DataFrame(data, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_vol", "trades", "taker_buy_base", "taker_buy_quote", "ignore",
    ])
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = df[c].astype(float)
    return df[["open", "high", "low", "close", "volume"]]


class HTFFeatureProvider:
    """Manages 1h/4h klines cache + delivers the 8-feature HTF vector per symbol.

    `start()` spawns one background task per (symbol, timeframe) that refreshes the
    cache periodically. `get_features(symbol)` is synchronous and reads the latest
    cached 8-vec from in-memory map (also persisted to Redis as JSON for inspection).
    """

    def __init__(self, redis_client, symbols: Iterable[str],
                 refresh_seconds: float = 1800.0):
        self.redis = redis_client
        self.symbols = list(symbols)
        self.refresh_seconds = float(refresh_seconds)
        self._latest: dict[str, np.ndarray] = {s: np.zeros(8, dtype=np.float32) for s in self.symbols}
        self._tasks: list[asyncio.Task] = []
        self._running = False

    def get_features(self, symbol: str) -> np.ndarray:
        """Return the latest cached 8-feature HTF vector for `symbol`, or zeros."""
        v = self._latest.get(symbol)
        return v if v is not None else np.zeros(8, dtype=np.float32)

    async def _refresh_one(self, symbol: str) -> None:
        try:
            df_1h, df_4h = await asyncio.gather(
                fetch_binance_klines(symbol, "1h"),
                fetch_binance_klines(symbol, "4h"),
            )
            v = np.zeros(8, dtype=np.float32)
            v[list(_TF_SLOTS_1H)] = compute_htf_features_for_tf(df_1h, "1h")
            v[list(_TF_SLOTS_4H)] = compute_htf_features_for_tf(df_4h, "4h")
            self._latest[symbol] = v
            if self.redis is not None:
                try:
                    await self.redis.setex(f"features:htf:{symbol}", 1800, json.dumps(v.tolist()))
                except Exception:
                    pass
            logger.debug("htf_refresh_ok", symbol=symbol, vec=[round(float(x), 4) for x in v])
        except Exception as e:
            logger.warning("htf_refresh_failed", symbol=symbol, error=str(e))

    async def _loop_for(self, symbol: str) -> None:
        while self._running:
            await self._refresh_one(symbol)
            await asyncio.sleep(self.refresh_seconds)

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        for sym in self.symbols:
            self._tasks.append(asyncio.create_task(self._loop_for(sym)))
        logger.info("htf_provider_started", symbols=self.symbols, refresh_seconds=self.refresh_seconds)

    def stop(self) -> None:
        self._running = False
        for t in self._tasks:
            t.cancel()
        self._tasks.clear()
