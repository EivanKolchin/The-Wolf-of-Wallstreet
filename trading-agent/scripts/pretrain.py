"""

Usage:
  python scripts/pretrain.py
  python scripts/pretrain.py --start-year 2020 --start-month 1 --epochs 40
  python scripts/pretrain.py --skip-download   # reuse cached CSVs
  python scripts/pretrain.py --symbols BTCUSDT ETHUSDT  # subset

"""

import argparse
import io
import math
import os
import shutil
import sys
import time
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import pandas as pd
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from structlog import get_logger
from torch.utils.data import DataLoader, TensorDataset, Dataset, ConcatDataset

# sklearn is only needed for end-of-training metric reports; importing lazily
# means tests + headless usage don't pull a heavy optional dependency.
def _classification_report(*args, **kwargs):
    from sklearn.metrics import classification_report   # type: ignore
    return classification_report(*args, **kwargs)
classification_report = _classification_report  # type: ignore

# pandas_ta is effectively abandoned: the classic 0.3.14b0 this code was written
# for was pulled from PyPI, and the only remaining 0.4.x demands
# numpy>=2.2.6 / pandas>=2.3.2 — which fights Colab's pinned pandas==2.2.2 and
# kept making training un-installable. We only used 8 standard indicators, so we
# vendor a tiny pure-pandas drop-in named `ta` with the SAME function names AND
# output column order (the code reads them positionally), so behaviour is
# unchanged but there is ZERO external indicator dependency.
class _TA:
    @staticmethod
    def _rma(s, length):
        # Wilder's smoothing (RMA), as used by pandas_ta's rsi/atr/adx.
        return s.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()

    @staticmethod
    def ema(close, length):
        return close.ewm(span=length, adjust=False).mean()

    @staticmethod
    def rsi(close, length=14):
        d = close.diff()
        ag = _TA._rma(d.clip(lower=0.0), length)
        al = _TA._rma((-d).clip(lower=0.0), length)
        rs = ag / al.replace(0.0, np.nan)
        return (100.0 - 100.0 / (1.0 + rs)).fillna(100.0)

    @staticmethod
    def macd(close, fast=12, slow=26, signal=9):
        macd = close.ewm(span=fast, adjust=False).mean() - close.ewm(span=slow, adjust=False).mean()
        sig = macd.ewm(span=signal, adjust=False).mean()
        # pandas_ta order: MACD, MACDh (histogram), MACDs (signal)
        return pd.DataFrame({"MACD": macd, "MACDh": macd - sig, "MACDs": sig})

    @staticmethod
    def stochrsi(close, length=14, rsi_length=14, k=3, d=3):
        r = _TA.rsi(close, rsi_length)
        lo = r.rolling(length).min()
        hi = r.rolling(length).max()
        st = 100.0 * (r - lo) / (hi - lo).replace(0.0, np.nan)
        kl = st.rolling(k).mean()
        # pandas_ta order: STOCHRSIk, STOCHRSId
        return pd.DataFrame({"STOCHRSIk": kl, "STOCHRSId": kl.rolling(d).mean()})

    @staticmethod
    def _tr(high, low, close):
        pc = close.shift(1)
        return pd.concat([(high - low), (high - pc).abs(), (low - pc).abs()], axis=1).max(axis=1)

    @staticmethod
    def atr(high, low, close, length=14):
        return _TA._rma(_TA._tr(high, low, close), length)

    @staticmethod
    def adx(high, low, close, length=14):
        up = high.diff()
        dn = -low.diff()
        plus_dm = ((up > dn) & (up > 0)).astype(float) * up
        minus_dm = ((dn > up) & (dn > 0)).astype(float) * dn
        atr = _TA._rma(_TA._tr(high, low, close), length).replace(0.0, np.nan)
        pdi = 100.0 * _TA._rma(plus_dm, length) / atr
        mdi = 100.0 * _TA._rma(minus_dm, length) / atr
        dx = 100.0 * (pdi - mdi).abs() / (pdi + mdi).replace(0.0, np.nan)
        # pandas_ta order: ADX, DMP (+DI), DMN (-DI)
        return pd.DataFrame({"ADX": _TA._rma(dx, length), "DMP": pdi, "DMN": mdi})

    @staticmethod
    def bbands(close, length=20, std=2.0):
        mid = close.rolling(length).mean()
        sd = close.rolling(length).std(ddof=0)
        lower, upper = mid - std * sd, mid + std * sd
        # pandas_ta order: BBL (lower), BBM (mid), BBU (upper), BBB, BBP
        return pd.DataFrame({
            "BBL": lower, "BBM": mid, "BBU": upper,
            "BBB": 100.0 * (upper - lower) / mid.replace(0.0, np.nan),
            "BBP": (close - lower) / (upper - lower).replace(0.0, np.nan),
        })

    @staticmethod
    def obv(close, volume):
        return (np.sign(close.diff().fillna(0.0)) * volume).cumsum()


ta = _TA
HAS_PANDAS_TA = True

# ── project root on path ───────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

# Load .env so ALPACA_API_KEY / ALPACA_SECRET_KEY are available to the stock-
# history loader (Phase 7b). If python-dotenv isn't installed, fall back to a
# tiny inline parser so the script keeps working without a new dependency.
def _load_dotenv() -> None:
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    try:
        import dotenv  # type: ignore
        dotenv.load_dotenv(env_path)
        return
    except ImportError:
        pass
    import os
    for line in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        v = v.strip().strip("'").strip('"')
        os.environ.setdefault(k.strip(), v)
_load_dotenv()

from backend.signals import feature_spec as fs

log = get_logger("pretrain_v2")

# =============================================================================
# CONFIG
# =============================================================================

FEATURE_VERSION = fs.VERSION      # canonical FeatureSpec version (single source of truth)
SEQ_LEN         = 60              # LSTM lookback (candles @ 5m = 5 hrs)
BASE_FEATURES   = fs.BASE         # matches live system (62)
HTF_FEATURES    = fs.HTF          # 4 from 1h + 4 from 4h (8)
NEWS_EMBED_FEATURES = fs.NEWS_EMBED_DIM  # Phase 3 semantic news embedding (16)
INPUT_SIZE      = fs.INPUT        # 86 (= 62 BASE + 8 HTF + 16 NEWS_EMBED)

SYMBOL_EMBED_DIM = 16
HIDDEN_SIZE      = 256
NUM_LSTM_LAYERS  = 3
DROPOUT          = 0.3
HORIZONS         = [3, 12, 48]    # candles ahead for each label head
THRESHOLDS       = [0.003, 0.005, 0.010]  # long/short threshold per horizon
NUM_CLASSES      = 3              # 0=long, 1=short, 2=hold

# Phase 7b: stock underlyings appended to the crypto list. Order MUST match the
# live `backend.agents.improved_model.SYMBOLS` exactly so embedding-table IDs
# stay aligned between offline pretraining and live inference.
SYMBOLS = [
    # ---- crypto (ids 0..7) ----
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "AAVEUSDT",
    "XLMUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT",
    # ---- US stocks (ids 8..12) — fetched from Alpaca ----
    "SNDK", "AMD", "MU", "AXTI", "BE",
    # ---- Cycle 6 additions (ids 13..17) — APPENDED so existing ids stay stable ----
    "RENDERUSDT", "NEARUSDT",          # crypto (Render = ex-RNDR; Near) — Binance
    "NVDA", "TSM", "SMCI",             # AI-chip stocks — help the model learn the AMD/MU sector
]
SYMBOL_TO_ID = {s: i for i, s in enumerate(SYMBOLS)}

DATA_DIR    = ROOT / "training_data" / "raw"
MODELS_DIR  = ROOT / "models"
ZSCORE_WIN  = 1000   # rolling normalisation window
BATCH_SIZE  = 128
LR          = 3e-4
# A4 anti-overfit: keep these in sync with the live config so offline training
# regularises the model the same way the online learner does.
try:
    from backend.core.config import settings as _settings
    WEIGHT_DECAY = float(getattr(_settings, "NN_WEIGHT_DECAY", 1e-4))
    LABEL_SMOOTHING = float(getattr(_settings, "NN_LABEL_SMOOTHING", 0.05))
    # Recency weighting: scale each sample's loss by a calendar half-life on its
    # age, so the model leans toward CURRENT market regimes (post-COVID normalisation,
    # the AI-driven equity boom) while still learning from older data. HALFLIFE=2y
    # ⇒ a 2-year-old bar counts half as much as today's; FLOOR keeps old data from
    # ever being fully ignored (so multi-regime history is still learned).
    RECENCY_HALFLIFE_YEARS = float(getattr(_settings, "NN_RECENCY_HALFLIFE_YEARS", 2.0))
    RECENCY_FLOOR = float(getattr(_settings, "NN_RECENCY_FLOOR", 0.25))
    # Focal loss exponent — down-weights easy/abundant 'hold' samples so learning
    # concentrates on the rare, hard long/short moves. 0 ⇒ plain weighted CE.
    FOCAL_GAMMA = float(getattr(_settings, "NN_FOCAL_GAMMA", 1.5))
    # Triple-barrier label width multiple (× rolling vol × √horizon).
    BARRIER_K = float(getattr(_settings, "NN_BARRIER_K", 1.0))
except Exception:
    WEIGHT_DECAY = 1e-4
    LABEL_SMOOTHING = 0.05
    RECENCY_HALFLIFE_YEARS = 2.0
    RECENCY_FLOOR = 0.25
    FOCAL_GAMMA = 1.5
    BARRIER_K = 1.0

VOL_WINDOW = 20   # rolling window (bars) for the 1-bar return vol that scales barriers

# Binance bulk portal base
BV_BASE = "https://data.binance.vision/data/spot/monthly/klines"

# =============================================================================
# IMPROVED MODEL
# =============================================================================

class AttentionLayer(nn.Module):
    """Additive (Bahdanau-style) attention over LSTM sequence."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.score = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, lstm_out: torch.Tensor):
        # lstm_out: (B, T, H)
        weights = self.score(lstm_out)          # (B, T, 1)
        weights = F.softmax(weights, dim=1)
        context = (weights * lstm_out).sum(1)   # (B, H)
        return context, weights.squeeze(-1)     # (B, H), (B, T)


class ImprovedTradingLSTM(nn.Module):
    """
    Architecture:
      ┌─ Symbol embedding ─┐
      │  (B, embed_dim)    │
      └──────┬─────────────┘
             cat with features → (B, T, INPUT_SIZE + embed_dim)
             ↓
        3-layer LSTM  (hidden=256, causal)
             ↓
        LayerNorm
             ↓
        Dot-product Attention → context (B, 256)
             ↓
        Dropout
             ↓
        Shared FC: 256→128→ReLU→Dropout→64→ReLU
             ↓
      ┌──────┴──────────┐
      │  Per-horizon    │   Size head
      │  direction heads│   64→32→ReLU→1→Sigmoid
      │  64→3 (logits)  │
      └─────────────────┘
    """

    def __init__(
        self,
        input_size:      int = INPUT_SIZE,
        hidden_size:     int = HIDDEN_SIZE,
        num_layers:      int = NUM_LSTM_LAYERS,
        dropout:         float = DROPOUT,
        num_symbols:     int = len(SYMBOLS),
        symbol_embed_dim: int = SYMBOL_EMBED_DIM,
        num_horizons:    int = len(HORIZONS),
        num_classes:     int = NUM_CLASSES,
    ):
        super().__init__()
        self.num_horizons = num_horizons
        self.symbol_embedding = nn.Embedding(num_symbols, symbol_embed_dim)

        # A4: keep this in lock-step with backend/agents/improved_model.py so
        # offline-trained checkpoints load into the live model. dropout + RNN
        # core type are config-driven (gru = fewer params, faster, less overfit).
        try:
            from backend.core.config import settings as _s
            dropout = float(getattr(_s, "NN_DROPOUT", dropout))
            rnn_type = str(getattr(_s, "NN_RNN_TYPE", "lstm")).lower()
        except Exception:
            rnn_type = "lstm"
        self.rnn_type = rnn_type

        lstm_in = input_size + symbol_embed_dim
        _rnn = nn.GRU if rnn_type == "gru" else nn.LSTM
        self.lstm = _rnn(
            input_size=lstm_in,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=False,   # causal — no future peek
        )
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.attention  = AttentionLayer(hidden_size)
        self.dropout    = nn.Dropout(dropout)

        self.shared = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

        # One direction head per horizon — multi-task learning
        self.direction_heads = nn.ModuleList(
            [nn.Linear(64, num_classes) for _ in range(num_horizons)]
        )

        # Position sizing (shared across horizons)
        self.size_head = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1),  nn.Sigmoid(),
        )

        # Learnable temperature for calibration
        self.temperature = nn.Parameter(torch.ones(1))

        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(p)
            elif "weight_hh" in name:
                nn.init.orthogonal_(p)
            elif "bias" in name:
                nn.init.zeros_(p)

    def forward(
        self,
        x: torch.Tensor,           # (B, T, INPUT_SIZE)
        symbol_ids: torch.Tensor,  # (B,)
    ):
        B, T, _ = x.shape
        emb = self.symbol_embedding(symbol_ids)          # (B, embed_dim)
        emb = emb.unsqueeze(1).expand(-1, T, -1)         # (B, T, embed_dim)
        x   = torch.cat([x, emb], dim=-1)                # (B, T, lstm_in)

        lstm_out, _ = self.lstm(x)                        # (B, T, H)
        lstm_out    = self.layer_norm(lstm_out)
        context, attn_w = self.attention(lstm_out)        # (B, H), (B, T)
        context = self.dropout(context)

        shared = self.shared(context)                     # (B, 64)

        logits_list = [h(shared) / self.temperature for h in self.direction_heads]
        probs_list  = [F.softmax(lg, dim=-1) for lg in logits_list]
        size        = self.size_head(shared)              # (B, 1)

        return logits_list, probs_list, size, attn_w

    def predict(self, x: torch.Tensor, symbol_ids: torch.Tensor, horizon_idx: int = 0):
        """Convenience for live inference — returns probs + size for one horizon."""
        self.eval()
        with torch.no_grad():
            _, probs_list, size, _ = self(x, symbol_ids)
        return probs_list[horizon_idx], size


# =============================================================================
# DATA DOWNLOAD — data.binance.vision + API gap-fill
# =============================================================================

def _bv_url(symbol: str, interval: str, year: int, month: int) -> str:
    fname = f"{symbol}-{interval}-{year:04d}-{month:02d}.zip"
    return f"{BV_BASE}/{symbol}/{interval}/{fname}"


def _download_monthly(symbol: str, interval: str, year: int, month: int) -> Optional[pd.DataFrame]:
    url = _bv_url(symbol, interval, year, month)
    try:
        r = requests.get(url, timeout=30)
        if r.status_code == 404:
            return None
        r.raise_for_status()
    except Exception as e:
        log.warning("Download failed", url=url, error=str(e))
        return None

    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        csv_name = z.namelist()[0]
        with z.open(csv_name) as f:
            df = pd.read_csv(f, header=None, names=[
                "timestamp", "open", "high", "low", "close", "volume",
                "close_time", "quote_volume", "trades",
                "taker_buy_base", "taker_buy_quote", "ignore"
            ])

    # Binance switched bulk-CSV kline timestamps from milliseconds to MICROSECONDS
    # in 2025, and some newer monthly files prepend a header row. Coerce to numeric
    # (a stray header becomes NaN and is dropped), then auto-detect the unit by
    # magnitude so pre-2025 (ms) and 2025+ (µs) months both parse correctly.
    ts = pd.to_numeric(df["timestamp"], errors="coerce")
    df = df.loc[ts.notna()].copy()
    ts = ts.loc[df.index]
    unit = "us" if (len(ts) > 0 and float(ts.iloc[0]) >= 1e14) else "ms"
    df["timestamp"] = pd.to_datetime(ts.astype("int64"), unit=unit)
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)

    return df[["timestamp", "open", "high", "low", "close", "volume"]].sort_values("timestamp")


def load_or_download(
    symbol: str,
    interval: str,
    start_year: int,
    start_month: int,
    end_year: int,
    end_month: int,
    skip_download: bool = False,
) -> pd.DataFrame:
    """
    Loads cached parquet if present, otherwise downloads month-by-month
    from data.binance.vision.  Missing months (symbol too new, or 404) are
    skipped gracefully.
    """
    cache_path = DATA_DIR / f"{symbol}_{interval}_{start_year}{start_month:02d}_{end_year}{end_month:02d}.parquet"
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if cache_path.exists() and not skip_download:
        log.info("Loading from cache", path=str(cache_path))
        return pd.read_parquet(cache_path)

    frames = []
    year, month = start_year, start_month
    while (year, month) <= (end_year, end_month):
        log.info("Downloading", symbol=symbol, interval=interval, year=year, month=month)
        df_m = _download_monthly(symbol, interval, year, month)
        if df_m is not None:
            frames.append(df_m)
        else:
            log.warning("Missing / not available", symbol=symbol, year=year, month=month)
        time.sleep(0.3)
        month += 1
        if month > 12:
            month = 1
            year += 1

    if not frames:
        raise RuntimeError(f"No data downloaded for {symbol} {interval}")

    result = pd.concat(frames, ignore_index=True).drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    result.to_parquet(cache_path)
    log.info("Saved to cache", path=str(cache_path), rows=len(result))
    return result


def api_gap_fill(symbol: str, interval: str, after_ts: pd.Timestamp) -> pd.DataFrame:
    """
    Fetches candles from Binance REST API from after_ts up to now.
    Used to bridge the gap between the last bulk CSV month and today.
    """
    log.info("API gap-fill", symbol=symbol, interval=interval, after=str(after_ts))
    # Use Binance's PUBLIC data mirror, not api.binance.com — the latter returns
    # HTTP 451 (geo-block) from US / cloud IPs (e.g. Google Colab). The bulk CSVs
    # already cover the full history, so this top-up is best-effort and NON-FATAL:
    # any failure is logged and skipped rather than crashing the whole run.
    url = "https://data-api.binance.vision/api/v3/klines"
    all_rows = []
    start_ms = int(after_ts.timestamp() * 1000) + 1
    try:
        while True:
            params = {"symbol": symbol, "interval": interval, "startTime": start_ms, "limit": 1000}
            r = requests.get(url, params=params, timeout=15)
            r.raise_for_status()
            data = r.json()
            if not data:
                break
            all_rows.extend(data)
            if len(data) < 1000:
                break
            start_ms = data[-1][0] + 1
            time.sleep(0.3)
    except Exception as e:
        log.warning("API gap-fill skipped (using bulk CSV history only)",
                    symbol=symbol, error=str(e)[:200])

    if not all_rows:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    df = pd.DataFrame(all_rows, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades",
        "taker_buy_base", "taker_buy_quote", "ignore"
    ])
    # Same ms/µs auto-detect as the bulk loader (Binance switched units in 2025).
    ts = pd.to_numeric(df["timestamp"], errors="coerce")
    unit = "us" if (len(ts) > 0 and float(ts.iloc[0]) >= 1e14) else "ms"
    df["timestamp"] = pd.to_datetime(ts.astype("int64"), unit=unit)
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)
    return df[["timestamp", "open", "high", "low", "close", "volume"]]


# =============================================================================
# Stock history via Alpaca (Phase 7b)
# =============================================================================

_ALPACA_DATA_BASE = "https://data.alpaca.markets/v2"
_ALPACA_TF = {"5m": "5Min", "1h": "1Hour", "4h": "4Hour"}

# Symbol classification — mirror backend/core/universe.STOCK_UNDERLYINGS.
_STOCK_SYMBOLS = {"SNDK", "AMD", "MU", "AXTI", "BE", "NVDA", "TSM", "SMCI"}


def _is_stock_symbol(sym: str) -> bool:
    return (sym or "").upper() in _STOCK_SYMBOLS


def _alpaca_headers() -> dict:
    """Pull Alpaca credentials from the same env the live agent uses."""
    import os
    return {
        "APCA-API-KEY-ID": os.environ.get("ALPACA_API_KEY", ""),
        "APCA-API-SECRET-KEY": os.environ.get("ALPACA_SECRET_KEY") or os.environ.get("ALPACA_SECRET", ""),
    }


def load_alpaca_history(symbol: str, start_year: int, start_month: int) -> Dict[str, pd.DataFrame]:
    """Page through Alpaca Stock Bars (free IEX feed) from start_year-start_month
    to now, for each TF used by the live model. Returns the same shape as
    ``load_full_history``: ``{"5m": df, "1h": df, "4h": df}`` with columns
    ``timestamp, open, high, low, close, volume``.
    """
    import requests
    import datetime as _dt

    hdrs = _alpaca_headers()
    if not hdrs["APCA-API-KEY-ID"] or not hdrs["APCA-API-SECRET-KEY"]:
        raise RuntimeError("ALPACA_API_KEY / ALPACA_SECRET_KEY not set in env")

    start_iso = f"{start_year:04d}-{start_month:02d}-01T00:00:00Z"
    out: Dict[str, pd.DataFrame] = {}
    for tf_short, tf_alpaca in _ALPACA_TF.items():
        all_rows: List[dict] = []
        page_token = None
        params_base = {
            "timeframe": tf_alpaca, "start": start_iso,
            "limit": "10000", "adjustment": "raw", "feed": "iex", "sort": "asc",
        }
        url = f"{_ALPACA_DATA_BASE}/stocks/{symbol.upper()}/bars"
        for _ in range(100):  # hard cap on pagination
            params = dict(params_base)
            if page_token:
                params["page_token"] = page_token
            r = requests.get(url, headers=hdrs, params=params, timeout=20)
            if r.status_code != 200:
                log.warning("alpaca_bars_http_error", symbol=symbol, status=r.status_code,
                             body=r.text[:200])
                break
            data = r.json() or {}
            bars = data.get("bars") or []
            for b in bars:
                try:
                    ts = _dt.datetime.fromisoformat(b["t"].replace("Z", "+00:00"))
                    all_rows.append({
                        "timestamp": ts, "open": float(b["o"]), "high": float(b["h"]),
                        "low": float(b["l"]), "close": float(b["c"]), "volume": float(b["v"]),
                    })
                except Exception:
                    continue
            page_token = data.get("next_page_token")
            if not page_token:
                break

        if not all_rows:
            raise RuntimeError(f"Alpaca returned 0 bars for {symbol} @ {tf_alpaca}")
        df = pd.DataFrame(all_rows).drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(None)
        out[tf_short] = df
        log.info("alpaca_history_loaded", symbol=symbol, tf=tf_short, rows=len(df))
    return out


# =============================================================================
# Phase 3: offline news alignment — fill the NEWS_EMBED block [70:86] with a
# semantic embedding of historical news aligned to each 5m bar. Free source:
# Alpaca's News API (Benzinga, historical to 2015) using the SAME credentials
# the bar loader uses. The embedding backend is the SAME NewsEmbedder the live
# agent uses, so offline-trained weights stay valid against live vectors.
#
# Gated behind PRETRAIN_NEWS_ALIGN=1 (and credentials). When disabled or on any
# error it returns zeros — pretraining still runs, the block is just inert,
# exactly mirroring "no fresh news" at inference time.
# =============================================================================

_ALPACA_NEWS_URL = "https://data.alpaca.markets/v1beta1/news"


def _fetch_alpaca_news(symbol: str, start_iso: str, end_iso: str) -> List[dict]:
    """Page through Alpaca/Benzinga historical news for one symbol."""
    import requests
    hdrs = _alpaca_headers()
    if not hdrs["APCA-API-KEY-ID"] or not hdrs["APCA-API-SECRET-KEY"]:
        return []
    items: List[dict] = []
    page_token = None
    for _ in range(200):  # hard cap on pagination
        params = {
            "symbols": symbol.upper(), "start": start_iso, "end": end_iso,
            "limit": "50", "sort": "asc", "include_content": "false",
        }
        if page_token:
            params["page_token"] = page_token
        try:
            r = requests.get(_ALPACA_NEWS_URL, headers=hdrs, params=params, timeout=20)
        except Exception as e:
            log.warning("alpaca_news_request_failed", symbol=symbol, error=str(e))
            break
        if r.status_code != 200:
            log.warning("alpaca_news_http_error", symbol=symbol, status=r.status_code, body=r.text[:160])
            break
        data = r.json() or {}
        for n in (data.get("news") or []):
            ts = n.get("created_at") or n.get("updated_at")
            if not ts:
                continue
            items.append({
                "ts": pd.to_datetime(ts, utc=True).tz_convert(None),
                "text": f"{n.get('headline', '')}. {n.get('summary', '')}".strip(),
            })
        page_token = data.get("next_page_token")
        if not page_token:
            break
    return items


# Effective news-embedding backend used during this build ("disabled" when news
# alignment is off → the news block is all zeros). Saved into the checkpoint meta.
_NEWS_BACKEND_USED = "disabled"


def build_news_embed_matrix(df: pd.DataFrame, symbol: str) -> np.ndarray:
    """(N, NEWS_EMBED_FEATURES) matrix aligned to df's 5m bars.

    For each bar, embeds the most recent news within a lookback window
    (PRETRAIN_NEWS_LOOKBACK_MIN, default 120 min). Bars with no recent news get
    zeros — identical to the live agent when there's no fresh, relevant news.
    """
    n = len(df)
    mat = np.zeros((n, NEWS_EMBED_FEATURES), dtype=np.float32)
    if os.environ.get("PRETRAIN_NEWS_ALIGN", "0") not in ("1", "true", "True"):
        return mat
    try:
        from backend.signals.news_embedding import NewsEmbedder
    except Exception as e:
        log.warning("news_embedder_import_failed", error=str(e))
        return mat

    ts_col = pd.to_datetime(df["timestamp"])
    start_iso = ts_col.iloc[0].strftime("%Y-%m-%dT%H:%M:%SZ")
    end_iso = ts_col.iloc[-1].strftime("%Y-%m-%dT%H:%M:%SZ")
    news = _fetch_alpaca_news(symbol, start_iso, end_iso)
    if not news:
        log.info("news_align_no_items", symbol=symbol)
        return mat

    lookback = pd.Timedelta(minutes=float(os.environ.get("PRETRAIN_NEWS_LOOKBACK_MIN", "120")))
    embedder = NewsEmbedder()
    # Record the backend actually producing these features → saved in the checkpoint
    # so the live agent can verify it uses the SAME one (else the 16 news dims drift).
    global _NEWS_BACKEND_USED
    _NEWS_BACKEND_USED = embedder.effective_backend()
    news.sort(key=lambda x: x["ts"])
    news_ts = pd.to_datetime([x["ts"] for x in news])

    # cache embeddings per unique text to avoid recompute
    emb_cache: dict[str, np.ndarray] = {}
    matched = 0
    for i in range(n):
        bar_t = ts_col.iloc[i]
        # most recent news at or before this bar, within the lookback window
        j = int(np.searchsorted(news_ts.values, np.datetime64(bar_t), side="right")) - 1
        if j < 0:
            continue
        if bar_t - news_ts[j] > lookback:
            continue
        text = news[j]["text"]
        if not text:
            continue
        if text not in emb_cache:
            emb_cache[text] = embedder.embed_text(text)
        mat[i] = emb_cache[text]
        matched += 1
    log.info("news_align_done", symbol=symbol, bars=n, matched=matched, unique_news=len(emb_cache))
    return mat


# Whether earnings features were actually aligned during this build (recorded in
# the checkpoint so live can match). Crypto / align-off / no-key → stays False.
_EARNINGS_ALIGNED = False


def build_earnings_matrix(df: pd.DataFrame, symbol: str) -> np.ndarray:
    """(N, EARNINGS_DIM) earnings-calendar features aligned to df's 5m bars.

    Stocks only (crypto → zeros). Gated behind PRETRAIN_EARNINGS_ALIGN (default off)
    because it hits the Finnhub calendar API; off → zeros (identical to a live agent
    with no earnings data). Leakage-safe: see backend.signals.earnings.
    """
    n = len(df)
    mat = np.zeros((n, fs.EARNINGS_DIM), dtype=np.float32)
    if not _is_stock_symbol(symbol):
        return mat
    if os.environ.get("PRETRAIN_EARNINGS_ALIGN", "0") not in ("1", "true", "True"):
        return mat
    try:
        from backend.core.config import settings as _s
        token = (getattr(_s, "FINNHUB_API_KEY", "") or "")
    except Exception:
        token = os.environ.get("FINNHUB_API_KEY", "")
    if not token:
        log.warning("earnings_align_no_finnhub_key", symbol=symbol)
        return mat
    try:
        from backend.signals.earnings import EarningsProvider, earnings_feature_matrix
    except Exception as e:
        log.warning("earnings_module_import_failed", error=str(e))
        return mat
    ts = pd.to_datetime(df["timestamp"])
    events = EarningsProvider(token).events(
        symbol, ts.iloc[0].strftime("%Y-%m-%d"), ts.iloc[-1].strftime("%Y-%m-%d"))
    if not events:
        log.info("earnings_align_no_events", symbol=symbol)
        return mat
    global _EARNINGS_ALIGNED
    _EARNINGS_ALIGNED = True
    log.info("earnings_align_done", symbol=symbol, events=len(events))
    return earnings_feature_matrix(events, df["timestamp"].to_numpy())


def load_full_history(
    symbol: str,
    start_year: int,
    start_month: int,
    skip_download: bool = False,
) -> Dict[str, pd.DataFrame]:
    """
    Returns dict: {'5m': df, '1h': df, '4h': df}
    For crypto: Binance bulk CSVs + API gap-fill.
    For US stocks: Alpaca Stock Bars (Phase 7b).
    """
    if _is_stock_symbol(symbol):
        return load_alpaca_history(symbol, start_year, start_month)

    now   = pd.Timestamp.now()
    ey, em = now.year, now.month - 1
    if em == 0:
        em, ey = 12, ey - 1

    dfs = {}
    for interval in ["5m", "1h", "4h"]:
        df = load_or_download(symbol, interval, start_year, start_month, ey, em, skip_download)
        gap = api_gap_fill(symbol, interval, df["timestamp"].iloc[-1])
        if len(gap):
            df = pd.concat([df, gap], ignore_index=True).drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
        dfs[interval] = df

    return dfs


# =============================================================================
# VECTORISED FEATURE ENGINEERING
# =============================================================================

def _safe(series, fill=0.0):
    return series.fillna(fill).values.astype(np.float32)


def build_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    """
    Builds (N, BASE_FEATURES=62) feature matrix for one symbol using
    fully vectorised pandas_ta operations — no row-by-row loops.

    Feature layout (matches live system):
      [0-2]   candle body, upper wick, lower wick (scaled)
      [3]     volume ratio vs 20-bar MA
      [4]     spread placeholder (0.5)
      [5-8]   EMA distances (9, 21, 50, 200)
      [9]     golden cross (ema50 > ema200)
      [10]    VWAP distance
      [11]    RSI normalised
      [12-13] MACD line, MACD hist (normalised)
      [14]    StochRSI %K
      [15]    ADX normalised
      [16]    RSI divergence (binary)
      [17]    ATR normalised
      [18-19] BB width, BB %b
      [20]    Volume momentum (5-bar vs 20-bar MA — was duplicate, now distinct)
      [21]    OBV slope sign
      [22-24] Fibonacci position, distance from 50%, range strength
      [25-34] 10 candlestick pattern flags
      [35-44] Orderbook slots (0.0 — filled live)
      [45-48] Regime one-hot (ranging, bull_trend, bear_trend, volatile)
      [49-52] News slots (0.0 — filled live)
      [53-56] Macro slots (0.5 neutral — filled live)
      [57-60] Time cyclical (sin/cos hour, sin/cos weekday)
      [61]    Regime confidence
    """
    assert HAS_PANDAS_TA, "pandas_ta required for vectorised features"

    n   = len(df)
    out = np.zeros((n, BASE_FEATURES), dtype=np.float32)

    op  = df["open"].values.astype(np.float32)
    hi  = df["high"].values.astype(np.float32)
    lo  = df["low"].values.astype(np.float32)
    cl  = df["close"].values.astype(np.float32)
    vol = df["volume"].values.astype(np.float32)

    eps = 1e-8

    # [0-2] Candle body / wicks
    out[:, 0] = np.clip((cl - op) / (op + eps), -0.05, 0.05) * 20.0
    out[:, 1] = np.clip((hi - op) / (op + eps),  0.0,  0.05) * 20.0
    out[:, 2] = np.clip((lo - op) / (op + eps), -0.05, 0.0 ) * 20.0

    # [3] Volume ratio (current / 20-bar MA)
    vol_ma20  = pd.Series(vol).rolling(20, min_periods=1).mean().values
    out[:, 3] = np.clip(vol / (vol_ma20 + eps), 0, 5) / 5.0

    # [4] Spread placeholder
    out[:, 4] = 0.5

    # [5-8] EMA distances
    c_series = df["close"]
    ema9   = _safe(ta.ema(c_series, length=9))
    ema21  = _safe(ta.ema(c_series, length=21))
    ema50  = _safe(ta.ema(c_series, length=50))
    ema200 = _safe(ta.ema(c_series, length=200))

    out[:, 5] = np.clip((cl - ema9)   / (cl + eps), -0.1, 0.1)
    out[:, 6] = np.clip((cl - ema21)  / (cl + eps), -0.1, 0.1)
    out[:, 7] = np.clip((cl - ema50)  / (cl + eps), -0.1, 0.1)
    out[:, 8] = np.clip((cl - ema200) / (cl + eps), -0.1, 0.1)

    # [9] Golden cross
    out[:, 9] = (ema50 > ema200).astype(np.float32)

    # [10] VWAP distance (session-level cumulative approximation)
    typical  = (hi + lo + cl) / 3.0
    cum_vol  = np.cumsum(vol)
    cum_tpv  = np.cumsum(typical * vol)
    vwap     = cum_tpv / (cum_vol + eps)
    out[:, 10] = np.clip((cl - vwap) / (cl + eps), -0.1, 0.1)

    # [11] RSI
    rsi = _safe(ta.rsi(c_series, length=14), fill=50.0)
    out[:, 11] = (rsi - 50.0) / 50.0

    # [12-13] MACD
    macd_df = ta.macd(c_series)
    if macd_df is not None and not macd_df.empty:
        macd_line = _safe(macd_df.iloc[:, 0])
        macd_hist = _safe(macd_df.iloc[:, 1] if macd_df.shape[1] > 1 else macd_df.iloc[:, 0])
    else:
        macd_line = macd_hist = np.zeros(n, dtype=np.float32)
    out[:, 12] = np.clip(macd_line / (cl + eps) * 100, -1, 1)
    out[:, 13] = np.clip(macd_hist / (cl + eps) * 100, -1, 1)

    # [14] StochRSI
    stoch = ta.stochrsi(c_series)
    if stoch is not None and not stoch.empty:
        out[:, 14] = _safe(stoch.iloc[:, 0], fill=50.0) / 100.0
    else:
        out[:, 14] = 0.5

    # [15] ADX
    adx_df = ta.adx(df["high"], df["low"], df["close"])
    if adx_df is not None and not adx_df.empty:
        out[:, 15] = np.clip(_safe(adx_df.iloc[:, 0]) / 100.0, 0, 1)

    # [16] RSI divergence (sign disagreement: RSI slope vs price slope over 5 bars)
    rsi_slope   = np.gradient(rsi)
    price_slope = np.gradient(cl)
    out[:, 16] = (np.sign(rsi_slope) != np.sign(price_slope)).astype(np.float32)

    # [17] ATR normalised
    atr = _safe(ta.atr(df["high"], df["low"], df["close"], length=14))
    out[:, 17] = np.clip(atr / (cl + eps), 0, 0.1) * 10.0

    # [18-19] Bollinger Bands
    bb = ta.bbands(c_series, length=20)
    if bb is not None and not bb.empty:
        bb_upper = _safe(bb.iloc[:, 0])
        bb_mid   = _safe(bb.iloc[:, 1])
        bb_lower = _safe(bb.iloc[:, 2])
        bb_rng   = bb_upper - bb_lower + eps
        out[:, 18] = np.clip((bb_upper - bb_lower) / (bb_mid + eps), 0, 0.2) * 5.0
        out[:, 19] = np.clip((cl - bb_lower) / bb_rng, -0.5, 1.5)

    # [20] Volume momentum (5-bar MA vs 20-bar MA — no longer a duplicate)
    vol_ma5   = pd.Series(vol).rolling(5,  min_periods=1).mean().values
    out[:, 20] = np.clip(vol_ma5 / (vol_ma20 + eps), 0, 5) / 5.0

    # [21] OBV slope sign
    obv = _safe(ta.obv(c_series, df["volume"]))
    out[:, 21] = np.sign(np.gradient(obv)).astype(np.float32)

    # [22-24] Fibonacci
    win = 50
    roll_hi = pd.Series(hi).rolling(win, min_periods=win).max().values
    roll_lo = pd.Series(lo).rolling(win, min_periods=win).min().values
    fib_rng  = roll_hi - roll_lo + eps
    fib_50   = roll_lo + fib_rng * 0.5
    out[:, 22] = np.clip((cl - roll_lo) / fib_rng, 0, 1)
    out[:, 23] = np.clip(np.abs(cl - fib_50) / fib_rng, 0, 0.5)
    out[:, 24] = np.clip(fib_rng / (cl + eps), 0, 0.2) * 5.0

    # [25-34] Candlestick patterns (10 binary flags)
    body       = np.abs(cl - op)
    rng        = (hi - lo) + eps
    lower_wick = np.where(cl > op, op - lo, cl - lo)
    upper_wick = np.where(cl > op, hi - cl, hi - op)
    cl_s  = pd.Series(cl)
    op_s  = pd.Series(op)
    prev_cl = cl_s.shift(1).values
    prev_op = op_s.shift(1).values

    out[:, 25] = (body / rng < 0.1).astype(np.float32)                                     # doji
    out[:, 26] = ((cl > op) & (body / rng > 0.6)).astype(np.float32)                       # bull marubozu
    out[:, 27] = ((cl < op) & (body / rng > 0.6)).astype(np.float32)                       # bear marubozu
    out[:, 28] = ((lower_wick > 2*body) & (upper_wick < body)).astype(np.float32)          # hammer
    out[:, 29] = ((upper_wick > 2*body) & (lower_wick < body)).astype(np.float32)          # shooting star
    out[:, 30] = ((cl_s > cl_s.shift(1)) & (cl_s.shift(1) > cl_s.shift(2))).astype(np.float32).values  # 3 up
    out[:, 31] = ((cl_s < cl_s.shift(1)) & (cl_s.shift(1) < cl_s.shift(2))).astype(np.float32).values  # 3 down
    out[:, 32] = ((cl > prev_op) & (op < prev_cl) & (prev_cl < prev_op)).astype(np.float32)  # bull engulf
    out[:, 33] = ((cl < prev_op) & (op > prev_cl) & (prev_cl > prev_op)).astype(np.float32)  # bear engulf
    out[:, 34] = 0.0   # spare

    # [35-44] Orderbook — zeros in historical (filled live)
    out[:, fs.ORDERBOOK] = 0.0   # 8 slots (35:43) — canonical FeatureSpec layout

    # [45-48] Regime — filled by detect_regime()
    # [49-52] News — zeros (filled live)
    out[:, fs.NEWS] = 0.0

    # [53-56] Macro — neutral
    out[:, 53] = 0.5
    out[:, 54] = 0.5
    out[:, 55] = 0.0
    out[:, 56] = 0.0

    # [57-60] Time cyclical
    ts = df["timestamp"]
    out[:, 57] = np.sin(2 * np.pi * ts.dt.hour.values / 24.0).astype(np.float32)
    out[:, 58] = np.cos(2 * np.pi * ts.dt.hour.values / 24.0).astype(np.float32)
    out[:, 59] = np.sin(2 * np.pi * ts.dt.dayofweek.values / 7.0).astype(np.float32)
    out[:, 60] = np.cos(2 * np.pi * ts.dt.dayofweek.values / 7.0).astype(np.float32)

    # [61] Regime confidence — filled by detect_regime()
    out[:, 61] = 0.5

    return out


def detect_regime(df: pd.DataFrame, features: np.ndarray) -> np.ndarray:
    """
    Classifies each bar into one of 4 regimes using ADX + EMA slope:
      ranging (0), bull_trend (1), bear_trend (2), volatile (3)

    Writes regime one-hot into features[:, 45:49] and
    confidence into features[:, 61].  Modifies in-place, returns array.
    """
    cl     = df["close"].values.astype(np.float64)
    ema21  = pd.Series(cl).ewm(span=21, adjust=False).mean().values
    ema50  = pd.Series(cl).ewm(span=50, adjust=False).mean().values
    atr14  = _safe(ta.atr(df["high"], df["low"], df["close"], length=14)).astype(np.float64)

    adx_df = ta.adx(df["high"], df["low"], df["close"])
    adx    = _safe(adx_df.iloc[:, 0]).astype(np.float64) if adx_df is not None else np.zeros(len(df))

    ema_slope    = ema21 - np.roll(ema21, 5)
    vol_norm     = atr14 / (cl + 1e-8)
    vol_ma       = pd.Series(vol_norm).rolling(50, min_periods=1).mean().values
    high_vol_flag = vol_norm > (1.5 * vol_ma)

    trending      = adx > 25
    bull          = trending & (ema_slope > 0) & (ema21 > ema50)
    bear          = trending & (ema_slope < 0) & (ema21 < ema50)
    volatile      = high_vol_flag & ~trending
    ranging       = ~trending & ~high_vol_flag

    # Clear and write the canonical 6-class regime one-hot (FeatureSpec 43:49).
    # bull->uptrend, bear->downtrend, ranging->ranging, volatile->high_volatility.
    # news_driven / low_liquidity are not detectable from OHLCV alone (filled live).
    features[:, fs.REGIME] = 0.0
    features[ranging,  fs.regime_index("ranging")] = 1.0
    features[bull,     fs.regime_index("uptrend")] = 1.0
    features[bear,     fs.regime_index("downtrend")] = 1.0
    features[volatile, fs.regime_index("high_volatility")] = 1.0

    # Confidence = normalised ADX (0–1)
    features[:, fs.REGIME_CONFIDENCE] = np.clip(adx / 50.0, 0, 1).astype(np.float32)

    return features


def build_htf_features(df_5m: pd.DataFrame, df_1h: pd.DataFrame, df_4h: pd.DataFrame) -> np.ndarray:
    """
    Builds (N_5m, HTF_FEATURES=8) matrix.
    For each 5m bar, looks up the most recent completed 1h / 4h bar.
    Strictly backward-looking — no lookahead.

    1h features (4): rsi_norm, ema21_dist, macd_hist_norm, atr_norm
    4h features (4): rsi_norm, ema21_dist, trend_dir, atr_norm
    """
    n = len(df_5m)
    htf = np.zeros((n, HTF_FEATURES), dtype=np.float32)
    eps = 1e-8

    def _make_htf_signals(df_htf: pd.DataFrame) -> pd.DataFrame:
        cl = df_htf["close"]
        rsi_n    = (_safe(ta.rsi(cl, length=14), 50.0) - 50.0) / 50.0
        ema21    = _safe(ta.ema(cl, length=21))
        ema50    = _safe(ta.ema(cl, length=50))
        cl_v     = cl.values.astype(np.float32)
        ema21_d  = np.clip((cl_v - ema21) / (cl_v + eps), -0.1, 0.1)
        macd_df  = ta.macd(cl)
        if macd_df is not None and not macd_df.empty:
            mh = _safe(macd_df.iloc[:, 1] if macd_df.shape[1] > 1 else macd_df.iloc[:, 0])
        else:
            mh = np.zeros(len(df_htf), np.float32)
        macd_n   = np.clip(mh / (cl_v + eps) * 100, -1, 1)
        atr_n    = np.clip(_safe(ta.atr(df_htf["high"], df_htf["low"], cl, length=14)) / (cl_v + eps), 0, 0.1) * 10.0
        trend    = (ema21 > ema50).astype(np.float32) * 2 - 1  # +1 bull, -1 bear

        return pd.DataFrame({
            "timestamp": df_htf["timestamp"].values,
            "rsi_n": rsi_n, "ema21_d": ema21_d,
            "macd_n": macd_n, "atr_n": atr_n, "trend": trend
        })

    sig1h = _make_htf_signals(df_1h).set_index("timestamp")
    sig4h = _make_htf_signals(df_4h).set_index("timestamp")

    ts5m = df_5m["timestamp"].values

    # For each 5m bar, find the last completed HTF bar (strictly before)
    for i, ts in enumerate(ts5m):
        # 1h
        idx1h = sig1h.index.searchsorted(ts, side="left") - 1
        if idx1h >= 0:
            row = sig1h.iloc[idx1h]
            htf[i, 0] = row["rsi_n"]
            htf[i, 1] = row["ema21_d"]
            htf[i, 2] = row["macd_n"]
            htf[i, 3] = row["atr_n"]
        # 4h
        idx4h = sig4h.index.searchsorted(ts, side="left") - 1
        if idx4h >= 0:
            row = sig4h.iloc[idx4h]
            htf[i, 4] = row["rsi_n"]
            htf[i, 5] = row["ema21_d"]
            htf[i, 6] = row["trend"]
            htf[i, 7] = row["atr_n"]

    return htf


def apply_rolling_zscore(features: np.ndarray, window: int = ZSCORE_WIN, min_periods: int = 50) -> np.ndarray:
    """
    Per-feature rolling z-score normalisation.
    Computes mean/std over a backward window only — zero future leakage.
    Columns that are clearly binary/one-hot (low variance) are skipped.
    """
    df = pd.DataFrame(features.astype(np.float64))
    stds = df.std()
    skip_cols = stds[stds < 0.05].index.tolist()  # skip binary/constant columns

    roll_mean = df.rolling(window=window, min_periods=min_periods).mean()
    roll_std  = df.rolling(window=window, min_periods=min_periods).std()

    normed = (df - roll_mean) / (roll_std + 1e-8)
    normed[skip_cols] = df[skip_cols]  # restore binary columns as-is
    normed.fillna(0.0, inplace=True)

    return normed.values.astype(np.float32)


# =============================================================================
# LABEL GENERATION — MULTI-HORIZON
# =============================================================================

def _rolling_vol(close: np.ndarray, window: int = VOL_WINDOW) -> np.ndarray:
    """Rolling std of 1-bar fractional returns (per-bar volatility), with a robust
    fill for the warm-up region so barriers are always defined."""
    n = len(close)
    r1 = np.zeros(n, dtype=np.float64)
    r1[1:] = np.diff(close) / (close[:-1] + 1e-12)
    sig = pd.Series(r1).rolling(window, min_periods=5).std().bfill().to_numpy()
    finite = sig[np.isfinite(sig) & (sig > 0)]
    med = float(np.median(finite)) if finite.size else 0.005
    med = med if med > 0 else 0.005
    return np.where(np.isfinite(sig) & (sig > 0), sig, med)


def _triple_barrier_one(high, low, close, sig, h: int, k: float):
    """Vol-scaled first-touch triple barrier for one horizon.

    Barrier width (fractional) = k · per-bar-vol · √h. Scanning bars i+1..i+h:
      • upper (k·vol·√h above close) touched first  → 0 (long)
      • lower touched first                          → 1 (short)
      • neither (or both same bar) within h          → 2 (hold, time barrier)
    Returns (labels (N,) int64 with -1 tail mask, returns (N,) float32): the signed
    realized move = +barrier for longs, -barrier for shorts, close-return at h for holds.
    """
    n = len(close)
    lab = np.full(n, 2, dtype=np.int64)
    ret = np.full(n, np.nan, dtype=np.float32)
    if n <= h + 1:
        lab[max(0, n - h):] = -1
        return lab, ret

    m       = n - h                                   # bars with a full future window: i in [0, m)
    barrier = np.clip(k * sig[:m] * math.sqrt(h), 1e-5, None)   # fractional, per bar
    up      = close[:m] * (1.0 + barrier)
    dn      = close[:m] * (1.0 - barrier)
    # Future windows bars i+1..i+h (zero-copy views), aligned to bars 0..m-1.
    fut_high = sliding_window_view(high, h)[1:m + 1]   # (m, h)
    fut_low  = sliding_window_view(low,  h)[1:m + 1]
    long_hit, short_hit = fut_high >= up[:, None], fut_low <= dn[:, None]
    any_long, any_short = long_hit.any(axis=1), short_hit.any(axis=1)
    first_long  = np.where(any_long,  long_hit.argmax(axis=1),  h)   # h = "never touched"
    first_short = np.where(any_short, short_hit.argmax(axis=1), h)
    is_long, is_short = first_long < first_short, first_short < first_long

    lab[:m][is_long]  = 0
    lab[:m][is_short] = 1
    r = (close[h:h + m] - close[:m]) / (close[:m] + 1e-12)   # hold/time-barrier return
    r[is_long]  =  barrier[is_long]                          # realized TP/SL move for touches
    r[is_short] = -barrier[is_short]
    ret[:m] = r.astype(np.float32)
    lab[m:] = -1                                            # mask tail (no full future)
    return lab, ret


def triple_barrier_labels(df: pd.DataFrame, horizons: List[int], k: float = BARRIER_K):
    """Vol-scaled triple-barrier labels + matched signed returns for ALL horizons.

    Returns ``(labels (N,H) int64, returns (N,H) float32)``. Encoding 0=long,
    1=short, 2=hold; last ``h`` rows of each horizon masked to -1. Replaces the old
    fixed-threshold labels: barriers adapt to each asset's volatility and match how
    trades actually exit (take-profit / stop-loss / time). High/low fall back to
    close when OHLC isn't available (e.g. unit tests)."""
    close = df["close"].to_numpy(np.float64)
    high  = (df["high"] if "high" in df else df["close"]).to_numpy(np.float64)
    low   = (df["low"]  if "low"  in df else df["close"]).to_numpy(np.float64)
    sig   = _rolling_vol(close)
    n, H  = len(close), len(horizons)
    labels  = np.full((n, H), 2, dtype=np.int64)
    returns = np.full((n, H), np.nan, dtype=np.float32)
    for hi, h in enumerate(horizons):
        labels[:, hi], returns[:, hi] = _triple_barrier_one(high, low, close, sig, h, k)
    return labels, returns


def build_labels(df: pd.DataFrame, horizons: List[int], thresholds: List[float] = None) -> np.ndarray:
    """(N, H) int64 triple-barrier labels (0=long,1=short,2=hold; -1 tail mask).
    ``thresholds`` is kept for signature compatibility but ignored — barriers are
    now volatility-scaled (see ``triple_barrier_labels``)."""
    return triple_barrier_labels(df, horizons, BARRIER_K)[0]


def build_label_returns(df: pd.DataFrame, horizons: List[int]) -> np.ndarray:
    """(N, H) float32 signed realized returns matched to ``build_labels`` (for the
    PnL-magnitude loss weight): +barrier for longs, -barrier for shorts, close
    return at the horizon for holds; NaN tail matches the label mask."""
    return triple_barrier_labels(df, horizons, BARRIER_K)[1]


def pnl_magnitude_weight(returns_h: np.ndarray,
                          floor: float = 0.25, cap: float = 4.0) -> np.ndarray:
    """Phase 16: per-sample loss weight ∝ |future_return| relative to the
    horizon's median absolute return. Big moves contribute more gradient,
    chop contributes less — aligns offline CE with the live AWR PnL objective.

    Returns a float32 (N,) array clipped to ``[floor, cap]``.
    Uses median (not mean) so a few outliers don't crush the baseline."""
    abs_r = np.abs(np.asarray(returns_h, dtype=np.float64))
    finite = abs_r[np.isfinite(abs_r) & (abs_r > 0)]
    med = float(np.median(finite)) if finite.size else 1e-4
    med = max(med, 1e-8)
    w = abs_r / med
    w = np.clip(w, floor, cap)
    # Replace NaN/inf (masked rows) with 1.0 (neutral) — caller may also mask via labels.
    w = np.where(np.isfinite(w), w, 1.0)
    return w.astype(np.float32)


# =============================================================================
# SEQUENCE BUILDER
# =============================================================================

def build_sequences(
    features: np.ndarray,
    labels: np.ndarray,
    seq_len: int = SEQ_LEN,
    returns: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Slides a window of seq_len over features, labels last timestep.
    Drops any sequence where any horizon label is -1 (masked).

    Phase 16: when ``returns`` is provided (parallel (N, H) float matrix from
    ``build_label_returns``), also emits the matched (M, H) returns block so the
    DataLoader can supply per-sample PnL-magnitude weights to the loss.

    Returns X: (M, seq_len, F), y: (M, H), returns: (M, H) or None.
    """
    n = len(features)
    X, y, R = [], [], []

    for i in range(seq_len, n):
        row_labels = labels[i - 1]
        if np.any(row_labels == -1):
            continue
        X.append(features[i - seq_len: i])
        y.append(row_labels)
        if returns is not None:
            R.append(returns[i - 1])

    X_arr = np.array(X, dtype=np.float32)
    y_arr = np.array(y, dtype=np.int64)
    R_arr = np.array(R, dtype=np.float32) if returns is not None else None
    return X_arr, y_arr, R_arr


def _build_sequences_to_file(
    features: np.ndarray,
    labels: np.ndarray,
    returns: np.ndarray,
    out_path: str,
    seq_len: int = SEQ_LEN,
    chunk: int = 8192,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """RAM-safe sibling of ``build_sequences`` for the ``--mmap`` path.

    ``build_sequences`` materialises the WHOLE symbol as one float32 array
    (BTCUSDT ≈ 464k × 60 × 86 × 4B ≈ 9.6 GB), then ``np.save`` adds a float16
    copy on top — peaking ~14 GB and getting OOM-killed on Colab (the spurious
    "^C" after "Building sequences"). This streams the sliding windows straight
    into a float16 ``.npy`` memmap on disk in small chunks, so peak RAM is ~one
    chunk (~0.2 GB) regardless of symbol size.

    Masking, endpoint indexing and X/y/R alignment match ``build_sequences``
    exactly. Returns ``(y, R, n_seq)``; the X windows live on disk at ``out_path``.
    """
    from numpy.lib.stride_tricks import sliding_window_view

    n, feat_dim = features.shape
    H = labels.shape[1]
    empty_y = np.empty((0, H), np.int64)
    empty_r = np.empty((0, returns.shape[1]), np.float32)
    if n <= seq_len:
        return empty_y, empty_r, 0

    # Endpoint i in [seq_len, n): window = features[i-seq_len:i], label = labels[i-1].
    # Drop windows whose label row is masked (-1) — identical to build_sequences.
    ends   = np.arange(seq_len, n)
    keep   = ~np.any(labels[ends - 1] == -1, axis=1)
    ends   = ends[keep]
    m      = int(ends.shape[0])
    if m == 0:
        return empty_y, empty_r, 0

    y = labels[ends - 1].astype(np.int64)
    R = returns[ends - 1].astype(np.float32)

    feats32 = np.ascontiguousarray(features, dtype=np.float32)
    # Zero-copy view of every window: swv[j] == features[j:j+seq_len].T  (F, seq_len)
    swv     = sliding_window_view(feats32, seq_len, axis=0)
    starts  = ends - seq_len

    Xmm = np.lib.format.open_memmap(
        out_path, mode="w+", dtype=np.float16, shape=(m, seq_len, feat_dim))
    for s in range(0, m, chunk):
        e = min(s + chunk, m)
        # Gather only this chunk's windows into RAM, then store as float16.
        block = swv[starts[s:e]]                       # (cs, F, seq_len) float32
        Xmm[s:e] = block.transpose(0, 2, 1).astype(np.float16)
    Xmm.flush()
    del Xmm, swv, feats32
    return y, R, m


# =============================================================================
# MULTI-SYMBOL DATASET
# =============================================================================

def _recency_weights(timestamps, ref_ts,
                     halflife_years: float = RECENCY_HALFLIFE_YEARS,
                     floor: float = RECENCY_FLOOR) -> np.ndarray:
    """Per-sample loss weights with a CALENDAR half-life: a sample `halflife_years`
    old weighs 0.5×, two half-lives 0.25×, … floored at `floor` so older regimes
    are still learned (never fully ignored). Calendar-based (not index-based) so the
    same date across BTC/AMD/etc. gets the same emphasis — the model leans toward
    the current regime (post-COVID, the AI boom) consistently across assets.

    `timestamps` is an array of the per-sample label times; `ref_ts` is "now".
    """
    ts  = pd.to_datetime(np.asarray(timestamps)).values            # datetime64[ns]
    ref = pd.Timestamp(ref_ts).to_datetime64()
    age_years = (ref - ts) / np.timedelta64(365, "D")              # float64 years
    age_years = np.maximum(age_years.astype(np.float64), 0.0)
    w = np.power(0.5, age_years / max(float(halflife_years), 1e-6))
    return np.clip(w, float(floor), 1.0).astype(np.float32)


class _SeqDataset(Dataset):
    """Fork-safe streaming dataset for the float16 memmap path. ``X`` is either a
    ``.npy`` PATH (opened lazily inside each DataLoader worker, so the memmap is
    never pickled/duplicated across workers) or an in-RAM array (tests). Yields one
    sequence cast to float32 plus its label / future-return / symbol-id / recency
    weight — a 5-tuple — so RAM stays tiny regardless of dataset size.

    ``lo`` lets a train/val split point at a sub-range of the symbol's file while
    ``y/R/s/w`` are pre-sliced to match (so the per-symbol memmap is reused for
    both splits without copying)."""
    def __init__(self, X, y, R, s, w=None, lo: int = 0):
        self._x_path = str(X) if isinstance(X, (str, os.PathLike)) else None
        self._X      = None if self._x_path else X
        self.lo      = int(lo)
        self.y = torch.from_numpy(np.ascontiguousarray(y))
        self.R = torch.from_numpy(np.ascontiguousarray(R))
        self.s = torch.from_numpy(np.ascontiguousarray(s))
        if w is None:
            w = np.ones(len(self.y), dtype=np.float32)
        self.w = torch.from_numpy(np.ascontiguousarray(np.asarray(w, dtype=np.float32)))

    def _arr(self):
        if self._X is None:                      # opened once per worker process
            self._X = np.load(self._x_path, mmap_mode="r")
        return self._X

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        xi = np.asarray(self._arr()[self.lo + i], dtype=np.float32)
        return torch.from_numpy(xi), self.y[i], self.R[i], self.s[i], self.w[i]


def build_dataset(
    symbols: List[str],
    start_year: int,
    start_month: int,
    skip_download: bool = False,
    mmap_dir: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Downloads, engineers features, builds sequences for all symbols.

    Returns:
      X:           (M_total, SEQ_LEN, INPUT_SIZE)   # float32 in-RAM, or float16 memmap when mmap_dir is set
      y:           (M_total, len(HORIZONS))
      future_rets: (M_total, len(HORIZONS))  # Phase 16: for PnL-weighted loss
      sym_ids:     (M_total,)

    When ``mmap_dir`` is set, each symbol's sequences are written to a temporary
    float16 file and then assembled into a single on-disk memmap, so peak RAM is
    bounded by ONE symbol (not the whole dataset). Use a fast LOCAL disk
    (e.g. /content on Colab), not a mounted Drive.
    """
    use_mmap = bool(mmap_dir)
    if use_mmap:
        # Start clean: stale files from an interrupted run must not pile up and
        # exhaust the local disk (a full build is ≈ 41 GB of float16 sequences;
        # leftovers + a fresh build was overflowing Colab's 112 GB and halting).
        if os.path.isdir(mmap_dir):
            shutil.rmtree(mmap_dir, ignore_errors=True)
        os.makedirs(mmap_dir, exist_ok=True)

    ref_ts = pd.Timestamp.now()          # recency reference ("now") shared by all symbols
    parts: list = []                     # per-symbol dicts (mmap path) — see below
    all_X, all_y, all_r, all_sym, all_w = [], [], [], [], []

    for sym in symbols:
        sym_id = SYMBOL_TO_ID[sym]
        log.info("Processing symbol", symbol=sym)

        try:
            dfs = load_full_history(sym, start_year, start_month, skip_download)
        except RuntimeError as e:
            log.error("Skipping symbol — no data", symbol=sym, error=str(e))
            continue

        df5m = dfs["5m"]
        df1h = dfs["1h"]
        df4h = dfs["4h"]

        log.info("Building base features", symbol=sym, rows=len(df5m))
        base_feats = build_feature_matrix(df5m)           # (N, 62)
        base_feats = detect_regime(df5m, base_feats)      # fills regime slots

        log.info("Building HTF features", symbol=sym)
        htf_feats  = build_htf_features(df5m, df1h, df4h) # (N, 8)

        combined   = np.concatenate([base_feats, htf_feats], axis=1)  # (N, 70)

        log.info("Normalising", symbol=sym)
        combined   = apply_rolling_zscore(combined)

        # Phase 3: append the semantic NEWS_EMBED block AFTER z-scoring, so the
        # raw L2-normalized embeddings match exactly what the live agent inserts
        # (the live builder does not z-score the embed block).
        log.info("Aligning news embeddings", symbol=sym)
        news_embed = build_news_embed_matrix(df5m, sym)               # (N, 16) raw or zeros
        combined   = np.concatenate([combined, news_embed], axis=1)   # (N, 86)

        # Cycle 7: earnings-calendar block (stocks only; zeros otherwise) → (N, 90).
        earnings   = build_earnings_matrix(df5m, sym)                 # (N, 4) raw or zeros
        combined   = np.concatenate([combined, earnings], axis=1)     # (N, 90)

        log.info("Building labels (triple-barrier, vol-scaled)", symbol=sym)
        # One pass → matched labels + signed returns (first-touch TP/SL/time barriers).
        labels, returns = triple_barrier_labels(df5m, HORIZONS, BARRIER_K)

        # Recency weight per surviving sample — computed from the SAME valid mask
        # the sequence builders apply (endpoints in [SEQ_LEN, N) whose label row
        # isn't masked), so it stays row-aligned with X/y/R without touching the
        # builders. `ends-1` is each sample's label timestamp.
        ends   = np.arange(SEQ_LEN, len(labels))
        keep   = ~np.any(labels[ends - 1] == -1, axis=1)
        ends   = ends[keep]
        w_rec  = _recency_weights(df5m["timestamp"].to_numpy()[ends - 1], ref_ts)

        log.info("Building sequences", symbol=sym)
        if use_mmap:
            # RAM-safe: stream this symbol's windows straight to a float16 file
            # (peak RAM ~one chunk) instead of materialising the whole symbol as
            # float32 (BTC ≈ 9.6 GB → Colab OOM). The per-symbol file is KEPT and
            # trained from directly — no 41 GB assembled second copy, so peak disk
            # is ~½ and the chronological train/val split stays per-symbol.
            p = os.path.join(mmap_dir, f"_X_{sym}.npy")
            y, R, n_seq = _build_sequences_to_file(combined, labels, returns, p)
            if n_seq == 0:
                log.warning("No sequences produced", symbol=sym)
                continue
            assert n_seq == len(w_rec), (sym, n_seq, len(w_rec))   # alignment guard
            parts.append({"path": p, "n": n_seq, "y": y, "R": R,
                          "s": np.full(n_seq, sym_id, np.int64), "w": w_rec})
            log.info("Symbol done", symbol=sym, sequences=n_seq,
                     class_dist_h0=str(np.bincount(y[:, 0])))
        else:
            X, y, R = build_sequences(combined, labels, returns=returns)
            if len(X) == 0:
                log.warning("No sequences produced", symbol=sym)
                continue
            assert len(X) == len(w_rec), (sym, len(X), len(w_rec))
            all_X.append(X); all_y.append(y); all_r.append(R)
            all_sym.append(np.full(len(X), sym_id, np.int64)); all_w.append(w_rec)
            log.info("Symbol done", symbol=sym, sequences=len(X),
                     class_dist_h0=str(np.bincount(y[:, 0])))

    # Tagged return so make_split_loaders knows which path to take:
    #   ("parts", [ {path, n, y, R, s, w} per symbol ])  — low-disk memmap path
    #   ("array", X, y, R, sym_ids, w)                    — in-RAM path (tests/local)
    if use_mmap:
        if not parts:
            raise RuntimeError("No symbols produced any sequences")
        total = sum(p["n"] for p in parts)
        gb    = total * SEQ_LEN * INPUT_SIZE * 2 / 1e9
        log.info("Dataset ready (per-symbol float16 memmaps — no assembly copy)",
                 total_sequences=total, approx_gb=round(gb, 2), symbols=len(parts))
        return ("parts", parts)
    if not all_X:
        raise RuntimeError("No symbols produced any sequences")
    X_all = np.concatenate(all_X, axis=0)
    log.info("Full dataset assembled (in-RAM)", total_sequences=len(X_all))
    return ("array", X_all, np.concatenate(all_y), np.concatenate(all_r),
            np.concatenate(all_sym), np.concatenate(all_w))


def make_split_loaders(build_out, batch_size: int, val_frac: float = 0.2, embargo: int = None):
    """Train/val DataLoaders with a PER-SYMBOL chronological split: the last
    ``val_frac`` of EACH symbol becomes validation. This fixes the old global-tail
    split (which put the dataset's last rows — entirely the stock symbols — into
    val, leaving crypto unvalidated and leaking symbol order). Every batch is a
    5-tuple ``(X, y, future_returns, symbol_id, recency_weight)``.

    ``embargo`` purges the last ``embargo`` TRAIN rows at each symbol's boundary so
    a train sample's label window (which peeks up to ``max(HORIZONS)`` candles
    ahead) can't overlap the val period — otherwise val loss is optimistic. Defaults
    to ``max(HORIZONS)``; pass 0 to disable.

    Returns ``(train_loader, val_loader, y_train, n_train, n_val)``. Handles both
    ``build_dataset`` outputs: ``"parts"`` trains straight off the per-symbol
    float16 memmaps (low disk/RAM); ``"array"`` uses in-RAM TensorDatasets.
    """
    if embargo is None:
        embargo = max(HORIZONS)
    embargo = int(max(0, embargo))
    kind = build_out[0]
    train_sets, val_sets, y_tr_blocks = [], [], []

    if kind == "parts":
        for pt in build_out[1]:
            n = pt["n"]; cut = max(1, min(n - 1, int(n * (1.0 - val_frac))))
            tr_hi = max(1, cut - embargo)         # purge label-overlap rows at the boundary
            # One float16 file feeds BOTH splits via the `lo` offset (no copy).
            train_sets.append(_SeqDataset(pt["path"], pt["y"][:tr_hi], pt["R"][:tr_hi],
                                          pt["s"][:tr_hi], pt["w"][:tr_hi], lo=0))
            val_sets.append(_SeqDataset(pt["path"], pt["y"][cut:], pt["R"][cut:],
                                        pt["s"][cut:], pt["w"][cut:], lo=cut))
            y_tr_blocks.append(pt["y"][:tr_hi])
        num_workers = 2
    elif kind == "array":
        _, X, y, R, s, w = build_out

        def _tensor_ds(ix):
            return TensorDataset(
                torch.from_numpy(np.ascontiguousarray(X[ix]).astype(np.float32)),
                torch.from_numpy(np.ascontiguousarray(y[ix])),
                torch.from_numpy(np.ascontiguousarray(R[ix])),
                torch.from_numpy(np.ascontiguousarray(s[ix])),
                torch.from_numpy(np.ascontiguousarray(w[ix]).astype(np.float32)),
            )

        for sid in np.unique(s):
            idx = np.where(s == sid)[0]
            cut = max(1, min(len(idx) - 1, int(len(idx) * (1.0 - val_frac))))
            tr_hi = max(1, cut - embargo)
            train_sets.append(_tensor_ds(idx[:tr_hi]))
            val_sets.append(_tensor_ds(idx[cut:]))
            y_tr_blocks.append(y[idx[:tr_hi]])
        num_workers = 0
    else:
        raise ValueError(f"Unknown build_dataset output kind: {kind!r}")

    train_ds = ConcatDataset(train_sets)
    val_ds   = ConcatDataset(val_sets)
    y_train  = np.concatenate(y_tr_blocks, axis=0)
    persist  = num_workers > 0          # keep workers (+ their lazy memmaps) across epochs
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              pin_memory=True, num_workers=num_workers,
                              persistent_workers=persist)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              pin_memory=True, num_workers=num_workers,
                              persistent_workers=persist)
    return train_loader, val_loader, y_train, len(train_ds), len(val_ds)


# =============================================================================
# TRAINING UTILITIES
# =============================================================================

def make_weighted_loss(y_train: np.ndarray, *, per_sample: bool = True) -> List[nn.CrossEntropyLoss]:
    """One weighted CrossEntropyLoss per horizon, computed from training split.

    Phase 16: ``per_sample=True`` returns ``reduction='none'`` losses so the
    caller can multiply by a PnL-magnitude weight before averaging."""
    loss_fns = []
    reduction = "none" if per_sample else "mean"
    for h in range(len(HORIZONS)):
        labels_h = y_train[:, h]
        counts   = np.bincount(labels_h, minlength=NUM_CLASSES).astype(np.float64)
        counts   = np.where(counts == 0, 1, counts)
        weights  = 1.0 / counts
        weights /= weights.sum()
        w_tensor = torch.FloatTensor(weights)
        # A4 anti-overfit: label smoothing softens the hard targets.
        loss_fns.append(nn.CrossEntropyLoss(
            weight=w_tensor, reduction=reduction, label_smoothing=LABEL_SMOOTHING))
        log.info(f"Horizon {HORIZONS[h]}: class weights = {weights.round(4)} "
                 f"(reduction={reduction}, label_smoothing={LABEL_SMOOTHING})")
    return loss_fns


def _apply_horizon_loss(loss_fn, logits, targets, sample_weights):
    """Helper: compute a single-horizon loss honouring per-sample weighting + focal.

    When ``loss_fn`` was built with ``reduction='none'`` it returns a ``(B,)``
    per-sample (class-weighted, label-smoothed) CE. We then:
      • apply a focal modulation ``(1 - p_true)^FOCAL_GAMMA`` so easy/abundant
        'hold' samples contribute less and the rare, hard directional moves drive
        learning (FOCAL_GAMMA = 0 disables it → plain weighted CE);
      • multiply by the per-sample weight (PnL-magnitude × recency) and average.
    Falls back gracefully when ``sample_weights`` is None."""
    raw = loss_fn(logits, targets)              # (B,) if reduction='none', else scalar
    if raw.dim() == 0:
        return raw                              # already reduced (mean) — no focal/weights
    if FOCAL_GAMMA > 0:
        # p_true = softmax(logits)[true class]; gradients flow through it (standard focal).
        p_true = torch.softmax(logits, dim=1).gather(1, targets.unsqueeze(1)).squeeze(1)
        raw = (1.0 - p_true).clamp_min(0.0).pow(FOCAL_GAMMA) * raw
    if sample_weights is None:
        return raw.mean()
    return (raw * sample_weights).mean()


def train_epoch(model, loader, optimizer, loss_fns, device, scaler=None, use_amp=False):
    model.train()
    total_loss = 0.0
    n_batches  = 0

    for batch in loader:
        # 5-tuple: X, y, future_returns (B,H), symbol_ids, recency_weight (B,)
        bx, by, br, bs, bw = batch
        bx = bx.to(device, non_blocking=True); by = by.to(device, non_blocking=True)
        br = br.to(device, non_blocking=True); bs = bs.to(device, non_blocking=True)
        bw = bw.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            out = model(bx, bs)
            logits_list = out[0]   # [0]=logits per horizon (4-/5-tuple tolerant)

            loss = 0.0
            for h_idx, (logits, loss_fn) in enumerate(zip(logits_list, loss_fns)):
                loss_fn_d = loss_fn.to(device)
                # Per-sample PnL-magnitude weight for this horizon …
                abs_r = br[:, h_idx].abs()
                med   = abs_r[abs_r > 0].median() if (abs_r > 0).any() else torch.tensor(1e-4, device=device)
                w_pnl = torch.clamp(abs_r / (med + 1e-8), 0.25, 4.0)
                w     = w_pnl * bw                      # … × calendar recency weight
                loss  = loss + _apply_horizon_loss(loss_fn_d, logits, by[:, h_idx], w)
            loss = loss / len(HORIZONS)

        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)                  # unscale before clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item()
        n_batches  += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def eval_epoch(model, loader, loss_fns, device, use_amp=False):
    model.eval()
    total_loss = 0.0
    n_batches  = 0
    all_preds  = [[] for _ in HORIZONS]
    all_labels = [[] for _ in HORIZONS]
    all_probs  = [[] for _ in HORIZONS]

    for batch in loader:
        bx, by, br, bs, bw = batch
        bx = bx.to(device, non_blocking=True); by = by.to(device, non_blocking=True)
        br = br.to(device, non_blocking=True); bs = bs.to(device, non_blocking=True)
        bw = bw.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            out = model(bx, bs)
            logits_list, probs_list = out[0], out[1]

            loss = 0.0
            for h_idx, (logits, probs, loss_fn) in enumerate(zip(logits_list, probs_list, loss_fns)):
                abs_r = br[:, h_idx].abs()
                med   = abs_r[abs_r > 0].median() if (abs_r > 0).any() else torch.tensor(1e-4, device=device)
                w     = torch.clamp(abs_r / (med + 1e-8), 0.25, 4.0) * bw
                loss  = loss + _apply_horizon_loss(loss_fn.to(device), logits, by[:, h_idx], w)
                all_preds[h_idx].extend(logits.argmax(dim=1).cpu().numpy())
                all_labels[h_idx].extend(by[:, h_idx].cpu().numpy())
                all_probs[h_idx].extend(probs.float().cpu().numpy())   # .float(): AMP-safe

        total_loss += (loss / len(HORIZONS)).item()
        n_batches  += 1

    return (
        total_loss / max(n_batches, 1),
        [np.array(p) for p in all_preds],
        [np.array(l) for l in all_labels],
        [np.array(p) for p in all_probs],
    )


def compute_metrics(preds: np.ndarray, labels: np.ndarray, probs: np.ndarray, horizon: int):
    """Prints accuracy, classification report, expectancy and a Sharpe proxy."""
    acc = (preds == labels).mean()
    log.info(f"\n=== Horizon +{horizon} candles === Accuracy: {acc:.4f}")
    print(classification_report(labels, preds, target_names=["long", "short", "hold"], zero_division=0))

    # Expectancy
    # Treat predicted long/short confidence as bet size proxy
    conf       = np.max(probs, axis=1)
    is_trade   = (preds != 2)
    correct    = (preds == labels) & is_trade
    wrong      = (preds != labels) & is_trade
    win_conf   = conf[correct].mean() if correct.any() else 0
    loss_conf  = conf[wrong].mean()   if wrong.any()   else 0
    win_rate   = correct.sum() / (is_trade.sum() + 1e-8)
    expectancy = win_rate * win_conf - (1 - win_rate) * loss_conf

    # Sharpe proxy: treat each correct trade as +1 return, wrong as -1
    returns = np.where(correct, 1.0, np.where(wrong, -1.0, 0.0))
    sharpe  = returns.mean() / (returns.std() + 1e-8) * np.sqrt(252)

    log.info(f"Expectancy: {expectancy:.4f}  |  Sharpe proxy: {sharpe:.4f}  |  "
             f"Win rate: {win_rate:.4f}  |  Trade pct: {is_trade.mean():.4f}")


# =============================================================================
# CHECKPOINT
# =============================================================================

def save_checkpoint(model, optimizer, epoch, val_loss, path: Path, label="checkpoint"):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch":                epoch,
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss":             val_loss,
        "feature_version":      FEATURE_VERSION,
        "input_size":           INPUT_SIZE,
        "seq_len":              SEQ_LEN,
        "horizons":             HORIZONS,
        "symbols":              SYMBOLS,
        "label":                label,
        "news_backend":         _NEWS_BACKEND_USED,   # live verifies it matches (Cycle 3)
        "earnings_aligned":     _EARNINGS_ALIGNED,    # Cycle 7
        "trunk_type":           getattr(model, "trunk_type", "lstm"),   # Cycle 8 (lstm|tcn)
    }, path)
    log.info("Checkpoint saved", path=str(path), epoch=epoch, val_loss=f"{val_loss:.4f}",
             news_backend=_NEWS_BACKEND_USED, earnings_aligned=_EARNINGS_ALIGNED,
             trunk=getattr(model, "trunk_type", "lstm"))


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Optimised pretraining pipeline v2")
    parser.add_argument("--start-year",    type=int, default=2020)
    parser.add_argument("--start-month",   type=int, default=1)
    parser.add_argument("--epochs",        type=int, default=30)
    parser.add_argument("--batch-size",    type=int, default=BATCH_SIZE)
    parser.add_argument("--lr",            type=float, default=LR)
    parser.add_argument("--patience",      type=int, default=5,
                        help="Early stopping patience (epochs)")
    parser.add_argument("--seq-len",       type=int, default=SEQ_LEN)
    parser.add_argument("--skip-download", action="store_true",
                        help="Reuse cached parquet files — skip all HTTP requests")
    parser.add_argument("--symbols",       nargs="+", default=SYMBOLS,
                        help="Subset of symbols to train on")
    parser.add_argument("--mmap", action="store_true",
                        help="Stream the dataset from a float16 disk memmap so ALL symbols × "
                             "multiple years fit in low RAM (e.g. Colab's 12.7 GB).")
    parser.add_argument("--mmap-dir", default="/content/_mmap_cache",
                        help="Local (fast) disk dir for the memmap — NOT a mounted Drive.")
    parser.add_argument("--amp", action="store_true",
                        help="Mixed-precision (fp16) training on CUDA — ~2-3× faster on "
                             "L4/A100 tensor cores + lower VRAM. Ignored on CPU.")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed. Train several seeds (0,1,2,…) for an ensemble; "
                             "non-default seeds save to seed-tagged checkpoints.")
    args = parser.parse_args()

    # ── reproducibility ──────────────────────────────────────────────────────
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device", device=str(device))

    # ── validate symbols ─────────────────────────────────────────────────────
    requested = [s.upper() for s in args.symbols]
    unknown   = [s for s in requested if s not in SYMBOL_TO_ID]
    if unknown:
        log.warning("Unknown symbols — will skip", unknown=unknown)
    symbols = [s for s in requested if s in SYMBOL_TO_ID]
    log.info("Training symbols", symbols=symbols)

    # ── build dataset ────────────────────────────────────────────────────────
    log.info("Building dataset", start=f"{args.start_year}-{args.start_month:02d}", mmap=bool(args.mmap))
    build_out = build_dataset(
        symbols, args.start_year, args.start_month, args.skip_download,
        mmap_dir=(args.mmap_dir if args.mmap else None),
    )

    # ── per-symbol chronological split + 5-tuple recency-weighted loaders ─────
    use_amp = bool(getattr(args, "amp", False)) and device.type == "cuda"
    train_loader, val_loader, y_tr, n_tr, n_va = make_split_loaders(
        build_out, args.batch_size, val_frac=0.2,
    )
    log.info("Split sizes", train=n_tr, val=n_va, amp=use_amp)
    for hi, h in enumerate(HORIZONS):
        log.info(f"H+{h} train class dist: {np.bincount(y_tr[:, hi])}")

    # ── model ────────────────────────────────────────────────────────────────
    model = ImprovedTradingLSTM(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LSTM_LAYERS,
        dropout=DROPOUT,
        num_symbols=len(SYMBOLS),
        symbol_embed_dim=SYMBOL_EMBED_DIM,
        num_horizons=len(HORIZONS),
        num_classes=NUM_CLASSES,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("Model initialised", trainable_params=n_params)

    # ── loss functions ───────────────────────────────────────────────────────
    loss_fns = make_weighted_loss(y_tr)

    # ── optimiser + scheduler ────────────────────────────────────────────────
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    # NOTE: no verbose= — it was deprecated and REMOVED in PyTorch 2.3+ (Colab
    # ships a newer torch); passing it raises TypeError. LR changes still apply.
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=3, factor=0.5
    )

    # ── mixed-precision scaler (CUDA only) — ~2-3× faster on L4/A100 tensor
    #    cores + lower VRAM, letting you push a larger --batch-size. fp32 fallback
    #    is automatic when --amp is off or on CPU. ──────────────────────────────
    scaler = None
    if use_amp:
        try:
            scaler = torch.amp.GradScaler("cuda")          # PyTorch 2.3+ API
        except (AttributeError, TypeError):
            scaler = torch.cuda.amp.GradScaler()           # older fallback
    log.info("Mixed precision (AMP)", enabled=use_amp)

    # ── paths ────────────────────────────────────────────────────────────────
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    # Default seed keeps the canonical name the live agent loads; other seeds get
    # tagged files so an ensemble's members don't overwrite each other.
    tag         = "" if args.seed == 42 else f"_seed{args.seed}"
    best_path   = MODELS_DIR / f"pretrain_v2_best{tag}.pt"
    latest_path = MODELS_DIR / f"trading_lstm_latest{tag}.pt"

    # ── training loop ────────────────────────────────────────────────────────
    best_val_loss    = float("inf")
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        t0         = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, loss_fns, device,
                                 scaler=scaler, use_amp=use_amp)
        val_loss, preds_list, labels_list, probs_list = eval_epoch(
            model, val_loader, loss_fns, device, use_amp=use_amp
        )
        scheduler.step(val_loss)
        elapsed = time.time() - t0

        # Primary horizon accuracy for quick read
        h0_acc = (preds_list[0] == labels_list[0]).mean()

        log.info(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train={train_loss:.4f}  val={val_loss:.4f}  "
            f"h0_acc={h0_acc:.4f}  lr={optimizer.param_groups[0]['lr']:.2e}  "
            f"t={elapsed:.1f}s"
        )

        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            patience_counter = 0
            save_checkpoint(model, optimizer, epoch, val_loss, best_path, label="best")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                log.info("Early stopping triggered", patience=args.patience)
                break

    # ── final metrics on validation set ──────────────────────────────────────
    log.info("\n===== FINAL VALIDATION METRICS =====")
    _, preds_list, labels_list, probs_list = eval_epoch(
        model, val_loader, loss_fns, device, use_amp=use_amp
    )
    for hi, h in enumerate(HORIZONS):
        compute_metrics(preds_list[hi], labels_list[hi], probs_list[hi], h)

    # ── promote best to latest ───────────────────────────────────────────────
    shutil.copy(best_path, latest_path)
    log.info("Best checkpoint promoted to latest", path=str(latest_path))
    log.info("Pretraining complete", best_val_loss=f"{best_val_loss:.4f}")


if __name__ == "__main__":
    main()






































'''import sys
from pathlib import Path
import time
import requests
import numpy as np
import pandas as pd
import math
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Add project root to path
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

from backend.signals.technical import build_technical_feature_dict
from backend.agents.nn_model import TradingLSTM
from structlog import get_logger

log = get_logger("scripts.pretrain")


def fetch_binance_data(symbol: str = "BTCUSDT", interval: str = "5m", limit: int = 8640) -> pd.DataFrame:
    """Fetch recent K-lines from Binance"""
    log.info("Fetching data from Binance", symbol=symbol, limit=limit)
    # Public data mirror — api.binance.com is geo-blocked (HTTP 451) from US/cloud IPs.
    url = "https://data-api.binance.vision/api/v3/klines"
    
    # Binance limits to 1000 per request. We loop backwards.
    all_data = []
    end_time = None
    
    needed = limit
    while needed > 0:
        batch_limit = min(needed, 1000)
        params = {"symbol": symbol, "interval": interval, "limit": batch_limit}
        if end_time:
            params["endTime"] = end_time
            
        res = requests.get(url, params=params)
        res.raise_for_status()
        data = res.json()
        
        if not data:
            break
            
        all_data = data + all_data
        end_time = data[0][0] - 1  # end before the first candle of this batch
        needed -= len(data)
        
        log.info(f"Fetched {len(data)} candles, {needed} remaining...")
        time.sleep(0.5)

    
    df = pd.DataFrame(all_data, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
    ])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
        
    return df.sort_values("timestamp").reset_index(drop=True)

def build_vector(idx: int, df: pd.DataFrame) -> np.ndarray:
    """Builds a 62-element feature vector for a specific index in df (simulating realtime context)."""
    # For technicals, we need to pass a slice of df to build_technical_feature_dict
    # Need at least 200 bars for some MAs/ATRs
    start_idx = max(0, idx - 250)
    sub_df = df.iloc[start_idx:idx+1].copy()
    
    tech = build_technical_feature_dict(sub_df)
    
    vec = np.zeros(62, dtype=np.float32)
    
    try:
        # 0-2: prices
        op, hi, lo, cl = sub_df.iloc[-1][["open", "high", "low", "close"]]
        vec[0] = np.clip((cl - op) / op, -0.05, 0.05) * 20.0  # scaled -1, 1
        vec[1] = np.clip((hi - op) / op, 0.0, 0.05) * 20.0
        vec[2] = np.clip((lo - op) / op, -0.05, 0.0) * 20.0
    except:
        pass
        
    vec[3] = tech.get("volume_ratio", 0.0)
    vec[4] = 0.5 # spread_pct neutral
    
    vec[5] = tech.get("ema_9_dist", 0.0)
    vec[6] = tech.get("ema_21_dist", 0.0)
    vec[7] = tech.get("ema_50_dist", 0.0)
    vec[8] = tech.get("ema_200_dist", 0.0)
    vec[9] = tech.get("golden_cross", 0.0)
    vec[10] = tech.get("vwap_dist", 0.0)
    vec[11] = tech.get("rsi", 0.0)
    vec[12] = tech.get("macd_norm", 0.0)
    vec[13] = tech.get("macd_hist_norm", 0.0)
    vec[14] = tech.get("stoch_rsi", 0.0)
    vec[15] = tech.get("adx_norm", 0.0)
    vec[16] = tech.get("rsi_divergence", 0.0)
    vec[17] = tech.get("atr_norm", 0.0)
    vec[18] = tech.get("bb_width_norm", 0.0)
    vec[19] = tech.get("bb_pct_b", 0.0)
    vec[20] = tech.get("volume_ratio", 0.0)
    vec[21] = tech.get("obv_slope", 0.0)
    vec[22] = tech.get("fib_nearest_level_pct", 0.0)
    vec[23] = tech.get("fib_distance", 0.0)
    vec[24] = tech.get("fib_strength", 0.0)
    
    patterns = tech.get("pattern_flags", [0.0]*10)
    for i in range(10):
        if i < len(patterns):
            vec[25 + i] = patterns[i]
            
    # Orderbook neutrals
    vec[35:43] = 0.5 # 0.5 or 0.0? features.py uses 0.0 for missing
    
    # Regime: let's assume ranging [0,0,1,0,0,0]
    vec[45] = 1.0 
    
    # News neutrals
    vec[49:53] = 0.0
    
    # Macro neutrals
    vec[53] = 0.5 
    vec[54] = 0.5 
    vec[55] = 0.0 
    vec[56] = 0.0
    
    # Time (mock hour based on row index - just placeholder)
    now = df.iloc[idx]["timestamp"]
    if isinstance(now, pd.Timestamp):
        vec[57] = math.sin(2 * math.pi * now.hour / 24.0)
        vec[58] = math.cos(2 * math.pi * now.hour / 24.0)
        vec[59] = math.sin(2 * math.pi * now.weekday() / 7.0)
        vec[60] = math.cos(2 * math.pi * now.weekday() / 7.0)
    
    vec[61] = 0.5 # regime_confidence
    return vec


def test_trading_lstm():
    df = fetch_binance_data(limit=8640)
    log.info("Computing features for each candle (this will take a minute...)")
    
    # We need to compute vectors and labels
    # Label: Look 3 candles ahead (15 minutes)
    # If future_close > current * 1.005 -> 0, < 0.995 -> 1, else 2.
    
    features_list = []
    labels = []
    
    # Pre-calculate features to speed up
    # However we need history per row. Wait, since build_technical_feature_dict is stateless and applies pandas_ta
    # across the entire df, we can just compute it over the WHOLE dataframe ONCE, then extract rows.
    # The prompt explicitly asks to use build_technical_feature_dict() "for each candle".
    # But running it 8640 times on slices would be very O(N^2) slow. Let's run it once on the full DF and extract rows if possible.
    # WAIT! build_technical_feature_dict() returns iloc[-1] only natively!
    # "All functions take a pandas DataFrame... and return the most recent bar's values only (iloc[-1])." -> Prompt 5.
    
    # Ok, let's run it continuously from idx 250 to end.
    n = len(df)
    valid_start = 250
    for i in range(valid_start, n - 3):
        if i % 1000 == 0:
            log.info(f"Processed {i}/{n} candles...")
        
        vec = build_vector(i, df)
        features_list.append(vec)
        
        # Determine label
        current_close = df.iloc[i]["close"]
        future_close = df.iloc[i+3]["close"]
        if future_close > current_close * 1.005:
            labels.append(0)
        elif future_close < current_close * 0.995:
            labels.append(1)
        else:
            labels.append(2)

    features_matrix = np.array(features_list)
    labels_array = np.array(labels)
    
    # Print feature stats
    means = features_matrix.mean(axis=0)
    stds = features_matrix.std(axis=0)
    
    log.info("Feature stats:")
    for i in range(62):
        if stds[i] < 0.01:
            log.warning(f"Feature {i} has low variance: std = {stds[i]:.4f}")
            
    # Build Sequence Dataset (SEQUENCE_LENGTH = 60, step = 1)
    seq_len = 60
    X_seq = []
    y_seq = []
    
    for i in range(len(features_matrix) - seq_len):
        X_seq.append(features_matrix[i:i+seq_len])
        y_seq.append(labels_array[i+seq_len-1])
        
    X_seq = np.array(X_seq, dtype=np.float32)
    y_seq = np.array(y_seq, dtype=np.int64)
    
    log.info(f"Sequence dataset shape: {X_seq.shape}")
    
    # Train/Val Split (80/20 Chronological)
    split_idx = int(len(X_seq) * 0.8)
    X_train, y_train = X_seq[:split_idx], y_seq[:split_idx]
    X_val, y_val = X_seq[split_idx:], y_seq[split_idx:]
    
    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_dataset = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    log.info(f"Class distribution - Train: {np.bincount(y_train)}, Val: {np.bincount(y_val)}")
    
    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TradingLSTM().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    epochs = 30
    best_val_loss = float("inf")
    
    models_dir = Path(root_dir) / "models"
    models_dir.mkdir(exist_ok=True)
    best_model_path = models_dir / "pretrain_best.pt"
    latest_model_path = models_dir / "trading_lstm_latest.pt"
    
    log.info(f"Starting training on {device}...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            
            probs, size = model(bx)
            # Use negative log likelihood for pre-softmaxed probs
            loss = F.nll_loss(torch.log(probs + 1e-8), by)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item() * bx.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        # Eval
        model.eval()
        val_loss = 0.0
        correct = 0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                probs, size = model(bx)
                
                loss = F.nll_loss(torch.log(probs + 1e-8), by)
                val_loss += loss.item() * bx.size(0)
                
                preds = probs.argmax(dim=1)
                correct += (preds == by).sum().item()
                
        val_loss /= len(val_loader.dataset)
        val_acc = correct / len(val_loader.dataset)
        
        log.info(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            log.info(f"New best model! Saving to {best_model_path}...")
            torch.save(model.state_dict(), best_model_path)
            
    # Load best model and copy
    log.info(f"Training complete. Copying best model to {latest_model_path}")
    import shutil
    shutil.copy(best_model_path, latest_model_path)
    
    # Optionally print extra summary
    log.info("Pretraining pipeline finished successfully!")

if __name__ == "__main__":
    test_trading_lstm()
'''