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
import pandas as pd
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from structlog import get_logger
from torch.utils.data import DataLoader, TensorDataset, Dataset

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
except Exception:
    WEIGHT_DECAY = 1e-4
    LABEL_SMOOTHING = 0.05

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
    url = "https://api.binance.com/api/v3/klines"
    all_rows = []
    start_ms = int(after_ts.timestamp() * 1000) + 1

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

    if not all_rows:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    df = pd.DataFrame(all_rows, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades",
        "taker_buy_base", "taker_buy_quote", "ignore"
    ])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)
    return df[["timestamp", "open", "high", "low", "close", "volume"]]


# =============================================================================
# Stock history via Alpaca (Phase 7b)
# =============================================================================

_ALPACA_DATA_BASE = "https://data.alpaca.markets/v2"
_ALPACA_TF = {"5m": "5Min", "1h": "1Hour", "4h": "4Hour"}

# Symbol classification — mirror backend/core/universe.STOCK_UNDERLYINGS.
_STOCK_SYMBOLS = {"SNDK", "AMD", "MU", "AXTI", "BE"}


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

def build_labels(df: pd.DataFrame, horizons: List[int], thresholds: List[float]) -> np.ndarray:
    """
    Returns (N, len(horizons)) int64 label matrix.
    Label encoding: 0=long, 1=short, 2=hold

    Each horizon h uses threshold t:
      future_return > +t  → 0 (long)
      future_return < -t  → 1 (short)
      else                → 2 (hold)

    Last max(horizons) rows are masked to -1 (ignored in loss).
    """
    cl  = df["close"].values.astype(np.float64)
    n   = len(cl)
    out = np.full((n, len(horizons)), 2, dtype=np.int64)  # default: hold

    for hi, (h, t) in enumerate(zip(horizons, thresholds)):
        for i in range(n - h):
            ret = (cl[i + h] - cl[i]) / (cl[i] + 1e-8)
            if ret > t:
                out[i, hi] = 0
            elif ret < -t:
                out[i, hi] = 1
            else:
                out[i, hi] = 2

        out[n - h:, hi] = -1  # mask — can't compute future

    return out


def build_label_returns(df: pd.DataFrame, horizons: List[int]) -> np.ndarray:
    """Phase 16: parallel to build_labels — emit the actual signed future_return
    per (sample, horizon). Used for PnL-magnitude weighting in the loss.

    Returns (N, len(horizons)) float32 with NaN for the last `h` rows of each
    horizon (matching the label mask). Sequence builder filters these out.
    """
    cl = df["close"].values.astype(np.float64)
    n = len(cl)
    out = np.full((n, len(horizons)), np.nan, dtype=np.float32)
    for hi, h in enumerate(horizons):
        for i in range(n - h):
            out[i, hi] = float((cl[i + h] - cl[i]) / (cl[i] + 1e-8))
    return out


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


# =============================================================================
# MULTI-SYMBOL DATASET
# =============================================================================

class _SeqDataset(Dataset):
    """Streaming dataset for the float16 memmap path: reads one sequence from
    disk per item and casts to float32, so RAM stays tiny regardless of dataset
    size (lets Colab train ALL symbols × multiple years in 12.7 GB)."""
    def __init__(self, X, y, R, s):
        self.X = X                              # memmap float16 (or ndarray)
        self.y = torch.from_numpy(np.ascontiguousarray(y))
        self.R = torch.from_numpy(np.ascontiguousarray(R))
        self.s = torch.from_numpy(np.ascontiguousarray(s))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        xi = np.asarray(self.X[i], dtype=np.float32)
        return torch.from_numpy(xi), self.y[i], self.R[i], self.s[i]


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
        os.makedirs(mmap_dir, exist_ok=True)
    all_X, all_y, all_r, all_sym = [], [], [], []
    tmp_paths: list = []   # (path, n) per symbol when memmapping
    total = 0
    feat_dim = INPUT_SIZE

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

        log.info("Building labels", symbol=sym)
        labels     = build_labels(df5m, HORIZONS, THRESHOLDS)
        # Phase 16: parallel signed future returns for PnL-magnitude weighting
        returns    = build_label_returns(df5m, HORIZONS)

        log.info("Building sequences", symbol=sym)
        X, y, R = build_sequences(combined, labels, returns=returns)

        if len(X) == 0:
            log.warning("No sequences produced", symbol=sym)
            continue

        sym_ids = np.full(len(X), sym_id, dtype=np.int64)

        if use_mmap:
            # Write this symbol's sequences to a temp float16 file and free RAM.
            feat_dim = X.shape[-1]
            p = os.path.join(mmap_dir, f"_X_{sym}.npy")
            np.save(p, X.astype(np.float16))
            tmp_paths.append((p, len(X)))
            total += len(X)
            del X
        else:
            all_X.append(X)
        all_y.append(y)
        all_r.append(R)
        all_sym.append(sym_ids)

        log.info("Symbol done", symbol=sym, sequences=len(sym_ids),
                 class_dist_h0=str(np.bincount(y[:, 0])))

    y_all   = np.concatenate(all_y,   axis=0)
    r_all   = np.concatenate(all_r,   axis=0)
    sym_all = np.concatenate(all_sym, axis=0)

    if use_mmap:
        # Assemble one on-disk float16 memmap from the per-symbol temp files.
        x_path = os.path.join(mmap_dir, "X.npy")
        X_all = np.lib.format.open_memmap(
            x_path, mode="w+", dtype=np.float16, shape=(total, SEQ_LEN, feat_dim))
        off = 0
        for p, n in tmp_paths:
            chunk = np.load(p, mmap_mode="r")
            X_all[off:off + n] = chunk
            off += n
            del chunk
            try:
                os.remove(p)
            except OSError:
                pass
        X_all.flush()
        log.info("Full dataset assembled (float16 memmap)", total_sequences=total,
                 approx_gb=round(total * SEQ_LEN * feat_dim * 2 / 1e9, 2), path=x_path)
    else:
        X_all = np.concatenate(all_X, axis=0)
        log.info("Full dataset assembled", total_sequences=len(X_all))
    return X_all, y_all, r_all, sym_all


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
    """Helper: compute a single-horizon loss honouring per-sample weighting.

    Phase 16: when ``loss_fn`` was built with ``reduction='none'`` it returns
    a ``(B,)`` per-sample loss; we multiply by ``sample_weights`` and take the
    mean. Falls back gracefully when ``sample_weights`` is None (legacy)."""
    raw = loss_fn(logits, targets)              # (B,) if reduction='none', else scalar
    if raw.dim() == 0:
        return raw                              # already reduced
    if sample_weights is None:
        return raw.mean()
    return (raw * sample_weights).mean()


def train_epoch(model, loader, optimizer, loss_fns, device):
    model.train()
    total_loss = 0.0
    n_batches  = 0

    for batch in loader:
        # Phase 16: optional 4th element = future_returns (B, H) for PnL weighting.
        if len(batch) == 4:
            bx, by, br, bs = batch
            br = br.to(device)
        else:
            bx, by, bs = batch
            br = None
        bx = bx.to(device); by = by.to(device); bs = bs.to(device)

        optimizer.zero_grad()
        out = model(bx, bs)
        logits_list = out[0]   # tolerant to 4-tuple (offline) and 5-tuple (live w/ exits)

        loss = 0.0
        for h_idx, (logits, loss_fn) in enumerate(zip(logits_list, loss_fns)):
            loss_fn_d = loss_fn.to(device)
            if br is not None:
                # Per-sample PnL-magnitude weights for this horizon
                abs_r = br[:, h_idx].abs()
                med = abs_r[abs_r > 0].median() if (abs_r > 0).any() else torch.tensor(1e-4, device=device)
                w = torch.clamp(abs_r / (med + 1e-8), 0.25, 4.0)
            else:
                w = None
            loss = loss + _apply_horizon_loss(loss_fn_d, logits, by[:, h_idx], w)
        loss = loss / len(HORIZONS)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches  += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def eval_epoch(model, loader, loss_fns, device):
    model.eval()
    total_loss = 0.0
    n_batches  = 0
    all_preds  = [[] for _ in HORIZONS]
    all_labels = [[] for _ in HORIZONS]
    all_probs  = [[] for _ in HORIZONS]

    for batch in loader:
        if len(batch) == 4:
            bx, by, br, bs = batch
            br = br.to(device)
        else:
            bx, by, bs = batch
            br = None
        bx = bx.to(device); by = by.to(device); bs = bs.to(device)

        out = model(bx, bs)
        logits_list, probs_list = out[0], out[1]

        loss = 0.0
        for h_idx, (logits, probs, loss_fn) in enumerate(zip(logits_list, probs_list, loss_fns)):
            if br is not None:
                abs_r = br[:, h_idx].abs()
                med = abs_r[abs_r > 0].median() if (abs_r > 0).any() else torch.tensor(1e-4, device=device)
                w = torch.clamp(abs_r / (med + 1e-8), 0.25, 4.0)
            else:
                w = None
            loss = loss + _apply_horizon_loss(loss_fn.to(device), logits, by[:, h_idx], w)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds[h_idx].extend(preds)
            all_labels[h_idx].extend(by[:, h_idx].cpu().numpy())
            all_probs[h_idx].extend(probs.cpu().numpy())

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
    }, path)
    log.info("Checkpoint saved", path=str(path), epoch=epoch, val_loss=f"{val_loss:.4f}")


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
    args = parser.parse_args()

    # ── reproducibility ──────────────────────────────────────────────────────
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

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
    X, y, R, sym_ids = build_dataset(
        symbols, args.start_year, args.start_month, args.skip_download,
        mmap_dir=(args.mmap_dir if args.mmap else None),
    )

    # ── chronological train/val split (last 20% = val) ───────────────────────
    # Sort by time is implicit since we concatenated per-symbol chronologically
    # For strict correctness: split each symbol individually then recombine
    split = int(len(X) * 0.8)
    X_tr, X_va     = X[:split],       X[split:]
    y_tr, y_va     = y[:split],       y[split:]
    R_tr, R_va     = R[:split],       R[split:]
    sym_tr, sym_va = sym_ids[:split], sym_ids[split:]

    log.info("Split sizes", train=len(X_tr), val=len(X_va))
    for hi, h in enumerate(HORIZONS):
        log.info(f"H+{h} class dist — train: {np.bincount(y_tr[:, hi])}  "
                 f"val: {np.bincount(y_va[:, hi])}")

    # ── data loaders (Phase 16 4-tuple: X, y, future_returns, sym_ids) ───────
    def make_loader(X_, y_, R_, s_, shuffle):
        if X_.dtype == np.float16:
            # Memmap/streaming path: read + cast to float32 per item; a couple of
            # workers hide the disk latency. (Linux/Colab — safe with workers.)
            ds = _SeqDataset(X_, y_, R_, s_)
            nw = 2
        else:
            ds = TensorDataset(
                torch.from_numpy(X_),
                torch.from_numpy(y_),
                torch.from_numpy(R_),
                torch.from_numpy(s_),
            )
            nw = 0
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle,
                          pin_memory=True, num_workers=nw)

    train_loader = make_loader(X_tr, y_tr, R_tr, sym_tr, shuffle=True)
    val_loader   = make_loader(X_va, y_va, R_va, sym_va, shuffle=False)

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
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=3, factor=0.5, verbose=True
    )

    # ── paths ────────────────────────────────────────────────────────────────
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    best_path   = MODELS_DIR / "pretrain_v2_best.pt"
    latest_path = MODELS_DIR / "trading_lstm_latest.pt"

    # ── training loop ────────────────────────────────────────────────────────
    best_val_loss    = float("inf")
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        t0         = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, loss_fns, device)
        val_loss, preds_list, labels_list, probs_list = eval_epoch(
            model, val_loader, loss_fns, device
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
        model, val_loader, loss_fns, device
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
    url = "https://api.binance.com/api/v3/klines"
    
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