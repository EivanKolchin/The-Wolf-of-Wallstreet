"""Phase 7b — stock pretraining for the unified policy network.

Pulls bars ONLINE (no permanent cache) from Alpaca historical (primary) with a
yfinance fallback. Trains the same ImprovedTradingLSTM architecture used live,
on the UNDERLYING stocks ONLY (never on ETPs), with a RECENCY-WEIGHTED loss so
newer data dominates the gradient.

Usage:
  python scripts/pretrain_stocks.py --symbols SNDK AMD MU AXTI BE --days 60 --epochs 5
  python scripts/pretrain_stocks.py --dry-run     # synthetic data path (no network/deps)
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import structlog

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.core.config import settings
from backend.core import universe as U

logger = structlog.get_logger("pretrain_stocks")

try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False

try:
    import aiohttp
    HAS_AIOHTTP = True
except Exception:
    HAS_AIOHTTP = False

try:
    import yfinance as yf
    HAS_YF = True
except Exception:
    HAS_YF = False


# ------------------------------------------------------------- recency weights
def recency_weights(n: int, decay: float = 0.7) -> np.ndarray:
    """Per-sample weights making newer samples dominate the loss.

    weight[i] = exp(-decay * (n-1-i) / n) so the last sample ~ 1.0 and older
    samples decay exponentially. Returned weights are normalised so their MEAN
    is 1.0 (keeps the loss scale comparable to the un-weighted version).
    """
    if n <= 0:
        return np.zeros(0, dtype=np.float64)
    idx = np.arange(n, dtype=np.float64)
    w = np.exp(-float(decay) * (n - 1 - idx) / float(n))
    w *= len(w) / w.sum()
    return w


# ------------------------------------------------------------- data fetching
ALPACA_DATA = "https://data.alpaca.markets/v2"


async def fetch_alpaca_bars(symbol: str, start: datetime, end: datetime, timeframe: str = "5Min") -> pd.DataFrame:
    if not HAS_AIOHTTP:
        raise RuntimeError("aiohttp not installed")
    key = getattr(settings, "ALPACA_API_KEY", "") or ""
    sec = getattr(settings, "ALPACA_SECRET_KEY", "") or getattr(settings, "ALPACA_SECRET", "") or ""
    if not key or not sec:
        raise RuntimeError("Alpaca API keys not configured in .env")

    url = f"{ALPACA_DATA}/stocks/{symbol}/bars"
    params = {
        "start": start.isoformat().replace("+00:00", "Z"),
        "end": end.isoformat().replace("+00:00", "Z"),
        "timeframe": timeframe, "feed": "iex", "limit": "10000", "adjustment": "raw",
    }
    headers = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": sec}

    rows = []
    next_token = None
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as s:
        for _ in range(20):  # safety: paginate up to 200k bars
            if next_token:
                params["page_token"] = next_token
            async with s.get(url, headers=headers, params=params) as r:
                data = await r.json()
            rows.extend(data.get("bars") or [])
            next_token = data.get("next_page_token")
            if not next_token:
                break

    if not rows:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    df = pd.DataFrame(rows).rename(columns={"t": "timestamp", "o": "open", "h": "high",
                                            "l": "low", "c": "close", "v": "volume"})
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df[["timestamp", "open", "high", "low", "close", "volume"]].sort_values("timestamp").reset_index(drop=True)


def fetch_yfinance_bars(symbol: str, period: str = "60d", interval: str = "5m") -> pd.DataFrame:
    if not HAS_YF:
        raise RuntimeError("yfinance not installed (`pip install yfinance`)")
    df = yf.Ticker(symbol).history(period=period, interval=interval, auto_adjust=False)
    if df.empty:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    df = df.reset_index().rename(columns={
        "Datetime": "timestamp", "Date": "timestamp",
        "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume",
    })
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
    return df[["timestamp", "open", "high", "low", "close", "volume"]]


def synthetic_bars(n: int = 1500, start_price: float = 100.0, seed: int = 0) -> pd.DataFrame:
    """Deterministic synthetic OHLCV for --dry-run paths (no network / deps)."""
    rng = np.random.default_rng(seed)
    rets = rng.normal(0, 0.005, n)
    close = start_price * np.exp(np.cumsum(rets))
    high = close * (1 + np.abs(rng.normal(0, 0.002, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.002, n)))
    opn = np.r_[start_price, close[:-1]]
    vol = rng.lognormal(8, 0.5, n)
    ts = pd.date_range(end=datetime.now(timezone.utc), periods=n, freq="5min")
    return pd.DataFrame({"timestamp": ts, "open": opn, "high": high, "low": low, "close": close, "volume": vol})


async def fetch_for_symbol(symbol: str, days: int, dry_run: bool, provider: str) -> pd.DataFrame:
    if dry_run:
        return synthetic_bars(n=max(1500, days * 78))  # ~78 5-min bars per US trading day
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    if provider == "alpaca":
        try:
            return await fetch_alpaca_bars(symbol, start, end)
        except Exception as e:
            logger.warning("alpaca_fetch_failed_falling_back_to_yfinance", symbol=symbol, error=str(e))
    return fetch_yfinance_bars(symbol)


# ------------------------------------------------------------- training entrypoint
def main():
    p = argparse.ArgumentParser(description="Phase 7b stock pretraining (underlyings only).")
    p.add_argument("--symbols", nargs="+", default=U.STOCK_UNDERLYINGS,
                   help="Underlying tickers (NEVER train on ETPs).")
    p.add_argument("--days", type=int, default=60)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--dry-run", action="store_true", help="Use synthetic bars (no network).")
    p.add_argument("--provider", choices=["alpaca", "yfinance"], default="alpaca")
    args = p.parse_args()

    # Guard: train ONLY on declared underlyings; silently filter anything else.
    args.symbols = [s for s in args.symbols if s in U.STOCK_UNDERLYINGS]
    if not args.symbols:
        print("No valid underlying symbols (universe.STOCK_UNDERLYINGS).")
        return 2

    logger.info("pretrain_stocks_start", symbols=args.symbols, days=args.days,
                epochs=args.epochs, dry_run=args.dry_run, provider=args.provider)

    async def _fetch_all():
        out = {}
        for sym in args.symbols:
            df = await fetch_for_symbol(sym, days=args.days, dry_run=args.dry_run, provider=args.provider)
            logger.info("fetched", symbol=sym, rows=len(df))
            out[sym] = df
        return out

    data = asyncio.run(_fetch_all())

    if not HAS_TORCH:
        logger.warning("torch_missing_skipping_training")
        return 0

    # The full training pipeline reuses scripts/pretrain.py (feature engineering +
    # model). That requires `pandas_ta`; when it isn't installed we short-circuit
    # cleanly rather than fail mid-run.
    try:
        from scripts.pretrain import (  # type: ignore
            build_feature_matrix, detect_regime, apply_rolling_zscore, build_labels,
            build_sequences, ImprovedTradingLSTM, HORIZONS, THRESHOLDS, SEQ_LEN,
            save_checkpoint,
        )
    except Exception as e:
        logger.warning("pretrain_pipeline_unavailable",
                       error=str(e), hint="`pip install pandas_ta scikit-learn requests` to enable training.")
        return 0

    all_X, all_y, all_w = [], [], []
    for sym, df in data.items():
        if len(df) < SEQ_LEN + max(HORIZONS) + 50:
            logger.warning("symbol_too_short", symbol=sym, rows=len(df))
            continue
        feats = build_feature_matrix(df)
        feats = detect_regime(df, feats)
        feats = apply_rolling_zscore(feats)
        labels = build_labels(df, HORIZONS, THRESHOLDS)
        X, y = build_sequences(feats, labels, seq_len=SEQ_LEN)
        if X.size == 0:
            continue
        all_X.append(X)
        all_y.append(y)
        all_w.append(recency_weights(len(X), decay=0.7))

    if not all_X:
        logger.warning("no_training_sequences_after_filtering")
        return 0

    X = torch.from_numpy(np.concatenate(all_X, axis=0))
    y = torch.from_numpy(np.concatenate(all_y, axis=0))
    w = torch.from_numpy(np.concatenate(all_w, axis=0)).float()
    sids = torch.zeros(len(X), dtype=torch.long)  # stock-symbol embedding ids are wired in Phase 10

    model = ImprovedTradingLSTM()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    bs = 128
    for epoch in range(args.epochs):
        perm = torch.randperm(len(X))
        ep_loss, nb = 0.0, 0
        for i in range(0, len(X), bs):
            idx = perm[i:i + bs]
            bx, by, bw, bsids = X[idx], y[idx], w[idx], sids[idx]
            opt.zero_grad()
            out = model(bx, bsids)
            logits_list = out[0]
            # Recency-weighted cross-entropy on the primary horizon.
            ce = F.cross_entropy(logits_list[0], by[:, 0], reduction="none")
            loss = (ce * bw).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += float(loss.item())
            nb += 1
        logger.info("epoch_done", e=epoch + 1, loss=ep_loss / max(nb, 1))

    out_path = ROOT / "models" / "pretrain_stocks_latest.pt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        save_checkpoint(model, opt, args.epochs, 0.0, out_path, label="stocks")
    except Exception as e:
        torch.save({"model_state_dict": model.state_dict()}, out_path)
        logger.info("saved_fallback", path=str(out_path), note=str(e))
    logger.info("pretrain_stocks_done", path=str(out_path))
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
