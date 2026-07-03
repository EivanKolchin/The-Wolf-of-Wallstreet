"""Daily EOD equity data for a BROAD, LONG-HISTORY universe — the data we need to test the
decades-robust factors (cross-sectional momentum, reversal, low-vol, value).

Alpaca's free IEX feed can't deliver decades × hundreds of names, so this uses yfinance
(free, ~20+ years, hundreds of tickers in one call) for RESEARCH. Adjusted closes + per-ticker
parquet cache so re-runs are instant. Output shape matches what the strategies/backtester
expect: ``{ticker: DataFrame[timestamp, open, high, low, close, volume]}``.

Not for live trading (unofficial Yahoo endpoint) — purely the factor-research data layer.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

try:
    from structlog import get_logger
    log = get_logger("equity_daily")
except Exception:  # pragma: no cover
    import logging
    log = logging.getLogger("equity_daily")

CACHE_DIR = Path(__file__).resolve().parents[2] / "training_data" / "equity_daily"

# A broad, liquid, multi-sector fallback universe (used if the S&P 500 scrape fails). ~60 names
# spanning tech, semis, financials, energy, health, staples, industrials, discretionary — enough
# cross-sectional breadth to test factors without depending on a live web scrape.
FALLBACK_UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "AVGO", "AMD", "QCOM", "INTC", "MU", "TSM",
    "TXN", "AMAT", "LRCX", "ADI", "CRM", "ORCL", "ADBE", "CSCO", "IBM", "NOW",
    "JPM", "BAC", "WFC", "GS", "MS", "C", "BLK", "SCHW",
    "XOM", "CVX", "COP", "SLB", "EOG",
    "UNH", "JNJ", "PFE", "MRK", "ABBV", "LLY", "TMO",
    "PG", "KO", "PEP", "WMT", "COST", "MCD", "NKE", "HD", "LOW",
    "CAT", "DE", "BA", "GE", "HON", "UPS", "UNP", "LMT",
    "DIS", "NFLX", "TSLA", "V", "MA",
]


_UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"}


def _clean_syms(syms) -> List[str]:
    out = [str(s).strip().replace(".", "-") for s in syms]
    return [s for s in out if s and s.isascii() and "-" != s]


def sp500_tickers() -> List[str]:
    """Full S&P 500 constituents (broadest free liquid universe). Tries Wikipedia WITH a browser
    user-agent (the default UA gets 403'd), then a reliable GitHub CSV mirror, then the curated
    64-name multi-sector fallback. Returns the fallback only if both network sources fail."""
    import io
    try:
        import requests
        html = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
                            headers=_UA, timeout=20).text
        syms = _clean_syms(pd.read_html(io.StringIO(html))[0]["Symbol"])
        if len(syms) > 400:
            log.info("sp500_from_wikipedia", n=len(syms))
            return syms
    except Exception as e:
        log.warning("sp500_wikipedia_failed", error=str(e)[:120])
    try:
        import requests
        url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
        csv = requests.get(url, headers=_UA, timeout=20).text
        syms = _clean_syms(pd.read_csv(io.StringIO(csv))["Symbol"])
        if len(syms) > 400:
            log.info("sp500_from_github_csv", n=len(syms))
            return syms
    except Exception as e:
        log.warning("sp500_github_failed", error=str(e)[:120])
    log.warning("sp500_using_curated_fallback", n=len(FALLBACK_UNIVERSE))
    return list(FALLBACK_UNIVERSE)


def load_daily(tickers: List[str], start: str = "2005-01-01", end: Optional[str] = None,
               skip_download: bool = False) -> Dict[str, pd.DataFrame]:
    """Daily OHLCV per ticker (adjusted). Cached per ticker+start to parquet; ``skip_download``
    loads cache only. Tickers with no data are skipped."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out: Dict[str, pd.DataFrame] = {}
    missing: List[str] = []
    for t in tickers:
        cache = CACHE_DIR / f"{t}_{start}_1d.parquet"
        if cache.exists():
            out[t] = pd.read_parquet(cache)
        elif not skip_download:
            missing.append(t)
    if missing and not skip_download:
        try:
            import yfinance as yf
        except ImportError:
            raise ImportError("yfinance is required for the daily equity universe. "
                              "Install it with: pip install yfinance")
        raw = yf.download(missing, start=start, end=end, interval="1d", auto_adjust=True,
                          group_by="ticker", progress=False, threads=True)
        for t in missing:
            try:
                df = raw[t] if isinstance(raw.columns, pd.MultiIndex) else raw
                df = df.dropna(how="all")
                if df.empty:
                    continue
                d = pd.DataFrame({
                    "timestamp": pd.to_datetime(df.index),
                    "open": df["Open"].to_numpy(np.float64), "high": df["High"].to_numpy(np.float64),
                    "low": df["Low"].to_numpy(np.float64), "close": df["Close"].to_numpy(np.float64),
                    "volume": df["Volume"].to_numpy(np.float64),
                }).dropna(subset=["close"]).reset_index(drop=True)
                if len(d) < 30:
                    continue
                d.to_parquet(CACHE_DIR / f"{t}_{start}_1d.parquet")
                out[t] = d
            except Exception as e:
                log.warning("ticker_load_failed", ticker=t, error=str(e)[:80])
    log.info("daily_universe_loaded", requested=len(tickers), loaded=len(out))
    return out
