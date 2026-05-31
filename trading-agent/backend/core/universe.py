"""Tradeable universe + ETP routing map (Phase 8) — single source of truth.

Signals are ALWAYS computed on the liquid US underlying. Orders route to the LSE
leveraged ETP when it exists and LSE is open, else to the US underlying via Alpaca.
AXTI/BE have no ETP and always trade as the plain stock.
"""
from __future__ import annotations

# Phase 1 reconciliation: unify the live crypto trading universe with the
# model's trained vocabulary (agents/improved_model.py SYMBOLS ids 0..7).
# Previously this was just ["BTCUSDT","ETHUSDT"] (a 2-symbol trading subset)
# while the model knew 8 — so the agent only ever considered 2 of the 8
# cryptos it can score, and the chart had no predictions for the other 6.
# Binance serves all 8 pairs, so the crypto agent can trade the full set.
CRYPTO_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "AAVEUSDT",
    "XLMUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT",
]

# US underlyings the agent trades (signal computation happens here)
STOCK_UNDERLYINGS = ["SNDK", "AMD", "MU", "AXTI", "BE"]

# native US listing venue (for the chart + extended-hours routing)
US_EXCHANGE = {
    "SNDK": "NASDAQ", "AMD": "NASDAQ", "MU": "NASDAQ", "AXTI": "NASDAQ", "BE": "NYSE",
}

# underlying -> LSE leveraged UCITS ETP routing.
#   etp_available=False  -> trade the underlying stock directly (Alpaca).
#   NOTE: long_etp/short_etp tickers are placeholders — confirm the real LSE
#   leveraged-ETP tickers before enabling live ETP routing.
ETP_MAP = {
    "SNDK": {"etp_available": True,  "long_etp": "", "short_etp": "", "venue": "lse"},
    "AMD":  {"etp_available": True,  "long_etp": "3LAM", "short_etp": "3SAM", "venue": "lse"},
    "MU":   {"etp_available": True,  "long_etp": "", "short_etp": "", "venue": "lse"},
    "AXTI": {"etp_available": False, "long_etp": "", "short_etp": "", "venue": "nasdaq"},
    "BE":   {"etp_available": False, "long_etp": "", "short_etp": "", "venue": "nyse"},
}


def all_symbols() -> list[str]:
    return CRYPTO_SYMBOLS + STOCK_UNDERLYINGS


def asset_class_of(symbol: str) -> str:
    s = (symbol or "").upper()
    if s in CRYPTO_SYMBOLS:
        return "crypto"
    if s in STOCK_UNDERLYINGS:
        return "us_stock"
    return "unknown"


def has_etp(symbol: str) -> bool:
    m = ETP_MAP.get((symbol or "").upper())
    return bool(m and m.get("etp_available"))


def etp_for(symbol: str, direction: str) -> str | None:
    m = ETP_MAP.get((symbol or "").upper())
    if not m or not m.get("etp_available"):
        return None
    return m.get("long_etp") if direction == "long" else m.get("short_etp")


def us_exchange(symbol: str) -> str | None:
    return US_EXCHANGE.get((symbol or "").upper())


def as_dict() -> dict:
    """Universe payload for the frontend (/api/universe)."""
    return {
        "crypto": list(CRYPTO_SYMBOLS),
        "stocks": list(STOCK_UNDERLYINGS),
        "us_exchange": dict(US_EXCHANGE),
        "etp_map": ETP_MAP,
    }
