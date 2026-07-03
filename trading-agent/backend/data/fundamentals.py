"""Point-in-time fundamentals from SEC EDGAR — the data layer for the value & quality factors.

This is the LAST untested alpha lever: price-only daily factors (momentum/reversal/low-vol) showed
no positive regression alpha on the liquid S&P 500 (their Sharpe was leaked market beta). Value and
quality are the factors the literature says carry genuine, low-momentum-correlation alpha — but they
live in fundamentals, and the No.1 way fundamental research produces FAKE alpha is look-ahead bias:
using a figure on a date before it was actually published.

EDGAR's XBRL ``companyfacts`` API solves this cleanly: every reported figure is stamped with the
``filed`` date it became public. We only ever use a fact ON OR AFTER its filing date — so the
point-in-time panel is strictly causal by construction. Free, no API key (SEC just requires a
descriptive User-Agent). XBRL coverage starts ~2009, so usable history is ~2009→present.

    cik = ticker_to_cik()["AAPL"]
    facts = fetch_company_facts("AAPL")              # cached to disk
    panel = point_in_time_panel(tickers, daily_index)  # {concept: (n_days, S) as-of arrays}

Not for live trading decisions on its own — it's the research data layer for the factor study.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

try:
    from structlog import get_logger
    log = get_logger("fundamentals")
except Exception:  # pragma: no cover
    import logging
    log = logging.getLogger("fundamentals")

CACHE_DIR = Path(__file__).resolve().parents[2] / "training_data" / "fundamentals"

# SEC requires a descriptive User-Agent identifying the requester (fair-access policy, ~10 req/s).
_UA = {"User-Agent": "WoW-research eivan.kolchin@gmail.com",
       "Accept-Encoding": "gzip, deflate", "Host": "data.sec.gov"}
_UA_WWW = {"User-Agent": "WoW-research eivan.kolchin@gmail.com"}

# XBRL concept aliases — the same economic line item is tagged differently across companies/eras, so
# each fundamental maps to an ORDERED list of candidate us-gaap tags (first one present wins, with a
# computed fallback for gross profit). ``flow`` = income/period amount (has start+end → annualised);
# ``stock`` = balance-sheet instant (end only → latest as-of value).
NET_INCOME = ["NetIncomeLoss", "NetIncomeLossAvailableToCommonStockholdersBasic", "ProfitLoss"]
EQUITY = ["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"]
ASSETS = ["Assets"]
GROSS_PROFIT = ["GrossProfit"]
REVENUE = ["RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues", "SalesRevenueNet",
           "RevenueFromContractWithCustomerIncludingAssessedTax"]
COST_OF_REV = ["CostOfRevenue", "CostOfGoodsAndServicesSold", "CostOfGoodsSold"]
SHARES = ["CommonStockSharesOutstanding", "CommonStockSharesIssued"]


def _get(url: str, headers: dict, timeout: int = 30):
    import requests
    return requests.get(url, headers=headers, timeout=timeout)


def ticker_to_cik() -> Dict[str, str]:
    """{TICKER -> 10-digit zero-padded CIK} from SEC's master list (cached to disk for the day)."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / "ticker_cik.json"
    if cache.exists():
        try:
            return json.loads(cache.read_text())
        except Exception:
            pass
    m = _get("https://www.sec.gov/files/company_tickers.json", _UA_WWW, 20).json()
    out = {v["ticker"].upper(): str(v["cik_str"]).zfill(10) for v in m.values()}
    cache.write_text(json.dumps(out))
    return out


def fetch_company_facts(ticker: str, cik_map: Optional[Dict[str, str]] = None,
                        throttle: float = 0.12) -> Optional[dict]:
    """Raw companyfacts JSON for one ticker (cached to disk). ``throttle`` spaces network calls to
    respect SEC's ~10 req/s fair-access limit. Returns None if the ticker has no CIK / no facts."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / f"facts_{ticker.upper()}.json"
    if cache.exists():
        try:
            return json.loads(cache.read_text())
        except Exception:
            pass
    cik_map = cik_map or ticker_to_cik()
    cik = cik_map.get(ticker.upper())
    if not cik:
        return None
    try:
        time.sleep(throttle)
        r = _get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json", _UA, 30)
        if r.status_code != 200:
            log.warning("companyfacts_http", ticker=ticker, status=r.status_code)
            return None
        facts = r.json()
        cache.write_text(json.dumps(facts))
        return facts
    except Exception as e:  # pragma: no cover - network
        log.warning("companyfacts_failed", ticker=ticker, error=str(e)[:100])
        return None


def _concept_facts(facts: dict, aliases: Sequence[str]) -> pd.DataFrame:
    """Tidy frame [end, val, filed, days] for the first present alias. ``days`` = period length
    (NaN for balance-sheet instants). Empty frame if no alias is present."""
    gaap = (facts or {}).get("facts", {}).get("us-gaap", {})
    for tag in aliases:
        if tag not in gaap:
            continue
        units = gaap[tag].get("units", {})
        unit_key = next((u for u in units if u in ("USD", "shares")), None)
        if unit_key is None:
            continue
        rows = []
        for p in units[unit_key]:
            end, val, filed = p.get("end"), p.get("val"), p.get("filed")
            if end is None or val is None or filed is None:
                continue
            start = p.get("start")
            days = (pd.Timestamp(end) - pd.Timestamp(start)).days if start else np.nan
            rows.append((end, float(val), filed, days))
        if rows:
            df = pd.DataFrame(rows, columns=["end", "val", "filed", "days"])
            df["end"] = pd.to_datetime(df["end"])
            df["filed"] = pd.to_datetime(df["filed"])
            return df.sort_values(["filed", "end"]).reset_index(drop=True)
    return pd.DataFrame(columns=["end", "val", "filed", "days"])


def _annualise(df: pd.DataFrame) -> pd.DataFrame:
    """Keep ANNUAL observations of a flow concept (period length ≈ 1 year). Robust across companies
    that file YTD vs 3-month quarterly amounts: we only trust the unambiguous ~365-day figure."""
    if df.empty:
        return df
    return df[(df["days"] >= 350) & (df["days"] <= 380)].reset_index(drop=True)


def point_in_time_series(df: pd.DataFrame, asof: pd.DatetimeIndex) -> np.ndarray:
    """Strictly-causal as-of values on the ``asof`` calendar. At each date d the value is the
    FRESHEST fact that was already PUBLISHED (filed ≤ d) — i.e. among facts with filed ≤ d, the one
    with the latest period ``end`` (ties → latest filed). Returns NaN before the first filing.

    This is the look-ahead firewall: nothing reported in the future can influence date d, and a
    later restatement of a period only takes effect once ITS filing date has passed."""
    out = np.full(len(asof), np.nan)
    if df.empty:
        return out
    # Build (filed_date -> best-known value), where "best" = fact with the latest period end seen so
    # far (a fresh new period or a restatement of the current latest period updates it; a late
    # restatement of an OLDER period does not override the newer one).
    best_end = pd.Timestamp.min
    best_val = np.nan
    change_dates, change_vals = [], []
    for filed, end, val in zip(df["filed"], df["end"], df["val"]):
        if end >= best_end:
            best_end, best_val = end, val
        change_dates.append(filed)
        change_vals.append(best_val)
    cd = pd.DatetimeIndex(change_dates)
    cv = np.asarray(change_vals, dtype=np.float64)
    # for each asof date, take the value effective at the last filing on/before it
    pos = np.searchsorted(cd.values, asof.values, side="right") - 1
    valid = pos >= 0
    out[valid] = cv[pos[valid]]
    return out


# Concepts the factor study needs: flows annualised, balance-sheet items as latest-known instant.
_CONCEPTS = {
    "net_income": (NET_INCOME, "flow"),
    "equity": (EQUITY, "stock"),
    "assets": (ASSETS, "stock"),
    "gross_profit": (GROSS_PROFIT, "flow"),
    "revenue": (REVENUE, "flow"),
    "cost_of_rev": (COST_OF_REV, "flow"),
    "shares": (SHARES, "stock"),
}


def build_concept(ticker: str, concept: str, asof: pd.DatetimeIndex,
                  cik_map: Optional[Dict[str, str]] = None,
                  facts: Optional[dict] = None) -> np.ndarray:
    """Point-in-time as-of series for one ``concept`` of one ticker on the ``asof`` calendar.
    Flow concepts are annualised first; gross_profit falls back to revenue − cost_of_rev."""
    facts = facts if facts is not None else fetch_company_facts(ticker, cik_map)
    if facts is None:
        return np.full(len(asof), np.nan)
    aliases, kind = _CONCEPTS[concept]
    df = _concept_facts(facts, aliases)
    if kind == "flow":
        df = _annualise(df)
    s = point_in_time_series(df, asof)
    if concept == "gross_profit" and np.all(np.isnan(s)):
        rev = build_concept(ticker, "revenue", asof, cik_map, facts)
        cog = build_concept(ticker, "cost_of_rev", asof, cik_map, facts)
        s = rev - cog
    return s


def point_in_time_panel(tickers: List[str], asof: pd.DatetimeIndex,
                        concepts: Optional[List[str]] = None,
                        cik_map: Optional[Dict[str, str]] = None,
                        skip_download: bool = False) -> Dict[str, np.ndarray]:
    """{concept -> (len(asof), n_tickers) point-in-time array}, columns ordered like ``tickers``.
    Every value is strictly as-of (no look-ahead). ``skip_download`` uses only on-disk caches."""
    concepts = concepts or ["net_income", "equity", "assets", "gross_profit", "shares"]
    cik_map = cik_map or ticker_to_cik()
    S = len(tickers)
    panel = {c: np.full((len(asof), S), np.nan) for c in concepts}
    loaded = 0
    for j, t in enumerate(tickers):
        if skip_download and not (CACHE_DIR / f"facts_{t.upper()}.json").exists():
            continue
        facts = fetch_company_facts(t, cik_map)
        if facts is None:
            continue
        loaded += 1
        for c in concepts:
            panel[c][:, j] = build_concept(t, c, asof, cik_map, facts)
    log.info("fundamentals_panel", requested=S, loaded=loaded, concepts=len(concepts))
    return panel
