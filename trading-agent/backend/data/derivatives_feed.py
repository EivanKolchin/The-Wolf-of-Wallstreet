"""Perpetual-futures derivatives data: funding rate + open interest.

This is genuinely NEW information for a crypto model — unlike the existing 90 features
(which are almost all restatements of the symbol's own recent price), funding and OI
encode *positioning and crowding*: who is paying to hold the trade, and how leveraged
the book is. Extreme funding / OI is a well-known mean-reversion and squeeze signal.

Two halves, deliberately separated:
  • PURE functions (causal alignment + feature builders) — numpy/pandas only, fully
    unit-tested offline, importable in the Colab audit AND the live agent.
  • NETWORK fetchers (Binance USD-M futures public API) — best-effort, geo-block-aware
    (fapi.binance.com returns HTTP 451 from US/cloud IPs, like the spot API), so any
    failure logs and returns empty rather than crashing the caller.

IMPORTANT (history limits): funding-rate history is available for the FULL symbol
history (8h cadence) → usable for offline training/audit. Open-interest history is only
retained ~30 days by Binance → treat OI as a LIVE-only enhancement, not a training input.

Per the measure-first discipline (see scripts/signal_audit.py): these features are fed
to the audit to check incremental IC/AUC BEFORE any feature_spec bump or retrain.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

try:
    from structlog import get_logger
    log = get_logger("derivatives_feed")
except Exception:  # pragma: no cover - structlog always present, but keep import-light
    import logging
    log = logging.getLogger("derivatives_feed")

BINANCE_FAPI = "https://fapi.binance.com"


# =============================================================================
# PURE: causal alignment + feature builders (no network — unit-tested)
# =============================================================================
def align_to_bars(event_ts_ms: np.ndarray, event_vals: np.ndarray,
                  bar_ts_ms: np.ndarray, fill: float = 0.0) -> np.ndarray:
    """Causally forward-fill an irregular event series onto bar timestamps.

    Each bar gets the most recent event value AT OR BEFORE it (no look-ahead). Bars
    before the first event get ``fill``. ``event_ts_ms`` must be sorted ascending.
    Used for both funding (8h cadence) and OI onto the 5m bar grid.
    """
    event_ts_ms = np.asarray(event_ts_ms, dtype=np.int64)
    event_vals = np.asarray(event_vals, dtype=np.float64)
    bar_ts_ms = np.asarray(bar_ts_ms, dtype=np.int64)
    out = np.full(bar_ts_ms.shape[0], float(fill), dtype=np.float64)
    if event_ts_ms.size == 0:
        return out
    idx = np.searchsorted(event_ts_ms, bar_ts_ms, side="right") - 1  # last event ≤ bar
    valid = idx >= 0
    out[valid] = event_vals[idx[valid]]
    return out


def funding_features(event_ts_ms: np.ndarray, funding_rate: np.ndarray,
                     bar_ts_ms: np.ndarray, *, z_window: int = 2000,
                     carry_window: int = 288) -> np.ndarray:
    """(N, 4) causal funding features aligned to ``bar_ts_ms``:

      [0] level       — current 8h funding rate (forward-filled), the carry sign/size
      [1] change      — funding rate minus the previous funding print (regime shift)
      [2] zscore      — rolling z-score of the level over ``z_window`` bars
                        (|z| large ⇒ crowded positioning ⇒ mean-reversion/squeeze risk)
      [3] cum_carry   — rolling sum of the level over ``carry_window`` bars (paid carry)

    All windows are trailing (causal). Returns float32. ``funding_rate`` is the raw
    decimal rate (e.g. 0.0001 = 0.01% per 8h); the audit is rank-based and any later NN
    use re-normalises, so values are left in natural units.
    """
    event_ts_ms = np.asarray(event_ts_ms, dtype=np.int64)
    funding_rate = np.asarray(funding_rate, dtype=np.float64)
    n = np.asarray(bar_ts_ms).shape[0]
    out = np.zeros((n, 4), dtype=np.float32)

    level = align_to_bars(event_ts_ms, funding_rate, bar_ts_ms)
    out[:, 0] = level

    # change: align the per-print diff (known as of that print) onto bars
    if funding_rate.size:
        prints_change = np.zeros_like(funding_rate)
        prints_change[1:] = np.diff(funding_rate)
        out[:, 1] = align_to_bars(event_ts_ms, prints_change, bar_ts_ms)

    lvl = pd.Series(level)
    roll_mean = lvl.rolling(z_window, min_periods=max(20, z_window // 50)).mean()
    roll_std = lvl.rolling(z_window, min_periods=max(20, z_window // 50)).std()
    out[:, 2] = ((lvl - roll_mean) / (roll_std + 1e-12)).fillna(0.0).to_numpy()
    out[:, 3] = lvl.rolling(carry_window, min_periods=1).sum().fillna(0.0).to_numpy()
    return out


def open_interest_features(event_ts_ms: np.ndarray, oi: np.ndarray,
                           bar_ts_ms: np.ndarray, *, z_window: int = 2000) -> np.ndarray:
    """(N, 2) causal open-interest features (LIVE-only — OI history ≤ ~30d):

      [0] oi_change   — fractional change of OI vs the previous OI sample (build-up/unwind)
      [1] oi_zscore   — rolling z-score of OI level over ``z_window`` bars (extreme leverage)
    """
    oi = np.asarray(oi, dtype=np.float64)
    n = np.asarray(bar_ts_ms).shape[0]
    out = np.zeros((n, 2), dtype=np.float32)
    level = align_to_bars(event_ts_ms, oi, bar_ts_ms, fill=0.0)
    s = pd.Series(level)
    out[:, 0] = (s.pct_change().replace([np.inf, -np.inf], 0.0).fillna(0.0).to_numpy())
    roll_mean = s.rolling(z_window, min_periods=max(20, z_window // 50)).mean()
    roll_std = s.rolling(z_window, min_periods=max(20, z_window // 50)).std()
    out[:, 1] = ((s - roll_mean) / (roll_std + 1e-12)).fillna(0.0).to_numpy()
    return out


# =============================================================================
# NETWORK: Binance USD-M futures fetchers (best-effort, geo-block-aware)
# =============================================================================
def _get_json(url: str, params: dict, timeout: int = 15):
    import requests
    try:
        r = requests.get(url, params=params, timeout=timeout)
        if r.status_code != 200:
            log.warning("fapi_http_error", url=url, status=r.status_code, body=r.text[:160])
            return None
        return r.json()
    except Exception as e:  # geo-block (451), network, parse — all non-fatal
        log.warning("fapi_request_failed", url=url, error=str(e)[:160])
        return None


def fetch_funding_history(symbol: str, start_ms: Optional[int] = None,
                          end_ms: Optional[int] = None, max_pages: int = 50) -> pd.DataFrame:
    """Full-history funding rate (8h cadence). Returns DataFrame[timestamp(ms), funding_rate].
    Paginates forward by fundingTime. Best-effort: empty DataFrame on any failure."""
    url = f"{BINANCE_FAPI}/fapi/v1/fundingRate"
    rows: list = []
    cursor = start_ms
    for _ in range(max_pages):
        params = {"symbol": symbol.upper(), "limit": 1000}
        if cursor is not None:
            params["startTime"] = int(cursor)
        if end_ms is not None:
            params["endTime"] = int(end_ms)
        data = _get_json(url, params)
        if not data:
            break
        for d in data:
            rows.append((int(d["fundingTime"]), float(d["fundingRate"])))
        if len(data) < 1000:
            break
        cursor = int(data[-1]["fundingTime"]) + 1
    if not rows:
        return pd.DataFrame(columns=["timestamp", "funding_rate"])
    df = pd.DataFrame(rows, columns=["timestamp", "funding_rate"]).drop_duplicates("timestamp")
    return df.sort_values("timestamp").reset_index(drop=True)


def fetch_open_interest_history(symbol: str, period: str = "5m", limit: int = 500) -> pd.DataFrame:
    """Recent open-interest history (Binance retains ~30 days). Returns
    DataFrame[timestamp(ms), open_interest, open_interest_value]. Best-effort."""
    url = f"{BINANCE_FAPI}/futures/data/openInterestHist"
    data = _get_json(url, {"symbol": symbol.upper(), "period": period, "limit": int(limit)})
    if not data:
        return pd.DataFrame(columns=["timestamp", "open_interest", "open_interest_value"])
    rows = [(int(d["timestamp"]), float(d["sumOpenInterest"]), float(d["sumOpenInterestValue"]))
            for d in data]
    df = pd.DataFrame(rows, columns=["timestamp", "open_interest", "open_interest_value"])
    return df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)


def fetch_current_funding(symbol: str) -> Optional[float]:
    """Latest funding rate (premiumIndex.lastFundingRate). None on failure."""
    data = _get_json(f"{BINANCE_FAPI}/fapi/v1/premiumIndex", {"symbol": symbol.upper()})
    if isinstance(data, dict) and "lastFundingRate" in data:
        try:
            return float(data["lastFundingRate"])
        except (TypeError, ValueError):
            return None
    return None
