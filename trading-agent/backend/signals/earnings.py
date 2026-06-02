"""Earnings-calendar features for the NN (Cycle 7) — occupies the EARNINGS block
[86:90]. Shared by the offline pretraining pipeline and the live agent so the two
ALWAYS produce identical vectors for the same (symbol, time).

Four leakage-safe features per bar:
  [0] time-to-next-earnings proximity  exp(-days_to_next / TAU) — ANTICIPATORY: the
      scheduled report date is known in advance, so this is information the market
      genuinely has before the release ("how soon does NVDA report?").
  [1] pre-earnings window flag         1.0 within PRE_DAYS before the next report.
  [2] post-earnings drift proximity    exp(-days_since_last / TAU) — recently reported.
  [3] last earnings surprise           clipped EPS surprise of the most recent
      ALREADY-REPORTED earnings (epsActual known only after release → no leakage),
      decayed by time since the release.

Crypto (and any symbol without earnings data) gets all-zeros — identical offline
and live, so the model simply learns the block is inert for those assets.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

try:
    import requests
except Exception:  # pragma: no cover
    requests = None  # type: ignore

from backend.signals import feature_spec as fs

DIM = fs.EARNINGS_DIM           # 4
TAU_DAYS = 5.0                  # exponential decay scale for the proximity features
PRE_DAYS = 2.0                  # "pre-earnings window" width (days before the report)
_HOUR_MAP = {"bmo": 9, "amc": 16, "dmh": 12}   # before-open / after-close / during-market


def _release_dt(date_str: str, hour: Optional[str]) -> pd.Timestamp:
    """Approximate release timestamp from Finnhub's date + session code."""
    d = pd.Timestamp(date_str)
    return d + pd.Timedelta(hours=_HOUR_MAP.get((hour or "").lower(), 12))


def normalize_events(raw: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Finnhub ``earningsCalendar`` rows → sorted internal events."""
    events = []
    for e in raw or []:
        if not e.get("date"):
            continue
        events.append({
            "dt": _release_dt(e["date"], e.get("hour")),
            "eps_actual": e.get("epsActual"),
            "eps_estimate": e.get("epsEstimate"),
        })
    return sorted(events, key=lambda x: x["dt"])


def earnings_feature_matrix(events: List[Dict[str, Any]], timestamps) -> np.ndarray:
    """(N, DIM) leakage-safe earnings features for an array of bar timestamps.
    Vectorized via searchsorted over the (few) earnings events."""
    ts = pd.to_datetime(np.asarray(timestamps)).values.astype("datetime64[ns]").astype(np.int64)
    out = np.zeros((len(ts), DIM), dtype=np.float32)
    ev = sorted(events, key=lambda x: x["dt"]) if events else []
    if not ev or len(ts) == 0:
        return out
    ev_ns = np.array([e["dt"].value for e in ev], dtype=np.int64)
    eps_act = np.array([np.nan if e.get("eps_actual") is None else e["eps_actual"] for e in ev], float)
    eps_est = np.array([np.nan if e.get("eps_estimate") is None else e["eps_estimate"] for e in ev], float)
    day_ns = 86400e9

    # --- anticipatory: next scheduled report strictly after each bar ---
    nxt = np.searchsorted(ev_ns, ts, side="right")
    has_next = nxt < len(ev)
    days_to = np.full(len(ts), np.inf)
    days_to[has_next] = (ev_ns[np.clip(nxt, 0, len(ev) - 1)][has_next] - ts[has_next]) / day_ns
    out[:, 0] = np.exp(-np.clip(days_to, 0, None) / TAU_DAYS)
    out[:, 1] = (days_to <= PRE_DAYS).astype(np.float32)

    # --- realized: most recent ALREADY-REPORTED earnings at/before each bar ---
    reported = ~np.isnan(eps_act)
    last_reported = np.full(len(ev), -1, dtype=np.int64)   # last reported index up to k
    cur = -1
    for k in range(len(ev)):
        if reported[k]:
            cur = k
        last_reported[k] = cur
    prev = nxt - 1                                           # last event index <= bar
    lp = np.where(prev >= 0, last_reported[np.clip(prev, 0, len(ev) - 1)], -1)
    valid = lp >= 0
    days_since = np.full(len(ts), np.inf)
    days_since[valid] = (ts[valid] - ev_ns[lp[valid]]) / day_ns
    drift = np.exp(-np.clip(days_since, 0, None) / TAU_DAYS)
    out[valid, 2] = drift[valid]
    est_v, act_v = eps_est[lp[valid]], eps_act[lp[valid]]
    with np.errstate(divide="ignore", invalid="ignore"):
        surprise = (act_v - est_v) / np.abs(est_v)
    surprise = np.where(np.isfinite(surprise), np.clip(surprise, -1.0, 1.0), 0.0)
    out[valid, 3] = (surprise * drift[valid]).astype(np.float32)
    return out


def earnings_features_at(events: List[Dict[str, Any]], ts) -> np.ndarray:
    """(DIM,) features for a single bar timestamp (live path)."""
    return earnings_feature_matrix(events, [ts])[0]


def fetch_finnhub_earnings(symbol: str, start_iso: str, end_iso: str, token: str,
                           timeout: float = 15.0) -> List[Dict[str, Any]]:
    """Earnings calendar (past actuals + future estimates) from Finnhub. Returns
    normalized, sorted events; [] on any error (→ zeros, never crashes the build)."""
    if not token or requests is None:
        return []
    try:
        r = requests.get("https://finnhub.io/api/v1/calendar/earnings",
                         params={"from": start_iso, "to": end_iso, "symbol": symbol, "token": token},
                         timeout=timeout)
        r.raise_for_status()
        return normalize_events(r.json().get("earningsCalendar", []))
    except Exception:
        return []


class EarningsProvider:
    """Per-symbol cache of earnings events (one fetch covers the whole range)."""

    def __init__(self, token: str = ""):
        self.token = token or ""
        self._cache: Dict[str, List[Dict[str, Any]]] = {}

    def events(self, symbol: str, start_iso: str, end_iso: str) -> List[Dict[str, Any]]:
        key = f"{symbol}:{start_iso}:{end_iso}"
        if key not in self._cache:
            self._cache[key] = fetch_finnhub_earnings(symbol, start_iso, end_iso, self.token)
        return self._cache[key]
