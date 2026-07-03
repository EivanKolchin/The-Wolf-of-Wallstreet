"""On-chain data: stablecoin supply (minting/burning) — the crypto "positioning" axis.

Stablecoin market-cap changes are a widely-cited *dry-powder* signal: net minting tends to
precede buying pressure, net burning to precede risk-off. Unlike the 90 price features (all
restatements of the coin's own recent price), this is genuinely different information.

BUT — the project's funding-rate episode is the cautionary precedent: funding was equally
"front-runs price" in the literature and measured |IC| only 0.009–0.016 → rejected. So this
follows the SAME measure-first protocol: pure causal feature builders here, wired into
scripts/signal_audit.py behind AUDIT_ONCHAIN=1, promoted ONLY on incremental IC/AUC, never on
narrative. Free source: DefiLlama stablecoins API (no key).

Two halves like derivatives_feed:
  * PURE builders (causal alignment + features) — numpy/pandas only, unit-tested.
  * NETWORK fetcher (DefiLlama) — best-effort; any failure logs and returns empty.

Cadence: DefiLlama total-circulating history is DAILY, so this is a slow, low-frequency
context feature (forward-filled onto the intraday bar grid), appropriate for a positioning
signal — not a microstructure one."""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

try:
    from structlog import get_logger
    log = get_logger("onchain_feed")
except Exception:  # pragma: no cover
    import logging
    log = logging.getLogger("onchain_feed")

DEFILLAMA_STABLES = "https://stablecoins.llama.fi/stablecoincharts/all"


# =============================================================================
# PURE: causal alignment + feature builders (no network — unit-tested)
# =============================================================================
def align_daily_to_bars(day_ts_ms: np.ndarray, day_vals: np.ndarray,
                        bar_ts_ms: np.ndarray) -> np.ndarray:
    """Causally forward-fill a DAILY series onto (intraday) bar timestamps: each bar gets the
    most recent daily value AT OR BEFORE it (no look-ahead). Bars before the first day → NaN."""
    day_ts_ms = np.asarray(day_ts_ms, dtype=np.int64)
    day_vals = np.asarray(day_vals, dtype=np.float64)
    bar_ts_ms = np.asarray(bar_ts_ms, dtype=np.int64)
    out = np.full(bar_ts_ms.shape[0], np.nan, dtype=np.float64)
    if day_ts_ms.size == 0:
        return out
    idx = np.searchsorted(day_ts_ms, bar_ts_ms, side="right") - 1
    valid = idx >= 0
    out[valid] = day_vals[idx[valid]]
    return out


def stablecoin_features(day_ts_ms: np.ndarray, mcap: np.ndarray, bar_ts_ms: np.ndarray,
                        *, mom_short: int = 7, mom_long: int = 30) -> np.ndarray:
    """(N, 3) causal stablecoin-supply features aligned to bars:

      [0] mint_mom_short  — fractional change in aggregate stablecoin mcap over the last
                            ``mom_short`` DAILY samples (recent minting/burning momentum)
      [1] mint_mom_long   — same over ``mom_long`` days (structural dry-powder trend)
      [2] mint_accel      — short minus long (is minting accelerating vs its trend?)

    Momentum is computed on the daily series (causal), THEN forward-filled to bars, so the
    intraday grid never sees a value before its day closed. NaN warm-up rows → 0."""
    day_ts_ms = np.asarray(day_ts_ms, dtype=np.int64)
    mcap = np.asarray(mcap, dtype=np.float64)
    n = np.asarray(bar_ts_ms).shape[0]
    out = np.zeros((n, 3), dtype=np.float32)
    if mcap.size < max(mom_short, mom_long) + 1:
        return out
    s = pd.Series(mcap)
    mom_s = s.pct_change(mom_short).to_numpy()
    mom_l = s.pct_change(mom_long).to_numpy()
    accel = mom_s - mom_l
    out[:, 0] = np.nan_to_num(align_daily_to_bars(day_ts_ms, mom_s, bar_ts_ms))
    out[:, 1] = np.nan_to_num(align_daily_to_bars(day_ts_ms, mom_l, bar_ts_ms))
    out[:, 2] = np.nan_to_num(align_daily_to_bars(day_ts_ms, accel, bar_ts_ms))
    return out


STABLECOIN_FEATURE_LABELS = ["stbl_mint_7d", "stbl_mint_30d", "stbl_mint_accel"]


# =============================================================================
# NETWORK: DefiLlama stablecoins (best-effort, no key)
# =============================================================================
def fetch_stablecoin_mcap_history() -> pd.DataFrame:
    """Aggregate stablecoin circulating mcap, daily. Returns DataFrame[timestamp(ms), mcap].
    Best-effort: empty DataFrame on any failure."""
    import requests
    try:
        r = requests.get(DEFILLAMA_STABLES, timeout=20)
        if r.status_code != 200:
            log.warning("defillama_http_error", status=r.status_code)
            return pd.DataFrame(columns=["timestamp", "mcap"])
        data = r.json()
    except Exception as e:
        log.warning("defillama_request_failed", error=str(e)[:120])
        return pd.DataFrame(columns=["timestamp", "mcap"])
    rows = []
    for d in (data or []):
        try:
            ts = int(d["date"]) * 1000                # DefiLlama dates are unix SECONDS
            tot = d.get("totalCirculatingUSD") or {}
            mcap = float(sum(float(v) for v in tot.values())) if isinstance(tot, dict) \
                else float(tot)
            rows.append((ts, mcap))
        except Exception:
            continue
    if not rows:
        return pd.DataFrame(columns=["timestamp", "mcap"])
    return pd.DataFrame(rows, columns=["timestamp", "mcap"]).drop_duplicates("timestamp") \
        .sort_values("timestamp").reset_index(drop=True)


def onchain_candidate(df5: pd.DataFrame, n: int) -> Optional[np.ndarray]:
    """(n, 3) causal stablecoin features aligned to df5's bars, or None. Network — the audit
    gates this behind AUDIT_ONCHAIN=1."""
    hist = fetch_stablecoin_mcap_history()
    if hist.empty:
        return None
    ts = df5["timestamp"].values.astype("datetime64[ms]").astype("int64")[:n]
    return stablecoin_features(hist["timestamp"].to_numpy(), hist["mcap"].to_numpy(), ts)
