"""Variable Attention Engine (Phase 11) — dynamic per-asset compute cadence.

HIGH (1s) when an asset needs close watching: high volatility, non-linear
(geometric/choppy) price action, high-volume windows, or the market-open hour.
LOW (5-10 min) on flat, low-volatility, straight-line trends or the market-close
hour. Manual UI overrides (Phase 12) win over the automatic decision.
"""
from __future__ import annotations

import enum
from datetime import datetime, time, timezone
from typing import Optional

import numpy as np

from backend.core import market_hours as MH
from backend.core.config import settings


class Attention(str, enum.Enum):
    HIGH = "high"
    LOW = "low"


class AttentionController:
    def __init__(self, high_interval: float | None = None, low_interval: float | None = None):
        self.high_interval = float(high_interval if high_interval is not None
                                   else getattr(settings, "ATTENTION_HIGH_SECONDS", 1.0))
        self.low_interval = float(low_interval if low_interval is not None
                                  else getattr(settings, "ATTENTION_LOW_SECONDS", 300.0))
        self._overrides: dict[str, Attention] = {}  # symbol -> forced attention (Phase 12 UI)

    # -- manual overrides (UI) -------------------------------------------------
    def set_override(self, symbol: str, attention: Optional[str]) -> None:
        if attention is None:
            self._overrides.pop(symbol, None)
        else:
            self._overrides[symbol] = Attention(attention)

    def get_override(self, symbol: str) -> Optional[Attention]:
        return self._overrides.get(symbol)

    def replace_overrides(self, overrides: dict) -> None:
        """Atomically replace all overrides from a (Redis-sourced) dict. Values
        must be "high" / "low" / None; anything else is ignored."""
        new: dict[str, Attention] = {}
        for sym, att in (overrides or {}).items():
            if att in (Attention.HIGH.value, Attention.LOW.value, "high", "low"):
                new[sym] = Attention(att)
        self._overrides = new

    # -- signal maths ----------------------------------------------------------
    @staticmethod
    def _volatility(prices) -> float:
        p = np.asarray(prices, dtype=float)
        if len(p) < 3:
            return 0.0
        base = p[:-1]
        base = np.where(base == 0, np.nan, base)
        rets = np.diff(p) / base
        rets = rets[np.isfinite(rets)]
        return float(np.std(rets)) if rets.size else 0.0

    @staticmethod
    def _linearity_r2(prices) -> float:
        """R^2 of a linear fit. High R^2 = clean straight-line trend (LOW attention);
        low R^2 = choppy / non-linear (HIGH attention). Flat series -> treated linear."""
        y = np.asarray(prices, dtype=float)
        n = len(y)
        if n < 5:
            return 1.0
        x = np.arange(n, dtype=float)
        A = np.vstack([x, np.ones(n)]).T
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        yhat = A @ coef
        ss_res = float(np.sum((y - yhat) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        if ss_tot <= 1e-12:
            return 1.0
        return max(0.0, 1.0 - ss_res / ss_tot)

    def evaluate(self, symbol: str, asset_class: str, prices, volume_ratio: float = 1.0,
                 now: datetime | None = None) -> Attention:
        ov = self._overrides.get(symbol)
        if ov is not None:
            return ov

        now = now or datetime.now(timezone.utc)
        market_open_hour = market_close_hour = False
        if asset_class in ("us_stock", "lse_etp"):
            if MH.us_session_state(now) == "regular":
                t = now.time()
                market_open_hour = time(14, 30) <= t < time(15, 30)   # first trading hour (approx UTC)
                market_close_hour = time(20, 0) <= t < time(21, 0)    # last trading hour

        if market_close_hour:
            return Attention.LOW

        high_vol = self._volatility(prices) > float(getattr(settings, "ATTENTION_VOL_THRESHOLD", 0.004))
        nonlinear = self._linearity_r2(prices) < float(getattr(settings, "ATTENTION_R2_THRESHOLD", 0.6))
        high_volume = volume_ratio > float(getattr(settings, "ATTENTION_VOLUME_THRESHOLD", 1.8))

        if market_open_hour or high_vol or nonlinear or high_volume:
            return Attention.HIGH
        return Attention.LOW

    def interval_for(self, attention: Attention) -> float:
        return self.high_interval if attention == Attention.HIGH else self.low_interval
