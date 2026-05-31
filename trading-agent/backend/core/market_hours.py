"""Market-session calendar (Phase 8) — dependency-light, UTC-based.

Lets the agent avoid monitoring stocks during dead hours and pick the right venue.
NOTE: holidays + DST are approximated (US offset fixed to EST, LSE to UK winter);
wire `exchange_calendars` for production-accurate sessions.
"""
from __future__ import annotations

from datetime import datetime, time, timezone


def _now(now: datetime | None) -> datetime:
    return now or datetime.now(timezone.utc)


def _is_weekday(dt: datetime) -> bool:
    return dt.weekday() < 5  # Mon–Fri


def us_session_state(now: datetime | None = None) -> str:
    """'closed' | 'pre' | 'regular' | 'after' | 'overnight' for US equities (approx, UTC).

    regular 14:30–21:00, pre 09:00–14:30, after 21:00–24:00, overnight 00:00–09:00
    (Blue Ocean 24/5). Weekends -> closed.
    """
    dt = _now(now)
    if not _is_weekday(dt):
        return "closed"
    t = dt.time()
    if time(14, 30) <= t < time(21, 0):
        return "regular"
    if time(9, 0) <= t < time(14, 30):
        return "pre"
    if time(21, 0) <= t <= time(23, 59, 59):
        return "after"
    if time(0, 0) <= t < time(9, 0):
        return "overnight"
    return "closed"


def us_tradeable(now: datetime | None = None, allow_extended: bool = True) -> bool:
    state = us_session_state(now)
    if state == "regular":
        return True
    if allow_extended and state in ("pre", "after", "overnight"):
        return True
    return False


def lse_open(now: datetime | None = None) -> bool:
    """LSE regular session ≈ 08:00–16:30 UTC (UK winter), Mon–Fri."""
    dt = _now(now)
    if not _is_weekday(dt):
        return False
    return time(8, 0) <= dt.time() <= time(16, 30)


def should_monitor(asset_class: str, now: datetime | None = None, allow_extended: bool = True) -> bool:
    """Whether to actively monitor an asset class right now."""
    if asset_class == "crypto":
        return True  # crypto trades 24/7
    if asset_class in ("us_stock", "lse_etp"):
        return us_tradeable(now, allow_extended) or lse_open(now)
    return False
