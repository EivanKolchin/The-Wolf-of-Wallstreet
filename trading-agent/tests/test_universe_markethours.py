"""Phase 8 tests: tradeable universe + ETP routing map + market-hours sessions."""
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from backend.core import universe as U
from backend.core import market_hours as MH


def test_universe_membership():
    assert "AMD" in U.STOCK_UNDERLYINGS and "BTCUSDT" in U.CRYPTO_SYMBOLS
    assert U.asset_class_of("AMD") == "us_stock"
    assert U.asset_class_of("BTCUSDT") == "crypto"
    assert U.asset_class_of("ZZZ") == "unknown"
    for s in ("SNDK", "AMD", "MU", "AXTI", "BE"):
        assert s in U.STOCK_UNDERLYINGS


def test_etp_routing_rules():
    assert U.has_etp("AMD") is True and U.has_etp("SNDK") is True and U.has_etp("MU") is True
    assert U.has_etp("AXTI") is False and U.has_etp("BE") is False  # trade underlying
    assert U.etp_for("AMD", "long") == "3LAM"
    assert U.etp_for("AXTI", "long") is None
    assert U.us_exchange("BE") == "NYSE" and U.us_exchange("AMD") == "NASDAQ"


def test_universe_as_dict():
    d = U.as_dict()
    assert {"crypto", "stocks", "etp_map", "us_exchange"} <= set(d.keys())
    assert "AMD" in d["stocks"] and "BTCUSDT" in d["crypto"]


def test_us_session_states():
    wed = datetime(2026, 5, 27, 15, 0, tzinfo=timezone.utc)
    assert wed.weekday() < 5  # guard: ensure a weekday
    assert MH.us_session_state(wed) == "regular"
    assert MH.us_session_state(wed.replace(hour=10)) == "pre"
    assert MH.us_session_state(wed.replace(hour=22)) == "after"
    assert MH.us_session_state(wed.replace(hour=3)) == "overnight"
    sat = datetime(2026, 5, 30, 15, 0, tzinfo=timezone.utc)
    assert sat.weekday() >= 5  # guard: ensure a weekend
    assert MH.us_session_state(sat) == "closed"


def test_lse_and_should_monitor():
    wed = datetime(2026, 5, 27, 12, 0, tzinfo=timezone.utc)
    assert MH.lse_open(wed) is True
    assert MH.lse_open(wed.replace(hour=18)) is False
    assert MH.should_monitor("crypto", wed) is True          # crypto 24/7
    assert MH.should_monitor("us_stock", wed) is True        # LSE open / US pre
    sat = datetime(2026, 5, 30, 12, 0, tzinfo=timezone.utc)
    assert MH.should_monitor("us_stock", sat) is False       # weekend
