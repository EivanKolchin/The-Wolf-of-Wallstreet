"""Phase 11 tests: Variable Attention Engine (high/low cadence selection)."""
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from backend.agents.attention_controller import AttentionController, Attention


def test_flat_series_is_low_attention():
    ac = AttentionController()
    assert ac.evaluate("BTCUSDT", "crypto", [100.0] * 40) == Attention.LOW


def test_straight_trend_is_low_attention():
    ac = AttentionController()
    trend = list(np.linspace(100.0, 101.0, 40))  # smooth straight line, negligible vol
    assert ac.evaluate("BTCUSDT", "crypto", trend) == Attention.LOW


def test_choppy_nonlinear_is_high_attention():
    ac = AttentionController()
    rng = np.random.default_rng(0)
    choppy = (100 + np.cumsum(rng.normal(0, 1.0, 40))).tolist()  # volatile / non-linear
    assert ac.evaluate("BTCUSDT", "crypto", choppy) == Attention.HIGH


def test_high_volume_triggers_high():
    ac = AttentionController()
    assert ac.evaluate("BTCUSDT", "crypto", [100.0] * 40, volume_ratio=3.0) == Attention.HIGH


def test_manual_override_wins():
    ac = AttentionController()
    flat = [100.0] * 40
    ac.set_override("BTCUSDT", "high")
    assert ac.evaluate("BTCUSDT", "crypto", flat) == Attention.HIGH
    ac.set_override("BTCUSDT", None)
    assert ac.evaluate("BTCUSDT", "crypto", flat) == Attention.LOW


def test_interval_mapping():
    ac = AttentionController(high_interval=1.0, low_interval=300.0)
    assert ac.interval_for(Attention.HIGH) == 1.0
    assert ac.interval_for(Attention.LOW) == 300.0


def test_market_close_hour_forces_low_for_stocks():
    ac = AttentionController()
    now = datetime(2026, 5, 27, 20, 15, tzinfo=timezone.utc)  # last trading hour
    assert now.weekday() < 5  # guard: weekday
    rng = np.random.default_rng(1)
    choppy = (100 + np.cumsum(rng.normal(0, 1.0, 40))).tolist()
    # close-hour wins even on a choppy series
    assert ac.evaluate("AMD", "us_stock", choppy, now=now) == Attention.LOW
