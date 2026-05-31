"""Phase 9 tests: statistical R:R boundary floors + recalc-on-breach monitor hook."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from backend.risk.manager import RiskManager
from backend.execution.position_monitor import PositionMonitor


# --- statistical R:R floor -----------------------------------------------

def test_enforce_floors_applies_vol_floor_and_min_rr():
    rm = RiskManager()
    # vol 1% with k=1.0 -> SL floor = 0.01; supplied SL=0.005 is bumped to 0.01.
    # min R:R 1.5 -> TP must be >= 0.01 * 1.5 = 0.015.
    sl, tp = rm.enforce_exit_floors(sl_frac=0.005, tp_frac=0.01, recent_vol=0.01)
    assert sl == pytest.approx(0.01)
    assert tp == pytest.approx(0.015)


def test_enforce_floors_preserves_better_values():
    rm = RiskManager()
    # low recent vol -> SL floor 0.3%; caller's SL=2% already above; TP 5% above 2%*1.5=3%.
    sl, tp = rm.enforce_exit_floors(sl_frac=0.02, tp_frac=0.05, recent_vol=0.001)
    assert sl == 0.02 and tp == 0.05


def test_enforce_floors_handles_none_inputs():
    rm = RiskManager()
    sl, tp = rm.enforce_exit_floors(sl_frac=None, tp_frac=None, recent_vol=None)
    assert sl >= 0.003 and tp >= sl * 1.5


# --- monitor recalc-on-breach -------------------------------------------

class _FakeTrade:
    def __init__(self, entry=100.0, sl=99.0, tp=102.0, trail=0.0, direction="long"):
        self.entry_price = entry
        self.stop_loss = sl
        self.take_profit = tp
        self.trailing_stop = trail
        self.highest_price_seen = entry
        class _D:  # mimic the SQLAlchemy enum interface
            value = direction
        self.direction = _D()


class _FakeEngine:
    def __init__(self, price): self._price = price; self.closed = []
    async def get_price(self, symbol): return self._price
    async def close_position(self, symbol, reason="signal"): self.closed.append((symbol, reason))


@pytest.mark.asyncio
async def test_recalc_returning_levels_skips_close():
    engine = _FakeEngine(price=98.0)        # below SL 99 -> stop_loss breach
    trade = _FakeTrade()
    open_trades = {"X": trade}

    async def recalc(symbol, t, reason):
        return {"stop_loss": 95.0}          # widen the stop so the breach no longer triggers

    monitor = PositionMonitor(open_trades, engine, poll_interval=0.01, on_breach_recalc=recalc)
    await monitor._tick()
    assert engine.closed == []
    assert trade.stop_loss == pytest.approx(95.0)


@pytest.mark.asyncio
async def test_no_recalc_closes_normally():
    engine = _FakeEngine(price=98.0)
    trade = _FakeTrade()
    monitor = PositionMonitor({"X": trade}, engine, poll_interval=0.01)
    await monitor._tick()
    assert engine.closed == [("X", "stop_loss")]


@pytest.mark.asyncio
async def test_recalc_returning_none_falls_through_to_close():
    engine = _FakeEngine(price=98.0)
    trade = _FakeTrade()

    async def recalc(symbol, t, reason): return None  # no adjustment -> close

    monitor = PositionMonitor({"X": trade}, engine, poll_interval=0.01, on_breach_recalc=recalc)
    await monitor._tick()
    assert engine.closed == [("X", "stop_loss")]


@pytest.mark.asyncio
async def test_recalc_supports_sync_callback():
    engine = _FakeEngine(price=98.0)
    trade = _FakeTrade()
    def recalc_sync(symbol, t, reason): return {"stop_loss": 90.0, "take_profit": 110.0}
    monitor = PositionMonitor({"X": trade}, engine, poll_interval=0.01, on_breach_recalc=recalc_sync)
    await monitor._tick()
    assert engine.closed == []
    assert trade.stop_loss == 90.0 and trade.take_profit == 110.0
