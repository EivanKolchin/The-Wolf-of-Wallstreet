"""IBKR live wiring (mocked ib_insync — no real Gateway calls in tests)."""
import sys
from pathlib import Path

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from backend.memory.database import Base, Trade, TradeStatus, TradeDirection, OrderType
from backend.execution.ibkr_broker import IBKRBroker, HAS_IB
from backend.core import ledger


class _FakeTicker:
    def __init__(self, last=10.0):
        self.last = last; self.close = last; self.bid = last; self.ask = last
    def marketPrice(self): return self.last


class _FakeIB:
    """Stand-in for ib_insync.IB — no network, records placed orders."""
    def __init__(self):
        self.orders = []
        self._connected = True
    def isConnected(self): return self._connected
    async def qualifyContractsAsync(self, c): return [c]
    def reqMktData(self, c, *args, **kw): return _FakeTicker()
    def placeOrder(self, c, order):
        self.orders.append((getattr(c, "symbol", "?"), order.action, order.totalQuantity))
        return None
    def disconnect(self): self._connected = False


def test_ibkr_unavailable_without_connection():
    assert IBKRBroker().is_available() is False


@pytest.mark.asyncio
async def test_ibkr_execute_places_order_and_persists_trade(tmp_path):
    if not HAS_IB:
        pytest.skip("ib_insync not installed")
    eng = create_async_engine(f"sqlite+aiosqlite:///{tmp_path.as_posix()}/i.db")
    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    SL = async_sessionmaker(eng, expire_on_commit=False)

    broker = IBKRBroker(db_session_factory=SL)
    broker.ib = _FakeIB()

    class Dec:
        symbol = "3LAM"; direction = "long"; size_pct = 0.1
        nn_confidence = 0.6; nn_probs = {"long": 0.6, "short": 0.2, "hold": 0.2}
        regime = "ranging"; active_news = None
        sl, tp, trail = 0.02, 0.04, 0.02
        target_price = 10.4; expected_execution_ts = 0.0

    trade = await broker.execute(Dec(), {"available_cash": 1000.0})
    assert trade is not None
    assert trade.asset == "3LAM"
    assert trade.broker == "ibkr" and trade.asset_class == "lse_etp" and trade.quote_asset == "GBP"
    assert "3LAM" in broker.get_open_trades()
    assert len(broker.ib.orders) == 1
    sym, side, qty = broker.ib.orders[0]
    assert side == "BUY" and qty >= 1


@pytest.mark.asyncio
async def test_ibkr_close_position_writes_statement_and_fires_callback(tmp_path, monkeypatch):
    if not HAS_IB:
        pytest.skip("ib_insync not installed")
    eng = create_async_engine(f"sqlite+aiosqlite:///{tmp_path.as_posix()}/ic.db")
    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    SL = async_sessionmaker(eng, expire_on_commit=False)

    broker = IBKRBroker(db_session_factory=SL)
    broker.ib = _FakeIB()

    async with SL() as s:
        row = Trade(
            asset="3LAM", direction=TradeDirection.long,
            size_usd=1000.0, entry_price=10.0, status=TradeStatus.open,
            order_type=OrderType.market, nn_confidence=0.6, nn_direction_probs={},
            regime_at_entry="ranging", stop_loss=9.8, take_profit=10.4, fee_paid=0.0,
        )
        s.add(row); await s.commit(); await s.refresh(row)
    broker._open["3LAM"] = row

    statements = []
    monkeypatch.setattr(ledger, "record_transaction", lambda txn: statements.append(txn))

    closed = []
    async def cb(t, pnl): closed.append((t, pnl))
    broker.set_trade_closed_callback(cb)

    out = await broker.close_position("3LAM", reason="signal")
    assert out is not None
    # opposite-side closing order placed
    assert any(side == "SELL" for _, side, _ in broker.ib.orders)
    async with SL() as s:
        row2 = (await s.execute(select(Trade).where(Trade.asset == "3LAM"))).scalar_one()
    assert row2.status == TradeStatus.closed
    assert len(statements) == 1 and statements[0]["broker"] == "ibkr" and statements[0]["quote_asset"] == "GBP"
    assert len(closed) == 1
