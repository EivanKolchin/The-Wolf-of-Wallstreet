"""Phase 2 integration test: execute -> real Trade row -> monitor stop-loss ->
close_position -> fee-aware PnL + transaction statement + closed callback.

Runs fully in paper mode against an isolated temp SQLite DB (never touches the
dev database) with a controllable mocked price.
"""
import sys
from pathlib import Path

import pytest
from unittest.mock import MagicMock, AsyncMock

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from sqlalchemy import select
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

import backend.execution.defi_engine as de
from backend.execution.defi_engine import DefiExecutionEngine, UniswapV3Executor, DefiPortfolioTracker
from backend.execution.position_monitor import PositionMonitor
from backend.memory.database import Base, Trade, TradeStatus
from backend.core import ledger

ZERO_ADDR = "0x0000000000000000000000000000000000000000"


class Decision:
    symbol = "ETHUSDT"
    direction = "long"
    size_pct = 0.1
    nn_confidence = 0.6
    nn_probs = {"long": 0.6, "short": 0.2, "hold": 0.2}
    regime = "ranging"
    active_news = None
    sl, tp, trail = 0.02, 0.04, 0.02  # 2% stop, 4% target, 2% trail


@pytest.mark.asyncio
async def test_monitor_stop_loss_closes_and_records(tmp_path, monkeypatch):
    # Isolated temp DB
    test_engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path.as_posix()}/t.db")
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    SessionLocal = async_sessionmaker(test_engine, expire_on_commit=False)

    # Controllable price (used for both entry and exit via the shared helper)
    price = {"v": 3000.0}

    async def fake_price(symbol, retries=3):
        return price["v"]

    monkeypatch.setattr(de, "fetch_binance_price", fake_price)

    # Capture statements instead of writing to the repo-root statements/ folder
    statements = []
    monkeypatch.setattr(ledger, "record_transaction", lambda txn: statements.append(txn))

    uni = UniswapV3Executor(web3=MagicMock(), wallet_address=ZERO_ADDR, private_key="0" * 64)
    uni.swap = AsyncMock()
    tracker = DefiPortfolioTracker(web3=MagicMock(), wallet_address=ZERO_ADDR)
    engine = DefiExecutionEngine(uniswap=uni, portfolio=tracker,
                                 db_session_factory=SessionLocal, paper_mode=True)

    closed = []

    async def on_closed(trade, pnl):
        closed.append((trade, pnl))

    engine.set_trade_closed_callback(on_closed)

    # 1) Open a real Trade row (paper) at price 3000
    open_trades = {}
    trade = await engine.execute(Decision(), {"available_usdc": 1000.0})
    assert trade is not None and trade.id is not None
    assert trade.stop_loss == pytest.approx(2940.0)  # 3000 * (1 - 0.02)
    open_trades["ETHUSDT"] = trade

    # 2) Price drops below the stop -> monitor must close it
    price["v"] = 2900.0
    monitor = PositionMonitor(open_trades, engine, poll_interval=0.01)
    await monitor._tick()

    # 3) The Trade row is closed in the DB, fee-aware loss recorded
    async with SessionLocal() as s:
        row = (await s.execute(select(Trade).where(Trade.asset == "ETHUSDT"))).scalar_one()
    assert row.status == TradeStatus.closed
    assert row.exit_reason == "stop_loss"
    assert row.exit_price == pytest.approx(2900.0)
    assert row.pnl_usd < 0          # closed at a loss, net of fees
    assert row.fee_paid > 0

    # 4) A transaction statement + the closed callback both fired exactly once
    assert len(statements) == 1 and statements[0]["exit_reason"] == "stop_loss"
    assert len(closed) == 1 and closed[0][1] < 0
