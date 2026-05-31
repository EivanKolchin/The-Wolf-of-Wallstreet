"""Phase 7b tests: Alpaca live paper order submission + close, /api/brokers,
pretrain_stocks helpers. Live broker calls are mocked — verifies the wiring
and the persistence/statement path without touching the real Alpaca API."""
import importlib.util
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from backend.memory.database import Base, Trade, TradeStatus, TradeDirection, OrderType
from backend.execution.alpaca_broker import AlpacaBroker
from backend.core import ledger
import backend.core.config as cfg

# Load the script directly (scripts/ isn't a package).
_spec = importlib.util.spec_from_file_location("pretrain_stocks_mod", str(ROOT / "scripts" / "pretrain_stocks.py"))
ps = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ps)


# ------------------------------------------------------------- pretrain helpers
def test_recency_weights_newer_outweighs_older():
    w = ps.recency_weights(100)
    assert len(w) == 100
    assert w[-1] > w[0]                              # newer > older
    assert abs(w.mean() - 1.0) < 1e-6                # normalised mean ~ 1.0


def test_recency_weights_zero_length():
    assert ps.recency_weights(0).size == 0


def test_synthetic_bars_shape_and_invariants():
    df = ps.synthetic_bars(n=200)
    assert len(df) == 200
    for col in ("timestamp", "open", "high", "low", "close", "volume"):
        assert col in df.columns
    assert (df["high"] >= df["low"]).all()


@pytest.mark.asyncio
async def test_fetch_for_symbol_dry_run_uses_synthetic():
    df = await ps.fetch_for_symbol("AMD", days=10, dry_run=True, provider="alpaca")
    assert len(df) >= 780  # 10 days * 78 bars
    assert "close" in df.columns


# ------------------------------------------------------------- AlpacaBroker live
def test_alpaca_unavailable_without_keys(monkeypatch):
    monkeypatch.setattr(cfg.settings, "ALPACA_API_KEY", "")
    monkeypatch.setattr(cfg.settings, "ALPACA_SECRET_KEY", "")
    assert AlpacaBroker().is_available() is False


@pytest.mark.asyncio
async def test_alpaca_execute_persists_trade_row(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg.settings, "ALPACA_API_KEY", "AKREALKEY1234567")
    monkeypatch.setattr(cfg.settings, "ALPACA_SECRET_KEY", "secretsecretsecret12")
    eng = create_async_engine(f"sqlite+aiosqlite:///{tmp_path.as_posix()}/a.db")
    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    SL = async_sessionmaker(eng, expire_on_commit=False)
    broker = AlpacaBroker(paper=True, db_session_factory=SL)

    # Pin the session to regular hours so this persistence test is deterministic
    # regardless of when it runs (execute() now skips truly-closed sessions, e.g.
    # weekends, where Alpaca would reject the order anyway).
    monkeypatch.setattr("backend.core.market_hours.us_session_state", lambda *a, **k: "regular")

    # Avoid the network for price; mock the order POST.
    monkeypatch.setattr(AlpacaBroker, "get_price", AsyncMock(return_value=100.0))
    with patch("aiohttp.ClientSession.post") as mock_post:
        resp = AsyncMock()
        resp.text = AsyncMock(return_value='{"id":"ord1","status":"accepted"}')
        resp.status = 200
        mock_post.return_value.__aenter__.return_value = resp

        class Dec:
            symbol = "AMD"; direction = "long"; size_pct = 0.1
            nn_confidence = 0.6; nn_probs = {"long": 0.6, "short": 0.2, "hold": 0.2}
            regime = "ranging"; active_news = None
            sl, tp, trail = 0.02, 0.04, 0.02
            target_price = 102.0; expected_execution_ts = 0.0

        trade = await broker.execute(Dec(), {"available_cash": 1000.0})

    assert trade is not None
    assert trade.asset == "AMD" and trade.direction == TradeDirection.long
    assert trade.broker == "alpaca" and trade.asset_class == "us_stock"
    assert trade.target_price == pytest.approx(102.0)
    assert "AMD" in broker.get_open_trades()


@pytest.mark.asyncio
async def test_alpaca_close_writes_statement_and_fires_callback(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg.settings, "ALPACA_API_KEY", "AKREALKEY1234567")
    monkeypatch.setattr(cfg.settings, "ALPACA_SECRET_KEY", "secretsecretsecret12")
    eng = create_async_engine(f"sqlite+aiosqlite:///{tmp_path.as_posix()}/c.db")
    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    SL = async_sessionmaker(eng, expire_on_commit=False)
    broker = AlpacaBroker(paper=True, db_session_factory=SL)

    # Seed an open Trade row directly.
    async with SL() as s:
        row = Trade(
            asset="AMD", direction=TradeDirection.long,
            size_usd=1000.0, entry_price=100.0, status=TradeStatus.open,
            order_type=OrderType.market, nn_confidence=0.6, nn_direction_probs={},
            regime_at_entry="ranging", stop_loss=98.0, take_profit=104.0, fee_paid=0.0,
        )
        s.add(row); await s.commit(); await s.refresh(row)
    broker._open["AMD"] = row

    statements = []
    monkeypatch.setattr(ledger, "record_transaction", lambda txn: statements.append(txn))
    monkeypatch.setattr(AlpacaBroker, "get_price", AsyncMock(return_value=95.0))  # exit -> loss

    closed = []
    async def cb(t, pnl): closed.append((t, pnl))
    broker.set_trade_closed_callback(cb)

    with patch("aiohttp.ClientSession.delete") as mock_del:
        resp = AsyncMock(); resp.text = AsyncMock(return_value=""); resp.status = 207
        mock_del.return_value.__aenter__.return_value = resp
        out = await broker.close_position("AMD", reason="signal")

    assert out is not None
    async with SL() as s:
        row2 = (await s.execute(select(Trade).where(Trade.asset == "AMD"))).scalar_one()
    assert row2.status == TradeStatus.closed
    assert row2.exit_price == pytest.approx(95.0)
    assert row2.pnl_usd < 0
    assert len(statements) == 1 and statements[0]["broker"] == "alpaca"
    assert len(closed) == 1 and closed[0][1] < 0


# ------------------------------------------------------------- /api/brokers
@pytest.mark.asyncio
async def test_api_brokers_payload_shape():
    from backend.api.routes import get_brokers
    payload = await get_brokers()
    assert {"crypto_uniswap", "alpaca_us_stock", "ibkr_lse_etp"} <= set(payload.keys())
    for v in payload.values():
        assert {"asset_class", "available", "venue"} <= set(v.keys())
