"""Phase 7 tests: broker interface conformance, registry routing, adapter availability,
and the FX-minimizing multi-currency cash ledger."""
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

from backend.execution.base import BrokerInterface, AssetClass
from backend.execution.broker_registry import BrokerRegistry
from backend.execution.defi_engine import DefiExecutionEngine, UniswapV3Executor, DefiPortfolioTracker
from backend.execution.alpaca_broker import AlpacaBroker
from backend.execution.ibkr_broker import IBKRBroker
from backend.execution.cash_ledger import CashLedgerService
from backend.memory.database import Base
import backend.core.config as cfg

ZERO = "0x0000000000000000000000000000000000000000"


class _Fake(BrokerInterface):
    def __init__(self, ac, avail=True):
        self.asset_class = ac
        self._a = avail

    async def get_price(self, s): return 1.0
    async def execute(self, d, p): return None
    async def close_position(self, s, reason="signal"): return None
    def is_available(self): return self._a


def test_crypto_engine_conforms_to_interface():
    web3 = MagicMock(); web3.eth.contract.return_value = MagicMock()
    uni = UniswapV3Executor(web3=web3, wallet_address=ZERO, private_key="0" * 64)
    tracker = DefiPortfolioTracker(web3=web3, wallet_address=ZERO)
    engine = DefiExecutionEngine(uniswap=uni, portfolio=tracker, db_session_factory=None, paper_mode=True)
    assert isinstance(engine, BrokerInterface)
    assert engine.asset_class == AssetClass.crypto.value


def test_registry_select_and_fallback():
    reg = BrokerRegistry()
    reg.register("us_stock", _Fake("us_stock", avail=True))
    reg.register("lse_etp", _Fake("lse_etp", avail=False))
    assert reg.select("us_stock") is not None
    assert reg.select("lse_etp") is None         # registered but unavailable
    assert reg.select("crypto") is None          # not registered
    # prefer LSE ETP, but it's unavailable -> fall back to an available us_stock broker
    b = reg.select("us_stock", prefer="lse_etp")
    assert b is not None and b.asset_class == "us_stock"
    assert set(reg.available().keys()) == {"us_stock"}


def test_alpaca_availability(monkeypatch):
    monkeypatch.setattr(cfg.settings, "ALPACA_API_KEY", "")
    monkeypatch.setattr(cfg.settings, "ALPACA_SECRET_KEY", "")
    assert AlpacaBroker().is_available() is False
    monkeypatch.setattr(cfg.settings, "ALPACA_API_KEY", "AKREALKEY1234567")
    monkeypatch.setattr(cfg.settings, "ALPACA_SECRET_KEY", "secretsecretsecret12")
    b = AlpacaBroker()
    assert b.is_available() is True
    assert b.asset_class == "us_stock" and b.quote_asset("AMD") == "USD"


@pytest.mark.asyncio
async def test_alpaca_get_price_none_when_unavailable(monkeypatch):
    monkeypatch.setattr(cfg.settings, "ALPACA_API_KEY", "")
    monkeypatch.setattr(cfg.settings, "ALPACA_SECRET_KEY", "")
    assert await AlpacaBroker().get_price("AMD") is None


def test_ibkr_unavailable_without_gateway():
    b = IBKRBroker()
    assert b.is_available() is False                 # no live Gateway connection
    assert b.asset_class == AssetClass.lse_etp.value
    assert b.quote_asset("3LAM") == "GBP"


@pytest.mark.asyncio
async def test_cash_ledger_is_fx_minimizing(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path.as_posix()}/c.db")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    svc = CashLedgerService(async_sessionmaker(engine, expire_on_commit=False))
    acct = "acct1"

    # US-stock proceeds settle in USD; LSE-ETP proceeds settle in GBP — no implicit FX
    await svc.credit(acct, "USD", 1000.0)
    await svc.credit(acct, "GBP", 500.0)
    assert await svc.get_balance(acct, "USD") == 1000.0
    assert await svc.get_balance(acct, "GBP") == 500.0

    # FX happens only on an explicit convert (deposit/withdraw), with a fee
    credited = await svc.convert(acct, "USD", "GBP", 200.0, rate=0.78, fee_pct=0.0015)
    assert await svc.get_balance(acct, "USD") == 800.0
    assert credited == pytest.approx(200.0 * 0.78 * (1 - 0.0015))
    assert await svc.get_balance(acct, "GBP") == pytest.approx(500.0 + credited)
