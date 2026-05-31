"""Multi-currency cash ledger with FX-minimizing settlement.

Rule (per user): trade proceeds **settle in the instrument's quote currency** (sell a
US stock → USD; sell an LSE ETP → GBP) with NO implicit conversion. FX only happens
on an explicit deposit to a single-currency venue or a manual withdrawal — modelled
as explicit `convert()` entries so conversion fees are never silently incurred per trade.
"""
from __future__ import annotations

import structlog
from sqlalchemy import select

from backend.memory.database import Account, CashLedger

logger = structlog.get_logger(__name__)


class CashLedgerService:
    def __init__(self, db_session_factory):
        self.db_session_factory = db_session_factory

    async def _row(self, session, account_id, currency) -> CashLedger:
        row = (await session.execute(
            select(CashLedger).where(CashLedger.account_id == account_id, CashLedger.currency == currency)
        )).scalar_one_or_none()
        if row is None:
            row = CashLedger(account_id=str(account_id), currency=currency, balance=0.0)
            session.add(row)
        return row

    async def get_balance(self, account_id, currency) -> float:
        async with self.db_session_factory() as session:
            row = (await session.execute(
                select(CashLedger).where(CashLedger.account_id == account_id, CashLedger.currency == currency)
            )).scalar_one_or_none()
            return float(row.balance) if row else 0.0

    async def credit(self, account_id, currency, amount: float) -> None:
        """Settle proceeds in `currency` — never converts."""
        async with self.db_session_factory() as session:
            row = await self._row(session, account_id, currency)
            row.balance = float(row.balance or 0.0) + float(amount)
            await session.commit()

    async def debit(self, account_id, currency, amount: float) -> None:
        async with self.db_session_factory() as session:
            row = await self._row(session, account_id, currency)
            row.balance = float(row.balance or 0.0) - float(amount)
            await session.commit()

    async def convert(self, account_id, from_ccy, to_ccy, amount: float, rate: float, fee_pct: float = 0.0) -> float:
        """Explicit FX (deposit/withdraw only). Returns the credited amount net of fee."""
        async with self.db_session_factory() as session:
            src = await self._row(session, account_id, from_ccy)
            dst = await self._row(session, account_id, to_ccy)
            src.balance = float(src.balance or 0.0) - float(amount)
            credited = float(amount) * float(rate) * (1.0 - float(fee_pct))
            dst.balance = float(dst.balance or 0.0) + credited
            await session.commit()
            logger.info("fx_convert", account=str(account_id), frm=from_ccy, to=to_ccy, amount=amount, credited=credited)
            return credited
