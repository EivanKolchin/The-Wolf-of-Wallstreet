"""Broker abstraction shared by every execution venue.

The interface mirrors the contract the crypto `DefiExecutionEngine` already
implements (`get_price` / `execute` / `close_position` / trade-closed callback),
so adding Alpaca (US stocks) and IBKR (LSE leveraged ETPs) is just new adapters —
the NN agent and position monitor stay venue-agnostic.
"""
from __future__ import annotations

import enum
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional


class AssetClass(str, enum.Enum):
    crypto = "crypto"
    us_stock = "us_stock"
    lse_etp = "lse_etp"


@dataclass
class Fee:
    amount: float = 0.0
    currency: str = "USD"
    breakdown: dict | None = None


@dataclass
class OrderRequest:
    symbol: str                 # the EXECUTION symbol (ETP ticker, stock, or crypto pair)
    direction: str              # "long" / "short"
    size_usd: float
    asset_class: str = AssetClass.crypto.value
    venue: str = ""             # "uniswap" | "nasdaq" | "nyse" | "lse"
    quote_asset: str = "USD"    # currency the position settles in (USD/GBP/USDC)
    sl: float = 0.0             # stop-loss fraction
    tp: float = 0.0             # take-profit fraction
    trail: float = 0.0          # trailing-stop fraction
    extended_hours: bool = False
    meta: dict = field(default_factory=dict)


@dataclass
class CloseResult:
    symbol: str
    exit_price: float = 0.0
    pnl_usd: float = 0.0
    pnl_pct: float = 0.0
    fee: Optional[Fee] = None
    reason: str = "signal"
    ok: bool = True


class BrokerInterface(ABC):
    """All execution venues implement this. `asset_class` tags what the broker trades."""

    asset_class: str = AssetClass.crypto.value

    @abstractmethod
    async def get_price(self, symbol: str) -> Optional[float]:
        ...

    @abstractmethod
    async def execute(self, decision: Any, portfolio_state: dict) -> Any:
        ...

    @abstractmethod
    async def close_position(self, symbol: str, reason: str = "signal") -> Any:
        ...

    def set_trade_closed_callback(self, callback) -> None:
        self._trade_closed_callback = callback

    def get_open_trades(self) -> dict:
        return {}

    def quote_asset(self, symbol: str) -> str:
        return "USD"

    def is_available(self) -> bool:
        """False when the venue's SDK/credentials/connection aren't configured."""
        return True
