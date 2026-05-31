"""Routes an asset/venue to the right broker.

Phase 7 holds the registry + asset-class selection. The full execution-routing
policy (compute on the US underlying; route to the LSE leveraged ETP when LSE is
open and an ETP exists, else the US underlying via Alpaca; AXTI/BE always as
stock) layers on top in Phase 8 using the universe map + market-hours calendar.
"""
from __future__ import annotations

from typing import Optional

import structlog

from backend.execution.base import BrokerInterface, AssetClass

logger = structlog.get_logger(__name__)


class BrokerRegistry:
    def __init__(self):
        self._brokers: dict[str, BrokerInterface] = {}

    def register(self, key: str, broker: BrokerInterface) -> None:
        self._brokers[key] = broker
        logger.info("broker_registered", key=key, available=bool(broker and broker.is_available()))

    def get(self, key: str) -> Optional[BrokerInterface]:
        return self._brokers.get(key)

    def available(self) -> dict[str, BrokerInterface]:
        return {k: b for k, b in self._brokers.items() if b is not None and b.is_available()}

    def all(self) -> dict[str, BrokerInterface]:
        return dict(self._brokers)

    def select(self, asset_class: str, prefer: Optional[str] = None) -> Optional[BrokerInterface]:
        """Pick an available broker for an asset class.

        `prefer` is an optional broker key tried first (e.g. route an equity ETP order
        to "lse_etp" first, falling back to "us_stock" when LSE is closed / no ETP)."""
        if prefer:
            b = self._brokers.get(prefer)
            if b is not None and b.is_available():
                return b
        for key, b in self._brokers.items():
            if b is not None and b.is_available() and getattr(b, "asset_class", None) == asset_class:
                return b
        return None
