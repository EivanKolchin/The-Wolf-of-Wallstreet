"""Resilient aiohttp session factory.

Why this exists
---------------
aiohttp's default async resolver fails ("Could not contact DNS servers") on a
chunk of Windows + ProactorEventLoop configurations even when ``socket.getaddrinfo``
works synchronously in the same process. The user-visible symptom is "stock
charts blank" — the Binance REST/WS path happens to use different mechanics so
it survives, while Alpaca-side aiohttp sessions silently fail.

What it does
------------
Provides ``make_resilient_session()`` returning an ``aiohttp.ClientSession`` whose
TCP connector uses ``SyncDNSResolver``: synchronous ``socket.getaddrinfo`` (run
in a thread executor) plus a Cloudflare DoH fallback when system DNS truly is
broken. ``family=AF_INET`` forces IPv4 to dodge the other common failure mode
(AAAA records resolving but IPv6 routing being broken).

This bypasses aiohttp's async resolver entirely and is the canonical workaround
for the aiohttp issue.
"""
from __future__ import annotations

import asyncio
import socket
from typing import Any, Sequence

import aiohttp
from aiohttp.abc import AbstractResolver

from backend.core.network_check import _doh_lookup, _original_getaddrinfo


class SyncDNSResolver(AbstractResolver):
    """Resolver that runs the OS's sync ``getaddrinfo`` in an executor and falls
    back to Cloudflare DoH on failure. Always returns IPv4 (AF_INET) entries —
    the IPv6 path is the other Windows-specific failure mode and IPv4-only is
    safe for every backend we talk to.
    """

    def __init__(self, *, prefer_ipv4: bool = True):
        self.prefer_ipv4 = prefer_ipv4

    async def resolve(self, host: str, port: int = 0,
                      family: int = socket.AF_INET) -> list[dict[str, Any]]:
        loop = asyncio.get_event_loop()
        fam = socket.AF_INET if self.prefer_ipv4 else family

        def _sync_lookup() -> list[Any]:
            # Use the un-patched original so we don't recurse if the monkey-patch
            # is installed; we'll do our own fallback below explicitly.
            return _original_getaddrinfo(host, port, fam, socket.SOCK_STREAM)

        try:
            infos = await loop.run_in_executor(None, _sync_lookup)
        except (socket.gaierror, OSError):
            infos = None

        if infos:
            return [{
                "hostname": host,
                "host": info[4][0],
                "port": info[4][1] if len(info[4]) > 1 else port,
                "family": info[0],
                "proto": info[2],
                "flags": 0,
            } for info in infos]

        # Sync resolver failed — try DoH (numeric IP request, no DNS dependency).
        ips = _doh_lookup(host, "A")
        if not ips:
            raise OSError(f"DNS resolution failed for {host} (system + DoH)")
        return [{
            "hostname": host, "host": ip, "port": port,
            "family": socket.AF_INET, "proto": 0, "flags": 0,
        } for ip in ips]

    async def close(self) -> None:  # required by AbstractResolver
        pass


def make_resilient_connector(*, family: int = socket.AF_INET,
                              limit: int = 50, ssl: Any = None) -> aiohttp.TCPConnector:
    """A TCPConnector wired to ``SyncDNSResolver``. Force IPv4 by default."""
    return aiohttp.TCPConnector(
        resolver=SyncDNSResolver(prefer_ipv4=(family == socket.AF_INET)),
        family=family,
        limit=limit,
        ssl=ssl,
    )


def make_resilient_session(*, family: int = socket.AF_INET,
                            timeout_total: float = 12.0,
                            headers: dict | None = None) -> aiohttp.ClientSession:
    """ClientSession with the resilient connector + sane timeout default."""
    return aiohttp.ClientSession(
        connector=make_resilient_connector(family=family),
        timeout=aiohttp.ClientTimeout(total=timeout_total),
        headers=headers or {},
    )
