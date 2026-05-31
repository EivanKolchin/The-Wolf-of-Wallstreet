"""DNS-resilience pre-flight + runtime fallback.

Why this exists
---------------
Some user machines have broken or restrictive DNS — corporate firewalls, dead
configured resolvers, VPN DNS leaks, IPv6 misrouting — that prevent the agent
from reaching providers like Alpaca even though Binance (already cached or via
a different code path) may still work. The result is empty stock charts and
opaque "DNS could not be contacted" errors deep in async stack traces.

What this module does
---------------------
1. At process startup, probe DNS resolution for the providers we depend on.
2. If any fail, monkey-patch ``socket.getaddrinfo`` to fall back to DNS-over-
   HTTPS via Cloudflare (1.1.1.1) for the remaining lifetime of the process.
   This is "temporary while the program runs" — it does NOT modify Windows
   DNS settings, doesn't need admin rights, and reverts when the process exits.
3. DoH uses plain HTTPS to https://1.1.1.1/dns-query so it works through any
   firewall that already allows HTTPS out (which all our other API calls need).

Critical hosts list is conservative — adding a non-essential host to it just
means a slower pre-flight; not adding one means we won't auto-recover for it.
"""
from __future__ import annotations

import json
import logging
import socket
import ssl
import threading
import urllib.parse
import urllib.request
from typing import Iterable

logger = logging.getLogger(__name__)

# Hosts we genuinely need to resolve. If any of these fail via the system
# resolver, we install the DoH fallback.
CRITICAL_HOSTS = (
    # Crypto
    "api.binance.com",
    "api.binance.us",
    "stream.binance.com",
    "fapi.binance.com",
    # Stocks (Alpaca free IEX feed)
    "data.alpaca.markets",
    "stream.data.alpaca.markets",
    "paper-api.alpaca.markets",
    # Macro / news feeds
    "api.coingecko.com",
    "api.alternative.me",
    # LLM
    "api.anthropic.com",
    "generativelanguage.googleapis.com",
    # Cloudflare (used by DoH itself — if THIS fails we can't recover)
    "1.1.1.1",
)

DOH_ENDPOINT = "https://1.1.1.1/dns-query"
DOH_TIMEOUT_S = 4.0

# Module-level cache so DoH lookups stay cheap inside hot paths.
_doh_cache: dict[str, tuple[float, list[str]]] = {}
_doh_cache_ttl = 300.0   # seconds
_cache_lock = threading.Lock()

# Keep a reference to the original resolver so we can fall back through it.
_original_getaddrinfo = socket.getaddrinfo
_patched = False


def _system_can_resolve(host: str, timeout_s: float = 2.5) -> bool:
    """Try a DNS lookup with a hard timeout. True iff it succeeded.

    Uses ``socket.getaddrinfo`` (NOT ``gethostbyname``) because that's the call
    aiohttp's resolver actually uses internally — they take different OS code
    paths and a machine can answer one and fail the other (IPv6 vs IPv4, glibc
    nsswitch ordering, etc.). Matching aiohttp's behaviour here means a green
    probe accurately predicts "aiohttp will be able to connect".
    """
    result: list[bool] = [False]

    def _try():
        try:
            socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
            result[0] = True
        except Exception:
            result[0] = False

    t = threading.Thread(target=_try, daemon=True)
    t.start()
    t.join(timeout=timeout_s)
    return result[0]


def check_dns_health(hosts: Iterable[str] = CRITICAL_HOSTS) -> dict[str, bool]:
    """Return ``{host: ok}`` for each critical host. Quick — runs in parallel."""
    hosts = list(hosts)
    statuses: dict[str, bool] = {}
    threads: list[tuple[str, threading.Thread, list[bool]]] = []
    for h in hosts:
        r = [False]
        t = threading.Thread(
            target=lambda host=h, slot=r: slot.__setitem__(0, _system_can_resolve(host, timeout_s=2.5)),
            daemon=True,
        )
        t.start()
        threads.append((h, t, r))
    for h, t, r in threads:
        t.join(timeout=4.0)
        statuses[h] = bool(r[0])
    return statuses


# --------------------------------------------------------------- DoH fallback
def _doh_lookup(host: str, record_type: str = "A") -> list[str]:
    """Resolve `host` via Cloudflare DoH. Returns a list of IPv4/IPv6 strings.
    Cached in-process for `_doh_cache_ttl` seconds. Empty list on failure.
    """
    import time
    key = f"{host}|{record_type}"
    now = time.time()
    with _cache_lock:
        cached = _doh_cache.get(key)
        if cached and (now - cached[0]) < _doh_cache_ttl:
            return list(cached[1])

    params = urllib.parse.urlencode({"name": host, "type": record_type})
    url = f"{DOH_ENDPOINT}?{params}"
    req = urllib.request.Request(url, headers={"accept": "application/dns-json"})
    ctx = ssl.create_default_context()
    try:
        with urllib.request.urlopen(req, timeout=DOH_TIMEOUT_S, context=ctx) as r:
            body = json.loads(r.read().decode("utf-8"))
    except Exception as e:
        logger.warning("doh_lookup_failed host=%s err=%s", host, e)
        return []

    addrs: list[str] = []
    for ans in body.get("Answer", []) or []:
        # Type 1 = A (IPv4), Type 28 = AAAA (IPv6), Type 5 = CNAME (chase).
        if ans.get("type") in (1, 28):
            data = ans.get("data")
            if isinstance(data, str) and data:
                addrs.append(data)

    with _cache_lock:
        _doh_cache[key] = (now, list(addrs))
    return addrs


def _patched_getaddrinfo(host, port, *args, **kwargs):
    """Drop-in for socket.getaddrinfo: try the system resolver first; on
    EAI_* / OSError fall back to DoH and return a synthesized addrinfo list."""
    try:
        return _original_getaddrinfo(host, port, *args, **kwargs)
    except (socket.gaierror, OSError):
        pass
    # Numeric IPs need no DNS — just hand them straight through.
    if isinstance(host, str):
        try:
            socket.inet_aton(host)
            return _original_getaddrinfo(host, port, *args, **kwargs)
        except OSError:
            pass

    ipv4 = _doh_lookup(host, "A") if isinstance(host, str) else []
    ipv6 = _doh_lookup(host, "AAAA") if isinstance(host, str) else []
    if not ipv4 and not ipv6:
        # Let the original failure mode propagate.
        return _original_getaddrinfo(host, port, *args, **kwargs)

    # Note: socket.getaddrinfo(host, port, family=0, type=0, proto=0, flags=0).
    # After the `host, port` positional intake the remaining args are
    # (family, type, proto, flags) — index from 0.
    family_hint = args[0] if len(args) >= 1 else kwargs.get("family", 0)
    type_hint = args[1] if len(args) >= 2 else kwargs.get("type", socket.SOCK_STREAM)
    proto_hint = args[2] if len(args) >= 3 else kwargs.get("proto", 0)
    out = []
    port_int = int(port) if port is not None else 0
    for ip in ipv4:
        if family_hint in (0, socket.AF_INET):
            out.append((socket.AF_INET, type_hint or socket.SOCK_STREAM, proto_hint or 0, "", (ip, port_int)))
    for ip in ipv6:
        if family_hint in (0, socket.AF_INET6):
            out.append((socket.AF_INET6, type_hint or socket.SOCK_STREAM, proto_hint or 0, "", (ip, port_int, 0, 0)))
    if not out:
        return _original_getaddrinfo(host, port, *args, **kwargs)
    return out


def install_doh_fallback() -> bool:
    """Monkey-patch socket.getaddrinfo to fall back to DoH on failure.

    Idempotent. Returns True if patched (or already was), False if DoH itself
    can't reach Cloudflare (in which case the program won't have network at all).
    """
    global _patched
    if _patched:
        return True
    # Quick sanity check that DoH itself works before patching.
    try:
        probe = _doh_lookup("data.alpaca.markets")
        if not probe:
            logger.warning("doh_unreachable_skip_install")
            return False
    except Exception as e:
        logger.warning("doh_probe_failed err=%s", e)
        return False
    socket.getaddrinfo = _patched_getaddrinfo  # type: ignore[assignment]
    _patched = True
    return True


# ---------------------------------------------------- single-call entry point
def pre_flight(hosts: Iterable[str] = CRITICAL_HOSTS, *, verbose: bool = True) -> dict:
    """Probe DNS for critical hosts; install DoH fallback if needed.

    Returns a summary dict suitable for printing or logging::
      {
        "system_ok":   bool,            # all hosts resolved via system DNS
        "failed_hosts": [str, ...],     # hosts the system couldn't resolve
        "doh_installed": bool,          # True if monkey-patch applied
        "doh_available": bool,          # True if Cloudflare DoH is reachable
      }
    """
    statuses = check_dns_health(hosts)
    failed = [h for h, ok in statuses.items() if not ok]
    if not failed:
        if verbose:
            logger.info("dns_preflight_all_ok n=%d", len(statuses))
        return {
            "system_ok": True, "failed_hosts": [],
            "doh_installed": False, "doh_available": True,
        }

    if verbose:
        logger.warning("dns_preflight_failures hosts=%s", failed)

    ok = install_doh_fallback()
    return {
        "system_ok": False,
        "failed_hosts": failed,
        "doh_installed": ok,
        "doh_available": ok,
    }


if __name__ == "__main__":
    # Allow `python -m backend.core.network_check` as a stand-alone diagnostic.
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    summary = pre_flight()
    print(json.dumps(summary, indent=2))
