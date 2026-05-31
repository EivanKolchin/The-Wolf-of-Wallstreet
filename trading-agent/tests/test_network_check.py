"""DNS pre-flight + DoH fallback unit tests."""
import socket
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))


def test_check_dns_health_returns_dict_for_each_host():
    from backend.core import network_check as nc
    results = nc.check_dns_health(["api.binance.com", "example.com"])
    assert set(results.keys()) == {"api.binance.com", "example.com"}
    for v in results.values():
        assert isinstance(v, bool)


def test_pre_flight_skips_doh_when_system_dns_works():
    from backend.core import network_check as nc

    with patch.object(nc, "check_dns_health", return_value={"a": True, "b": True}):
        with patch.object(nc, "install_doh_fallback") as install:
            summary = nc.pre_flight(["a", "b"], verbose=False)
            install.assert_not_called()
    assert summary["system_ok"] is True
    assert summary["doh_installed"] is False
    assert summary["failed_hosts"] == []


def test_pre_flight_installs_doh_on_failure():
    from backend.core import network_check as nc

    with patch.object(nc, "check_dns_health",
                       return_value={"good.com": True, "bad.com": False}):
        with patch.object(nc, "install_doh_fallback", return_value=True) as install:
            summary = nc.pre_flight(["good.com", "bad.com"], verbose=False)
            install.assert_called_once()
    assert summary["system_ok"] is False
    assert summary["failed_hosts"] == ["bad.com"]
    assert summary["doh_installed"] is True


def test_patched_getaddrinfo_uses_doh_when_system_fails(monkeypatch):
    """The patched resolver must call DoH when the original raises gaierror,
    then synthesize an addrinfo tuple from the IPs DoH returned."""
    from backend.core import network_check as nc

    def boom(*a, **kw):
        raise socket.gaierror("simulated failure")

    monkeypatch.setattr(nc, "_original_getaddrinfo", boom)
    monkeypatch.setattr(nc, "_doh_lookup",
                         lambda host, record_type="A": ["1.2.3.4"] if record_type == "A" else [])

    out = nc._patched_getaddrinfo("bad.com", 443, 0, socket.SOCK_STREAM, 0)
    assert isinstance(out, list) and len(out) == 1
    family, socktype, proto, _canon, sockaddr = out[0]
    assert family == socket.AF_INET
    assert socktype == socket.SOCK_STREAM
    assert sockaddr == ("1.2.3.4", 443)


def test_patched_getaddrinfo_passes_through_when_system_works(monkeypatch):
    """When the system resolves fine, the patch must not call DoH."""
    from backend.core import network_check as nc

    called = {"doh": 0}
    monkeypatch.setattr(nc, "_original_getaddrinfo",
                         lambda *a, **kw: [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("9.9.9.9", a[1]))])
    monkeypatch.setattr(nc, "_doh_lookup",
                         lambda host, record_type="A": called.__setitem__("doh", called["doh"] + 1) or [])

    out = nc._patched_getaddrinfo("ok.com", 443)
    assert out[0][4] == ("9.9.9.9", 443)
    assert called["doh"] == 0, "DoH must not run when system DNS succeeds"


def test_doh_cache_avoids_repeat_network_calls(monkeypatch):
    """A second DoH lookup for the same host within the TTL must hit the cache."""
    from backend.core import network_check as nc

    nc._doh_cache.clear()
    calls = {"n": 0}

    class FakeResp:
        def __init__(self, body): self._body = body
        def read(self): return self._body
        def __enter__(self): return self
        def __exit__(self, *a): pass

    def fake_urlopen(req, timeout=None, context=None):
        calls["n"] += 1
        return FakeResp(b'{"Answer":[{"type":1,"data":"5.6.7.8"}]}')

    monkeypatch.setattr(nc.urllib.request, "urlopen", fake_urlopen)
    a = nc._doh_lookup("example.com", "A")
    b = nc._doh_lookup("example.com", "A")
    assert a == b == ["5.6.7.8"]
    assert calls["n"] == 1
