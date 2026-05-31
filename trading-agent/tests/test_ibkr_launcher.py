"""Tests for the IB Gateway detect/launch helper."""
import socket
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))


def test_port_open_returns_false_for_closed_port():
    from backend.core.ibkr_launcher import port_open
    # Pick a port we're virtually certain is not in use.
    assert port_open("127.0.0.1", 1, timeout_s=0.5) is False


def test_port_open_returns_true_for_listening_port():
    """Open a real socket on an ephemeral port and verify we detect it."""
    from backend.core.ibkr_launcher import port_open
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    s.listen(1)
    try:
        port = s.getsockname()[1]
        assert port_open("127.0.0.1", port, timeout_s=1.0) is True
    finally:
        s.close()


def test_ensure_short_circuits_when_already_running(monkeypatch):
    from backend.core import ibkr_launcher as ibl
    monkeypatch.setattr(ibl, "port_open", lambda *a, **kw: True)
    summary = ibl.ensure_ib_gateway_running("127.0.0.1", 4002)
    assert summary["already_running"] is True
    assert summary["port_open"] is True
    assert summary["launched"] is False


def test_ensure_reports_missing_install(monkeypatch):
    from backend.core import ibkr_launcher as ibl
    monkeypatch.setattr(ibl, "port_open", lambda *a, **kw: False)
    monkeypatch.setattr(ibl, "find_ib_gateway_exe", lambda: None)
    summary = ibl.ensure_ib_gateway_running("127.0.0.1", 4002)
    assert summary["already_running"] is False
    assert summary["launched"] is False
    assert summary["exe_found"] is None
    assert "not found" in summary["note"].lower()


def test_ensure_launches_and_waits(monkeypatch, tmp_path):
    """Simulate: port closed → exe found → launch succeeds → port opens."""
    from backend.core import ibkr_launcher as ibl

    exe = tmp_path / "ibgateway.exe"
    exe.write_bytes(b"")  # placeholder
    calls = {"port_open": 0}

    def fake_port_open(*a, **kw):
        calls["port_open"] += 1
        return calls["port_open"] > 2

    monkeypatch.setattr(ibl, "port_open", fake_port_open)
    monkeypatch.setattr(ibl, "find_ib_gateway_exe", lambda: exe)
    monkeypatch.setattr(ibl, "launch_ib_gateway", lambda e: MagicMock())

    summary = ibl.ensure_ib_gateway_running("127.0.0.1", 4002, wait_seconds=10.0)
    assert summary["exe_found"] == str(exe)
    assert summary["launched"] is True
    assert summary["port_open"] is True
    assert "open" in summary["note"].lower()


def test_ensure_launches_but_port_stays_closed(monkeypatch, tmp_path):
    from backend.core import ibkr_launcher as ibl
    exe = tmp_path / "ibgateway.exe"
    exe.write_bytes(b"")
    monkeypatch.setattr(ibl, "port_open", lambda *a, **kw: False)
    monkeypatch.setattr(ibl, "find_ib_gateway_exe", lambda: exe)
    monkeypatch.setattr(ibl, "launch_ib_gateway", lambda e: MagicMock())
    summary = ibl.ensure_ib_gateway_running("127.0.0.1", 4002, wait_seconds=1.0)
    assert summary["launched"] is True
    assert summary["port_open"] is False
    assert "not yet open" in summary["note"].lower() or "login" in summary["note"].lower()


def test_find_under_walks_files(tmp_path):
    from backend.core.ibkr_launcher import _find_under
    sub = tmp_path / "ibgateway" / "1019"
    sub.mkdir(parents=True)
    exe = sub / "ibgateway.exe"
    exe.write_bytes(b"")
    found = _find_under(tmp_path, ("ibgateway.exe",), max_depth=3)
    assert found == exe
