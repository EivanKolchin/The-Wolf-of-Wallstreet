"""IB Gateway / TWS auto-launcher.

Why this exists
---------------
The IBKR broker (``backend/execution/ibkr_broker.py``) connects via ib_insync to
TWS or IB Gateway on a TCP port (4002 paper-gateway by default). If the gateway
process isn't running the broker logs ``ibkr_connect_failed`` and the LSE
leveraged-ETP route is silently unavailable. This module:

  1. Detects whether something is already listening on the configured port.
  2. If not, searches the standard Windows install paths for ``ibgateway.exe``
     (and falls back to TWS executables if no Gateway install is present).
  3. Launches it in a detached subprocess and waits up to a configurable
     deadline for the port to become reachable — that's the moment the user
     has completed their manual login + the API is accepting connections.

Cannot bypass the login screen — IBKR requires interactive credentials on
every cold start. The launcher does the next best thing: get the UI on screen
so the user only has to type their password.
"""
from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

# Standard install hints. Probed in order; first match wins.
_INSTALL_HINTS = [
    Path(r"C:\Jts\ibgateway"),
    Path(r"C:\Program Files\IBKR\ibgateway-stable"),
    Path(r"C:\Program Files (x86)\IBKR\ibgateway-stable"),
    Path(os.path.expanduser(r"~\Jts\ibgateway")),
]

_TWS_HINTS = [
    Path(r"C:\Jts"),
    Path(os.path.expanduser(r"~\Jts")),
]

_EXE_NAMES = ("ibgateway.exe", "tws.exe")


# ---------------------------------------------------------------- detection
def port_open(host: str, port: int, timeout_s: float = 1.0) -> bool:
    """Return True iff a TCP connection to ``host:port`` succeeds quickly."""
    try:
        with socket.create_connection((host, port), timeout=timeout_s):
            return True
    except OSError:
        return False


def find_ib_gateway_exe() -> Path | None:
    """Search standard install paths for ibgateway.exe (or tws.exe as fallback).
    Recursive but capped at depth 3 so a typo'd path can't hang us."""
    # Prefer the Gateway over TWS — Gateway is lighter and the API is identical.
    for root in _INSTALL_HINTS:
        if not root.exists():
            continue
        # Standard install creates versioned subdirs like ``C:\Jts\ibgateway\1019``
        cand = _find_under(root, _EXE_NAMES, max_depth=3)
        if cand:
            return cand
    for root in _TWS_HINTS:
        if not root.exists():
            continue
        cand = _find_under(root, _EXE_NAMES, max_depth=3)
        if cand:
            return cand
    # Last resort — system PATH.
    for name in _EXE_NAMES:
        p = shutil.which(name)
        if p:
            return Path(p)
    return None


def _find_under(root: Path, names: tuple[str, ...], max_depth: int = 3) -> Path | None:
    targets = {n.lower() for n in names}
    stack: list[tuple[Path, int]] = [(root, 0)]
    while stack:
        d, depth = stack.pop()
        try:
            for entry in d.iterdir():
                if entry.is_file() and entry.name.lower() in targets:
                    return entry
                if entry.is_dir() and depth < max_depth:
                    stack.append((entry, depth + 1))
        except (PermissionError, OSError):
            continue
    return None


# ---------------------------------------------------------------- launch
def launch_ib_gateway(exe: Path) -> subprocess.Popen | None:
    """Launch the Gateway/TWS executable in a detached process.
    Returns the Popen handle (caller can store it for later termination)."""
    try:
        # On Windows we want it to keep running after we exit; DETACHED_PROCESS
        # + CREATE_NEW_PROCESS_GROUP achieves that.
        flags = 0
        if sys.platform == "win32":
            flags = getattr(subprocess, "DETACHED_PROCESS", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        proc = subprocess.Popen(
            [str(exe)],
            cwd=str(exe.parent),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            creationflags=flags,
            close_fds=True,
        )
        return proc
    except Exception:
        return None


def wait_for_port(host: str, port: int, *, deadline_s: float = 120.0,
                  poll_interval_s: float = 2.0) -> bool:
    """Poll until ``host:port`` accepts a TCP connection or the deadline lapses."""
    end = time.monotonic() + deadline_s
    while time.monotonic() < end:
        if port_open(host, port, timeout_s=1.5):
            return True
        time.sleep(poll_interval_s)
    return False


# ---------------------------------------------------------------- orchestration
def ensure_ib_gateway_running(host: str = "127.0.0.1", port: int = 4002, *,
                                wait_seconds: float = 0.0) -> dict:
    """High-level entry point used by the start script.

    Returns a summary::
      {
        "already_running": bool,        # was something listening before we touched it?
        "exe_found": str | None,        # path to ibgateway.exe / tws.exe (if any)
        "launched": bool,               # did we just spawn it?
        "port_open": bool,              # is the TCP port reachable now?
        "host": str, "port": int,       # echoed for clarity
        "note": str,                    # one-line summary suitable for stdout
      }

    ``wait_seconds > 0`` means: after launching, block until the port becomes
    open OR the deadline lapses. Use a positive value from a start script
    where the user is actively logging in. Use 0 from a hot path so we don't
    stall the backend.
    """
    summary: dict = {
        "already_running": False, "exe_found": None, "launched": False,
        "port_open": False, "host": host, "port": int(port), "note": "",
    }
    if port_open(host, port):
        summary.update({"already_running": True, "port_open": True,
                         "note": f"IB Gateway already accepting connections on {host}:{port}."})
        return summary

    exe = find_ib_gateway_exe()
    if exe is None:
        summary["note"] = (
            "IB Gateway not found in standard install paths. "
            "Install IB Gateway from interactivebrokers.com if you want LSE-ETP execution."
        )
        return summary
    summary["exe_found"] = str(exe)

    proc = launch_ib_gateway(exe)
    if proc is None:
        summary["note"] = f"Found IB Gateway at {exe} but failed to launch."
        return summary
    summary["launched"] = True

    if wait_seconds > 0:
        if wait_for_port(host, port, deadline_s=wait_seconds):
            summary["port_open"] = True
            summary["note"] = (
                f"IB Gateway launched and API port {host}:{port} is open."
            )
        else:
            summary["note"] = (
                f"IB Gateway launched ({exe.name}). API port {host}:{port} not yet "
                f"open after {wait_seconds:.0f}s — finish login in the Gateway window. "
                f"The backend will keep retrying."
            )
    else:
        summary["note"] = (
            f"IB Gateway launched ({exe.name}). Complete the login in its window; "
            f"the backend will pick up the connection once the API port is open."
        )
    return summary


if __name__ == "__main__":
    # Stand-alone diagnostic / launcher.
    import argparse, json
    p = argparse.ArgumentParser(description="IB Gateway detect + launch + wait.")
    p.add_argument("--host", default=os.environ.get("IBKR_HOST", "127.0.0.1"))
    p.add_argument("--port", type=int, default=int(os.environ.get("IBKR_PORT", "4002")))
    p.add_argument("--wait", type=float, default=120.0,
                    help="Seconds to wait for the port after launch (0 = don't wait).")
    args = p.parse_args()
    out = ensure_ib_gateway_running(args.host, args.port, wait_seconds=args.wait)
    print(json.dumps(out, indent=2))
