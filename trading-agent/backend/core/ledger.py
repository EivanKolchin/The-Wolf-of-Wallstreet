"""Trade statements + API-call logging, written to an easily-accessible folder at
the repo root: ``trading-agent/statements/``.

- ``statements/transactions.csv``  — one row per REALIZED (closed) trade.
- ``statements/transactions.jsonl`` — same data, machine-readable.
- ``statements/api_calls.jsonl``    — outbound API calls (price fetches, swaps).

Designed to be import-light and process-safe enough for append-only logging.
"""
from __future__ import annotations

import csv
import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# repo root = .../trading-agent  (ledger.py is at backend/core/ledger.py)
ROOT = Path(__file__).resolve().parents[2]
STATEMENTS_DIR = ROOT / "statements"
TRANSACTIONS_CSV = STATEMENTS_DIR / "transactions.csv"
TRANSACTIONS_JSONL = STATEMENTS_DIR / "transactions.jsonl"
API_CALLS_JSONL = STATEMENTS_DIR / "api_calls.jsonl"

_TXN_FIELDS = [
    "closed_at", "opened_at", "trade_id", "symbol", "asset_class", "broker",
    "direction", "size_usd", "entry_price", "exit_price",
    "pnl_usd", "pnl_pct", "fee_paid", "quote_asset", "exit_reason", "paper",
    "target_price", "error_pct", "abs_error_pct",     # Phase 13 post-trade error feedback
]

_lock = threading.Lock()


def _ensure_dir() -> None:
    STATEMENTS_DIR.mkdir(parents=True, exist_ok=True)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def record_transaction(txn: dict[str, Any]) -> None:
    """Append a realized-trade statement to the CSV + JSONL ledgers.

    Phase 13: if `target_price` + `exit_price` + `direction` are present, also
    records the signed error (realized vs forecast target) and its absolute value
    so the feedback loop can read variance from the persistent statement log."""
    target = float(txn.get("target_price") or 0.0)
    exit_p = float(txn.get("exit_price") or 0.0)
    direction = str(txn.get("direction", ""))
    if target > 0 and exit_p > 0:
        deviation = (exit_p - target) / target
        signed = deviation if direction == "long" else -deviation
        txn.setdefault("error_pct", round(signed, 6))
        txn.setdefault("abs_error_pct", round(abs(deviation), 6))
    else:
        txn.setdefault("error_pct", 0.0)
        txn.setdefault("abs_error_pct", 0.0)

    row = {k: txn.get(k, "") for k in _TXN_FIELDS}
    row.setdefault("closed_at", _now_iso())
    try:
        with _lock:
            _ensure_dir()
            write_header = not TRANSACTIONS_CSV.exists()
            with TRANSACTIONS_CSV.open("a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=_TXN_FIELDS)
                if write_header:
                    w.writeheader()
                w.writerow(row)
            with TRANSACTIONS_JSONL.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    except Exception:
        # Never let ledger I/O break trading.
        pass


def log_api_call(
    provider: str,
    method: str,
    url: str,
    *,
    status: int | None = None,
    ok: bool | None = None,
    latency_ms: float | None = None,
    note: str | None = None,
) -> None:
    """Append an outbound API-call record. Only non-sensitive metadata is logged
    (never request bodies, keys, or private data)."""
    entry = {
        "ts": _now_iso(),
        "provider": provider,
        "method": method,
        "url": url,
        "status": status,
        "ok": ok,
        "latency_ms": round(latency_ms, 1) if latency_ms is not None else None,
        "note": note,
    }
    try:
        with _lock:
            _ensure_dir()
            with API_CALLS_JSONL.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
    except Exception:
        pass
