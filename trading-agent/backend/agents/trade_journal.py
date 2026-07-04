"""Trade journal — the append-only JSONL record of everything the strategy book does.

Purpose (owner's design, 2026-07-04): every rebalance, order, mark-to-market and per-symbol
outcome is written to ONE reviewable file (``training_data/trade_journal.jsonl``) so that after
any period the owner can hand the file to a strong reviewing model (Claude) for a deep
post-mortem — while a lighter automated loop (scripts/postmortem.py) periodically distils
AGGREGATE lessons into the local LLM's skill books.

Design constraints, learned the hard way in this project:
  * The journal RECORDS; it never decides. Per-loss system mutation is the online-AWR failure
    mode (fitting noise — a correct +EV book loses ~45% of individual bets). All learning from
    this file happens at the AGGREGATE level, elsewhere.
  * Append-only JSONL, one self-contained event per line, ISO+epoch timestamps — greppable,
    diff-able, and safe to truncate/rotate by hand.
  * Best-effort: journaling failures must NEVER break a rebalance (log-and-continue).

Row kinds:
  rebalance  — one per rebalance: order/skip counts, gross/net targets, equity, news/overlay notes
  order      — one per routed order: symbol, target weight, delta, decision price, mode
  mark       — one per mark-to-market: equity, book return since last mark, drawdown
  outcome    — one per HELD symbol per mark: weight and its return contribution since last mark
  note       — free-form annotations (halts, blocks, anomalies)
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional

try:
    import structlog
    logger = structlog.get_logger("trade_journal")
except Exception:  # pragma: no cover
    import logging
    logger = logging.getLogger("trade_journal")

DEFAULT_PATH = Path("training_data") / "trade_journal.jsonl"


@dataclass
class TradeJournal:
    path: Path = field(default_factory=lambda: DEFAULT_PATH)
    _lock: Lock = field(default_factory=Lock, repr=False)

    def record(self, kind: str, **fields) -> Optional[dict]:
        """Append one event row. Returns the row, or None if the write failed (never raises)."""
        now = time.time()
        row = {"ts": round(now, 3),
               "iso": datetime.fromtimestamp(now, tz=timezone.utc).isoformat(timespec="seconds"),
               "kind": str(kind), **fields}
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self._lock, self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, default=_json_safe) + "\n")
            return row
        except OSError as e:
            logger.warning("trade_journal_write_failed", error=str(e)[:100])
            return None

    # ---------------------------------------------------------------- readers
    def read(self, since_ts: Optional[float] = None,
             kinds: Optional[List[str]] = None) -> List[dict]:
        """All rows (optionally filtered by time/kind). Tolerant of a torn final line."""
        if not self.path.exists():
            return []
        out: List[dict] = []
        want = set(kinds) if kinds else None
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue                                   # torn tail from a crash — skip
            if since_ts is not None and float(row.get("ts", 0)) < since_ts:
                continue
            if want is not None and row.get("kind") not in want:
                continue
            out.append(row)
        return out

    # ------------------------------------------------- convenience recorders
    def record_rebalance(self, plan: dict, *, equity: float, prices: Dict[str, float],
                         paper: bool = True) -> None:
        """One `rebalance` row + one `order` row per routed order. `plan` is the dict
        StrategyAgent.rebalance_once returns."""
        targets = plan.get("targets") or {}
        self.record(
            "rebalance",
            n_orders=len(plan.get("orders") or []),
            n_skipped=len(plan.get("skipped") or []),
            gross=round(sum(abs(v) for v in targets.values()), 4),
            net=round(sum(targets.values()), 4),
            equity=round(float(equity), 2),
            blocked=plan.get("blocked"),
            news=plan.get("news") or {},
            overlay=plan.get("overlay") or {},
            paper=bool(paper),
        )
        for o in plan.get("orders") or []:
            sym = o.get("symbol")
            self.record(
                "order",
                symbol=sym,
                target_weight=round(float(o.get("target_weight", 0.0)), 4),
                delta=round(float(o.get("delta", 0.0)), 4),
                decision_price=prices.get(sym),
                notional=round(abs(float(o.get("delta", 0.0))) * float(equity), 2),
                paper=bool(paper),
            )

    def record_mark(self, *, equity: float, book_return: float, drawdown: float,
                    contributions: Dict[str, float], weights: Dict[str, float]) -> None:
        """One `mark` row + one `outcome` row per held symbol (its return contribution since
        the previous mark — the raw material for hit-rate/attribution in the post-mortem)."""
        self.record("mark", equity=round(float(equity), 2),
                    book_return=round(float(book_return), 6),
                    drawdown=round(float(drawdown), 4))
        for sym, contrib in contributions.items():
            w = float(weights.get(sym, 0.0))
            if abs(w) < 1e-9 and abs(contrib) < 1e-12:
                continue
            self.record("outcome", symbol=sym, weight=round(w, 4),
                        contribution=round(float(contrib), 6))


def _json_safe(o):
    try:
        return float(o)
    except (TypeError, ValueError):
        return str(o)
