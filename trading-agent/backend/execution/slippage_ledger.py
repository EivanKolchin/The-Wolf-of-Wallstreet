"""Realized-slippage ledger — closes the loop between the backtest cost model and reality.

Every routed order records (decision price, fill price, style, venue) to a JSONL file.
``summary()`` aggregates realized slippage in bps per venue/style so the assumed cost
(15 bps round-trip in the backtests) can be validated — or corrected — with live data.
This must exist BEFORE leverage is raised: doubling leverage doubles turnover, and an
unmeasured 10 bps of slippage at 2× turnover is silent alpha bleed.

Sign convention: slippage_bps > 0 means the fill was WORSE than the decision price
(paid more on a buy / received less on a sell)."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Optional

DEFAULT_PATH = Path("training_data") / "slippage_log.jsonl"


def slippage_bps(side: str, decision_price: float, fill_price: float) -> Optional[float]:
    """Signed slippage in bps (positive = adverse). None when prices are unusable."""
    try:
        d, f = float(decision_price), float(fill_price)
    except (TypeError, ValueError):
        return None
    if d <= 0 or f <= 0:
        return None
    sign = 1.0 if str(side).upper() in ("BUY", "B", "LONG") else -1.0
    return sign * (f - d) / d * 1e4


@dataclass
class SlippageLedger:
    path: Path = field(default_factory=lambda: DEFAULT_PATH)
    assumed_bps: float = 7.5          # per-side assumption used by the backtests (15 rt / 2)
    _lock: Lock = field(default_factory=Lock, repr=False)

    def record(self, *, symbol: str, venue: str, side: str, notional: float,
               decision_price: float, fill_price: Optional[float],
               style: str = "market", status: str = "") -> Optional[dict]:
        bps = slippage_bps(side, decision_price, fill_price) if fill_price else None
        row = {
            "ts": time.time(), "symbol": symbol, "venue": venue, "side": str(side).upper(),
            "notional": round(float(notional), 2), "decision_price": decision_price,
            "fill_price": fill_price, "style": style, "status": status,
            "slippage_bps": (round(bps, 3) if bps is not None else None),
            "assumed_bps": self.assumed_bps,
        }
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self._lock, self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
        except OSError:
            return None
        return row

    def summary(self) -> dict:
        """{(venue, style): {n, mean_bps, worst_bps, vs_assumed}} from the log so far."""
        out: dict = {}
        if not self.path.exists():
            return out
        buckets: dict = {}
        for line in self.path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("slippage_bps") is None:
                continue
            buckets.setdefault((row.get("venue"), row.get("style")), []).append(
                float(row["slippage_bps"]))
        for key, vals in buckets.items():
            n = len(vals)
            mean = sum(vals) / n
            out[f"{key[0]}/{key[1]}"] = {
                "n": n, "mean_bps": round(mean, 3), "worst_bps": round(max(vals), 3),
                "vs_assumed_bps": round(mean - self.assumed_bps, 3),
            }
        return out
