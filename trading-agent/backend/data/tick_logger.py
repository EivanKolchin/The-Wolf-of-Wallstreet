"""L1 quote logger — the training corpus a future RL execution agent will need.

You cannot buy your venue's OWN past top-of-book cheaply, so capture must start LONG before
the RL Smart-Order-Router is justified. This streams Binance USD-M futures ``bookTicker``
(best bid/ask, pushed on every change — no polling) to gz-rotated daily JSONL. A few MB/day
per symbol; entirely passive; OFF by default (TICK_LOGGER_ENABLED).

When the slippage ledger later shows the deterministic TWAP slicer is leaving basis points on
the table, this archive is what the PPO SOR trains on. Until then it just accumulates, cheaply.

Split like the rest of the data layer: a PURE, unit-tested writer (rotation + serialization,
no network) and a best-effort async websocket reader that feeds it."""
from __future__ import annotations

import gzip
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

try:
    import structlog
    log = structlog.get_logger("tick_logger")
except Exception:  # pragma: no cover
    import logging
    log = logging.getLogger("tick_logger")

WS_BASE = "wss://fstream.binance.com/stream?streams="
WS_TESTNET = "wss://stream.binancefuture.com/stream?streams="


@dataclass
class TickWriter:
    """Append L1 quote rows to a gz JSONL file that rotates by UTC date. Pure + testable."""
    out_dir: Path = field(default_factory=lambda: Path("training_data/ticks"))
    _day: str = ""
    _fh: Optional[gzip.GzipFile] = None
    _n: int = 0

    def _path_for(self, day: str) -> Path:
        return self.out_dir / f"bookticker_{day}.jsonl.gz"

    def _rotate_if_needed(self, ts: float) -> None:
        day = time.strftime("%Y%m%d", time.gmtime(ts))
        if day != self._day:
            self.close()
            self.out_dir.mkdir(parents=True, exist_ok=True)
            self._fh = gzip.open(self._path_for(day), "at", encoding="utf-8")
            self._day = day

    def write(self, symbol: str, bid: float, ask: float, bid_qty: float = 0.0,
              ask_qty: float = 0.0, ts: Optional[float] = None) -> None:
        ts = time.time() if ts is None else ts
        self._rotate_if_needed(ts)
        row = {"ts": round(ts, 3), "s": symbol, "b": bid, "a": ask,
               "B": bid_qty, "A": ask_qty}
        assert self._fh is not None
        self._fh.write(json.dumps(row, separators=(",", ":")) + "\n")
        self._n += 1
        if self._n % 500 == 0:      # periodic flush so a crash loses ≤500 rows
            self._fh.flush()

    def close(self) -> None:
        if self._fh is not None:
            try:
                self._fh.flush(); self._fh.close()
            except Exception:       # pragma: no cover
                pass
            self._fh = None


def read_ticks(path: Path) -> List[Dict]:
    """Read a gz tick file back into row dicts (for tests / offline RL dataset assembly).

    Tolerant of a TRUNCATED tail: a tick logger that crashed mid-write leaves the gz member
    without its trailer, so decompression raises EOFError at the end. We keep every row read
    before that point — crash recovery of the archive must not lose the flushed data."""
    out: List[Dict] = []
    try:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    break                          # partial final line from a crash → stop
    except (EOFError, OSError):
        pass                                       # truncated gz tail → return what we have
    return out


class TickLogger:
    """Best-effort async bookTicker stream → TickWriter. Reconnects with backoff; never raises
    into the caller (a tick-capture hiccup must not affect trading)."""

    def __init__(self, symbols: List[str], *, out_dir: Optional[str] = None,
                 testnet: Optional[bool] = None):
        from backend.core.config import settings
        self.symbols = [s.lower() for s in symbols]
        self.writer = TickWriter(out_dir=Path(out_dir or getattr(
            settings, "TICK_LOGGER_DIR", "training_data/ticks")))
        tn = settings.BINANCE_FUTURES_TESTNET if testnet is None else testnet
        streams = "/".join(f"{s}@bookTicker" for s in self.symbols)
        self.url = (WS_TESTNET if tn else WS_BASE) + streams
        self._stop = False

    async def run(self) -> None:
        import asyncio
        try:
            import aiohttp
        except Exception:
            log.warning("tick_logger_no_aiohttp"); return
        backoff = 1.0
        while not self._stop:
            try:
                async with aiohttp.ClientSession() as s:
                    async with s.ws_connect(self.url, heartbeat=30) as ws:
                        log.info("tick_logger_connected", symbols=self.symbols)
                        backoff = 1.0
                        async for msg in ws:
                            if self._stop:
                                break
                            if msg.type == aiohttp.WSMsgType.TEXT:
                                self._on_message(msg.data)
                            elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                                break
            except Exception as e:
                log.warning("tick_logger_ws_error", error=str(e)[:120])
            if not self._stop:
                await asyncio.sleep(min(60.0, backoff))
                backoff *= 2
        self.writer.close()

    def _on_message(self, data: str) -> None:
        try:
            payload = json.loads(data)
            d = payload.get("data", payload)      # combined-stream wraps in {"stream","data"}
            self.writer.write(symbol=d["s"], bid=float(d["b"]), ask=float(d["a"]),
                              bid_qty=float(d.get("B", 0.0)), ask_qty=float(d.get("A", 0.0)),
                              ts=float(d.get("E", time.time() * 1000)) / 1000.0)
        except Exception as e:  # pragma: no cover — malformed frame, skip
            log.warning("tick_logger_parse_failed", error=str(e)[:80])

    def stop(self) -> None:
        self._stop = True
