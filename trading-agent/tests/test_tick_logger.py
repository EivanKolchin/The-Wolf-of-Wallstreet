"""L1 tick logger — writer rotation, round-trip read, and message parsing (no network)."""
import gzip
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.data.tick_logger import TickWriter, TickLogger, read_ticks  # noqa: E402

_ = gzip  # used by fixtures below


def test_writer_roundtrip(tmp_path):
    w = TickWriter(out_dir=tmp_path)
    ts = time.mktime(time.strptime("2026-07-01", "%Y-%m-%d"))
    w.write("BTCUSDT", 100.0, 100.1, 5.0, 6.0, ts=ts)
    w.write("BTCUSDT", 100.2, 100.3, ts=ts + 1)
    w.close()
    files = list(tmp_path.glob("bookticker_*.jsonl.gz"))
    assert len(files) == 1
    rows = read_ticks(files[0])
    assert len(rows) == 2
    assert rows[0]["s"] == "BTCUSDT" and rows[0]["b"] == 100.0 and rows[0]["a"] == 100.1
    assert rows[0]["B"] == 5.0 and rows[0]["A"] == 6.0


def test_writer_rotates_by_utc_day(tmp_path):
    w = TickWriter(out_dir=tmp_path)
    d1 = time.mktime(time.strptime("2026-07-01 23:00", "%Y-%m-%d %H:%M"))
    d2 = time.mktime(time.strptime("2026-07-02 00:30", "%Y-%m-%d %H:%M"))
    # write in UTC terms — use explicit gmtime-aligned epochs
    import calendar
    e1 = calendar.timegm(time.strptime("2026-07-01 23:00", "%Y-%m-%d %H:%M"))
    e2 = calendar.timegm(time.strptime("2026-07-02 00:30", "%Y-%m-%d %H:%M"))
    w.write("BTCUSDT", 1, 2, ts=e1)
    w.write("BTCUSDT", 3, 4, ts=e2)
    w.close()
    files = sorted(p.name for p in tmp_path.glob("bookticker_*.jsonl.gz"))
    assert files == ["bookticker_20260701.jsonl.gz", "bookticker_20260702.jsonl.gz"]


def test_flush_survives_no_close(tmp_path):
    """Crash recovery: rows flushed before an abrupt stop (no close → truncated gz tail) are
    still recoverable via the tolerant read_ticks."""
    w = TickWriter(out_dir=tmp_path)
    import calendar
    e = calendar.timegm(time.strptime("2026-07-01", "%Y-%m-%d"))
    for i in range(1200):
        w.write("BTCUSDT", 100 + i * 1e-4, 100 + i * 1e-4 + 0.1, ts=e + i)
    # simulate a crash: DON'T close. flushes fired at 500 and 1000.
    f = next(tmp_path.glob("bookticker_*.jsonl.gz"))
    rows = read_ticks(f)
    assert len(rows) >= 1000                        # the flushed rows survived the "crash"
    w.close()


def test_logger_parses_combined_stream_message(tmp_path):
    lg = TickLogger(["BTCUSDT"], out_dir=str(tmp_path), testnet=True)
    msg = ('{"stream":"btcusdt@bookTicker","data":{"s":"BTCUSDT","b":"64000.1",'
           '"a":"64000.5","B":"1.2","A":"0.8","E":1751000000000}}')
    lg._on_message(msg)
    lg.writer.close()
    rows = read_ticks(next(tmp_path.glob("*.gz")))
    assert len(rows) == 1 and rows[0]["s"] == "BTCUSDT"
    assert rows[0]["b"] == 64000.1 and rows[0]["a"] == 64000.5


def test_logger_url_uses_testnet_flag():
    lg = TickLogger(["BTCUSDT", "ETHUSDT"], testnet=True)
    assert "stream.binancefuture.com" in lg.url
    assert "btcusdt@bookTicker" in lg.url and "ethusdt@bookTicker" in lg.url
    lg2 = TickLogger(["BTCUSDT"], testnet=False)
    assert "fstream.binance.com" in lg2.url
