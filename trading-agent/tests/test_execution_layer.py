"""Execution layer: passive-limit flow (fill / timeout-sweep / post-only reject), slippage
ledger math, and router telemetry. All with fakes — no network, no keys."""
import asyncio
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from backend.execution.binance_futures_broker import BinanceFuturesBroker  # noqa: E402
from backend.execution.slippage_ledger import SlippageLedger, slippage_bps  # noqa: E402
from backend.execution.live_router import LiveRouter  # noqa: E402


class FakeBinance(BinanceFuturesBroker):
    """Overrides the network surface: scripted order lifecycle, canned prices."""

    def __init__(self, *, fill_after_polls=0, gtx_expires=False, partial_qty=0.0):
        super().__init__(paper=True)
        self._live_override = True                      # exercise the live code path
        self.fill_after_polls = fill_after_polls        # -1 = never fill
        self.gtx_expires = gtx_expires
        self.partial_qty = partial_qty
        self.polls = 0
        self.calls = []                                  # (method, path, params)

    @property
    def live(self):  # force the live branch without credentials
        return self._live_override

    async def get_price(self, symbol):
        return 100.0

    async def get_book_ticker(self, symbol):
        return {"bid": "99.99", "ask": "100.01"}

    async def set_leverage(self, symbol):
        return None

    async def _signed(self, method, path, params):
        self.calls.append((method, path, dict(params)))
        if method == "POST" and params.get("type") == "LIMIT":
            if self.gtx_expires:
                return {"orderId": 1, "status": "EXPIRED", "executedQty": "0"}
            return {"orderId": 1, "status": "NEW", "executedQty": "0"}
        if method == "POST" and params.get("type") == "MARKET":
            return {"orderId": 2, "status": "FILLED", "avgPrice": "100.05",
                    "executedQty": str(params.get("quantity"))}
        if method == "GET" and path.endswith("/order"):
            self.polls += 1
            if 0 <= self.fill_after_polls < self.polls:
                return {"orderId": 1, "status": "FILLED", "avgPrice": "99.99",
                        "executedQty": "1.0"}
            return {"orderId": 1, "status": "NEW", "avgPrice": "0",
                    "executedQty": str(self.partial_qty)}
        if method == "DELETE":
            return {"orderId": 1, "status": "CANCELED"}
        raise AssertionError(f"unexpected call {method} {path}")


def _run(coro):
    return asyncio.get_event_loop_policy().new_event_loop().run_until_complete(coro)


def test_passive_fills_without_sweep():
    b = FakeBinance(fill_after_polls=1)
    res = _run(b.submit_notional("BTCUSDT", 100.0, style="passive",
                                 limit_timeout_s=5.0, poll_s=0.01))
    assert res["status"] == "live" and res["style"] == "passive"
    assert res["fill_price"] == 99.99                       # joined the bid, earned the spread
    market_calls = [c for c in b.calls if c[2].get("type") == "MARKET"]
    assert not market_calls                                  # no sweep needed


def test_passive_timeout_sweeps_remainder_with_market():
    b = FakeBinance(fill_after_polls=-1, partial_qty=0.4)    # never fills; 0.4 done passively
    res = _run(b.submit_notional("BTCUSDT", 100.0, style="passive",
                                 limit_timeout_s=0.05, poll_s=0.01))
    assert res["status"] == "live" and res["style"] == "passive+sweep"
    cancels = [c for c in b.calls if c[0] == "DELETE"]
    sweeps = [c for c in b.calls if c[2].get("type") == "MARKET"]
    assert cancels and sweeps
    assert abs(float(sweeps[0][2]["quantity"]) - (1.0 - 0.4)) < 1e-9   # only the remainder
    assert res["executed_qty"] == pytest.approx(1.0)


def test_gtx_reject_falls_back_to_market():
    b = FakeBinance(gtx_expires=True)
    res = _run(b.submit_notional("BTCUSDT", 100.0, style="passive",
                                 limit_timeout_s=1.0, poll_s=0.01))
    assert res["status"] == "live" and res["style"] == "market"
    assert res["fill_price"] == 100.05


def test_paper_mode_never_hits_signed_endpoints():
    b = FakeBinance()
    b._live_override = False
    res = _run(b.submit_notional("BTCUSDT", 100.0, style="passive"))
    assert res["status"] == "paper"
    assert not [c for c in b.calls if c[0] == "POST"]


def test_slippage_bps_sign_convention():
    assert slippage_bps("BUY", 100.0, 100.10) == pytest.approx(10.0)    # paid up = adverse
    assert slippage_bps("SELL", 100.0, 99.90) == pytest.approx(10.0)    # sold down = adverse
    assert slippage_bps("BUY", 100.0, 99.95) == pytest.approx(-5.0)     # price improvement
    assert slippage_bps("BUY", 0.0, 100.0) is None


def test_ledger_roundtrip_and_summary(tmp_path):
    led = SlippageLedger(path=tmp_path / "slip.jsonl", assumed_bps=7.5)
    led.record(symbol="BTCUSDT", venue="binance", side="BUY", notional=100,
               decision_price=100.0, fill_price=100.05, style="passive", status="live")
    led.record(symbol="BTCUSDT", venue="binance", side="SELL", notional=100,
               decision_price=100.0, fill_price=99.97, style="passive", status="live")
    s = led.summary()
    assert s["binance/passive"]["n"] == 2
    assert s["binance/passive"]["mean_bps"] == pytest.approx((5.0 + 3.0) / 2)
    assert s["binance/passive"]["vs_assumed_bps"] == pytest.approx(4.0 - 7.5)


def test_router_passes_style_and_records_ledger(tmp_path):
    b = FakeBinance(fill_after_polls=1)
    led = SlippageLedger(path=tmp_path / "slip.jsonl")
    router = LiveRouter(equity=10_000.0, binance=b, perp_order_style="passive",
                        limit_timeout_s=5.0, slippage_ledger=led)
    # patch poll cadence through the broker call by shrinking the timeout window
    out = _run(router("BTCUSDT", target_weight=0.01, delta=0.01))
    assert out["venue"] == "binance"
    assert out["result"]["style"] in ("passive", "passive+sweep")
    assert led.summary()                                    # a slippage row was written


class CountingBinance(FakeBinance):
    """Records each submit_notional call's notional to verify TWAP child sizing."""

    def __init__(self, **kw):
        super().__init__(fill_after_polls=1, **kw)
        self.submits = []

    async def submit_notional(self, symbol, notional_usd, *, reduce_only=False,
                              style="market", limit_timeout_s=30.0, poll_s=2.0):
        self.submits.append(notional_usd)
        return {"status": "live", "style": style, "decision_price": 100.0,
                "fill_price": 100.0, "executed_qty": abs(notional_usd) / 100.0}


def test_twap_slices_large_order_into_children():
    b = CountingBinance()
    # threshold 2000, 4 children, tiny window so the test doesn't actually sleep long.
    # max_order_notional raised so the 8k order isn't clamped before slicing.
    router = LiveRouter(equity=100_000.0, binance=b, slice_threshold=2_000.0,
                        slice_children=4, slice_window_s=0.02, max_order_notional=20_000.0)
    out = _run(router("BTCUSDT", target_weight=0.08, delta=0.08))  # 0.08 * 100k = 8k > 2k
    assert out["sliced"] is True
    assert len(out["children"]) == 4
    assert len(b.submits) == 4
    assert all(abs(n - 8_000 / 4) < 1e-6 for n in b.submits)      # equal children
    assert abs(sum(b.submits) - 8_000) < 1e-6                      # sum to the parent


def test_twap_not_triggered_below_threshold():
    b = CountingBinance()
    router = LiveRouter(equity=100_000.0, binance=b, slice_threshold=2_000.0, slice_children=4)
    out = _run(router("BTCUSDT", target_weight=0.01, delta=0.01))  # 1k < 2k → single order
    assert out.get("sliced") is not True
    assert len(b.submits) == 1
