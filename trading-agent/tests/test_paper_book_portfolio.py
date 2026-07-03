"""PaperBook mark-to-market + dashboard portfolio view (real P&L, cash, allocations, curve)."""
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.agents.strategy_agent import PaperBook  # noqa: E402


def _run(coro):
    return asyncio.get_event_loop_policy().new_event_loop().run_until_complete(coro)


def test_mark_to_market_reflects_pnl():
    b = PaperBook(equity=100_000.0, cost_bps=0.0)   # no cost → clean P&L check
    _run(b.apply("SPY", 0.5, 0.5))
    _run(b.apply("BTCUSDT", 0.2, 0.2))
    b.mark_to_market({"SPY": 100.0, "BTCUSDT": 50_000.0}, time.time())      # seed prices
    # SPY +10%, BTC -5% → book return = 0.5*0.10 + 0.2*(-0.05) = 0.04
    b.mark_to_market({"SPY": 110.0, "BTCUSDT": 47_500.0}, time.time() + 3600)
    v = b.portfolio_view()
    assert abs(v["total_value"] - 104_000.0) < 1.0
    assert abs(v["total_pnl"] - 4_000.0) < 1.0
    assert abs(v["total_pnl_pct"] - 4.0) < 0.02


def test_cost_drag_applied_on_rebalance():
    b = PaperBook(equity=100_000.0, cost_bps=5.0)
    _run(b.apply("SPY", 1.0, 1.0))               # 100% turnover → 5bps = $50
    assert abs(b.equity - 99_950.0) < 1e-6
    assert abs(b.realized_cost - 50.0) < 1e-6


def test_view_shape_cash_and_allocations():
    b = PaperBook(equity=100_000.0, cost_bps=0.0)
    _run(b.apply("SPY", 0.4, 0.4))
    _run(b.apply("BTCUSDT", -0.2, 0.2))          # a short
    b.mark_to_market({"SPY": 100.0, "BTCUSDT": 50_000.0}, time.time())
    v = b.portfolio_view()
    assert abs(v["gross_exposure"] - 0.6) < 1e-9
    assert abs(v["net_exposure"] - 0.2) < 1e-9
    assert abs(v["cash"] - 100_000.0 * 0.4) < 1.0   # uninvested = equity*(1-gross)
    syms = {a["symbol"]: a for a in v["allocations"]}
    assert syms["SPY"]["side"] == "long" and syms["BTCUSDT"]["side"] == "short"
    assert abs(syms["SPY"]["value"] - 40_000.0) < 1.0
    assert syms["BTCUSDT"]["value"] < 0            # short shows negative dollar exposure


def test_history_accumulates_and_caps():
    b = PaperBook(equity=100_000.0, cost_bps=0.0, max_history=10)
    _run(b.apply("SPY", 1.0, 1.0))
    for i in range(20):
        b.mark_to_market({"SPY": 100.0 + i}, time.time() + i)
    v = b.portfolio_view()
    assert len(b.net_worth_history) <= 10          # capped
    assert len(v["history"]) >= 1
    assert v["history"][-1]["value"] == b.net_worth_history[-1]["value"]


def test_seeded_with_initial_point():
    b = PaperBook(equity=100_000.0)
    assert b.initial_equity == 100_000.0
    assert len(b.net_worth_history) == 1           # __post_init__ seeds one point
    assert b.net_worth_history[0]["value"] == 100_000.0
