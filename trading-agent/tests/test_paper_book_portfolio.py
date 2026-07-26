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


def test_state_persistence_roundtrip():
    """to_state/load_state must resume the book EXACTLY — the fix for restart-wiped P&L."""
    b = PaperBook(equity=100_000.0, cost_bps=5.0)
    _run(b.apply("BTCUSDT", 0.5, 0.5))
    b.mark_to_market({"BTCUSDT": 100.0}, time.time())
    b.mark_to_market({"BTCUSDT": 110.0}, time.time() + 1)      # +10% → book +5% (minus cost)
    st = b.to_state()
    restored = PaperBook(equity=100_000.0)                     # fresh seed (simulates a restart)
    restored.load_state(st)
    assert abs(restored.equity - b.equity) < 1e-6             # equity resumed, NOT reset to 100k
    assert restored.positions == b.positions
    assert abs(restored.last_prices["BTCUSDT"] - 110.0) < 1e-6
    assert abs(restored.realized_cost - b.realized_cost) < 1e-6
    assert restored.initial_equity == 100_000.0
    assert len(restored.net_worth_history) == len(b.net_worth_history)
    # a subsequent mark continues from the resumed equity (no phantom reset gain)
    r = restored.mark_to_market({"BTCUSDT": 110.0}, time.time() + 2)
    assert abs(r) < 1e-9


def test_conviction_tilt_bets_bigger_on_aligned_trends():
    """Kelly-conviction: a position aligned with a strong trend gets multiplier > 1; a counter-trend
    position < 1; a flat/no-edge position ~ 1. Bounded by [1/cap, cap]."""
    import numpy as np
    import pandas as pd
    from backend.agents.strategy_agent import StrategyAgent

    def ramp(slope, n=150, p0=100.0, seed=3):
        rng = np.random.default_rng(seed)
        close = p0 * np.cumprod(1.0 + slope * 0.01 + rng.normal(0, 0.004, n))   # trend + noise
        return pd.DataFrame({"timestamp": pd.date_range("2025-01-01", periods=n, freq="1D"), "close": close})

    agent = StrategyAgent(universe=[], bar_provider=None, portfolio=PaperBook(),
                          conviction_gain=0.5, conviction_cap=2.0)
    bars = {"UP": ramp(+1.0), "DOWN": ramp(-1.0)}
    conv = agent._conviction_scale({"UP": 0.3, "DOWN": 0.3}, bars)   # both LONG
    # core property: aligned trend (long an uptrend) sized bigger than counter-trend (long a downtrend)
    assert conv["UP"] > 1.0 > conv["DOWN"]
    assert conv["UP"] > conv["DOWN"]
    # bounded by [1/cap, cap]
    assert all(0.5 <= v <= 2.0 for v in conv.values())
    # short an aligned downtrend → bigger bet (sign-aware)
    conv_s = agent._conviction_scale({"DOWN": -0.3}, {"DOWN": ramp(-1.0)})
    assert conv_s["DOWN"] > 1.0 and conv_s["DOWN"] <= 2.0
    # gain=0 → strictly neutral (multiplier 1.0 everywhere → the rebalance skips the tilt entirely)
    off = StrategyAgent(universe=[], bar_provider=None, portfolio=PaperBook(), conviction_gain=0.0)
    off_conv = off._conviction_scale({"UP": 0.3}, {"UP": ramp(+1.0)})
    assert all(abs(v - 1.0) < 1e-9 for v in off_conv.values())


def test_book_realized_vol_estimate():
    """_book_realized_vol should return a sane annualized vol from mixed-frequency bars."""
    import numpy as np
    import pandas as pd
    from backend.agents.strategy_agent import StrategyAgent
    rng = np.random.default_rng(1)

    def mk(vol_daily, n=180, p0=100.0):
        r = rng.normal(0.0, vol_daily, n)
        close = p0 * np.cumprod(1 + r)
        return pd.DataFrame({"timestamp": pd.date_range("2025-01-01", periods=n, freq="1D"),
                             "close": close})

    bars = {"SPY": mk(0.01), "BTCUSDT": mk(0.03)}
    rv = StrategyAgent._book_realized_vol({"SPY": 0.3, "BTCUSDT": 0.3}, bars, window=63)
    assert rv is not None and 0.02 < rv < 1.5
    # no held names → None (caller then skips scaling)
    assert StrategyAgent._book_realized_vol({"SPY": 0.0}, bars) is None
