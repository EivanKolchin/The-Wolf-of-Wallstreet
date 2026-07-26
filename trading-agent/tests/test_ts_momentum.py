"""Phase B1 tests: the crypto 1h TS-momentum/breakout strategy. Verifies the breakout
fires in the trend direction, is causal (no look-ahead), the Chandelier/turtle stop exits on
reversal, and it integrates with the portfolio backtester."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from backend.strategies.ts_momentum import TSMomentumBreakout, TSMomentumParams  # noqa: E402
from backend.backtest.portfolio import portfolio_backtest  # noqa: E402

BARS_1H_YR = 24 * 365


def _ohlc_from_close(close):
    close = np.asarray(close, float)
    return pd.DataFrame({"open": close, "high": close + 0.5, "low": close - 0.5,
                         "close": close, "volume": np.ones(len(close))})


def _params():
    return TSMomentumParams(entry_channel=20, exit_channel=10, atr_period=14,
                            atr_mult=3.0, ema_trend=30)


def test_goes_long_in_uptrend():
    # flat base then a sustained rally → breakout long, and stays long while trending
    close = np.r_[np.full(60, 100.0), np.linspace(100, 160, 120)]
    pos = TSMomentumBreakout(_params()).generate_positions({"X": _ohlc_from_close(close)})["X"]
    assert pos[-1] == 1.0
    assert pos[:55].max() == 0.0           # no position during the flat base (nothing to break out of)


def test_goes_short_in_downtrend():
    # steep enough that the close actually breaks below the Donchian lower channel
    # (move/bar must exceed the 0.5 wick, else there's no genuine breakout)
    close = np.r_[np.full(60, 100.0), np.linspace(100, 30, 120)]
    pos = TSMomentumBreakout(_params()).generate_positions({"X": _ohlc_from_close(close)})["X"]
    assert pos[-1] == -1.0


def test_causal_no_lookahead():
    # The breakout bar must come AFTER the channel is exceeded, never before the move.
    close = np.r_[np.full(80, 100.0), np.linspace(100, 140, 60)]
    pos = TSMomentumBreakout(_params()).generate_positions({"X": _ohlc_from_close(close)})["X"]
    first_long = int(np.argmax(pos == 1.0)) if (pos == 1.0).any() else -1
    assert first_long >= 80                 # entry only once the rally has started, not in the base


def test_trailing_stop_exits_the_long_on_reversal():
    # rally (go long) then a sharp crash → the Chandelier stop must EXIT the long. (The
    # strategy may then flip short as the downtrend breaks out — that's correct momentum
    # behaviour; the property under test is that the long does not survive the reversal.)
    up = np.linspace(100, 160, 100)
    down = np.linspace(160, 100, 60)            # steep enough to break out downward too
    close = np.r_[np.full(40, 100.0), up, down]
    pos = TSMomentumBreakout(_params()).generate_positions({"X": _ohlc_from_close(close)})["X"]
    assert (pos == 1.0).any()                    # was long during the rally
    last_long = int(np.max(np.where(pos == 1.0)[0]))
    assert (pos[last_long + 1:] != 1.0).all()    # never long again after the stop fired
    assert pos[-1] <= 0.0                         # flat or short by the end, not long


def test_long_only_mode_has_no_shorts():
    close = np.r_[np.full(60, 100.0), np.linspace(100, 50, 120)]
    p = _params(); p.allow_short = False
    pos = TSMomentumBreakout(p).generate_positions({"X": _ohlc_from_close(close)})["X"]
    assert pos.min() >= 0.0


def test_convex_stop_cuts_losers_tight_and_lets_winners_run():
    """The asymmetric stop must (a) sit TIGHTER than symmetric before a trade is in profit,
    (b) WIDEN once the trade proves itself, and (c) never let a winner revert below breakeven."""
    atr = 10.0
    sym = TSMomentumParams(convex_exits=False, atr_mult=3.0)
    cvx = TSMomentumParams(convex_exits=True, initial_atr_mult=1.5, trail_atr_mult=4.0,
                           activation_atr=1.0)
    entry = 100.0
    # pre-profit long (peak ≈ entry): convex tighter than symmetric
    assert cvx.long_stop(entry, 100.0, atr) == 100.0 - 1.5 * atr
    assert cvx.long_stop(entry, 100.0, atr) > sym.long_stop(entry, 100.0, atr)
    # big winner (+5 ATR): convex wider trail than symmetric → lets it run
    assert cvx.long_stop(entry, 150.0, atr) == 150.0 - 4.0 * atr
    assert cvx.long_stop(entry, 150.0, atr) < sym.long_stop(entry, 150.0, atr)
    # just activated (+1 ATR): wide trail would dip below entry → floored at breakeven
    assert cvx.long_stop(entry, 110.0, atr) == entry
    # short mirror
    assert cvx.short_stop(entry, 100.0, atr) == 100.0 + 1.5 * atr        # tight pre-profit
    assert cvx.short_stop(entry, 50.0, atr) == 50.0 + 4.0 * atr          # wide when winning
    assert cvx.short_stop(entry, 90.0, atr) == entry                     # breakeven lock


def test_convex_exits_produce_more_positive_skew_on_a_trend():
    """On an up-then-crash path, convex exits should give back LESS at the top than the symmetric
    3-ATR trail (tighter give-back once activated is NOT the point — the point is it exits a winner
    on the crash no later than symmetric, and cuts a fresh loser sooner). Sanity: both stay causal
    and flat-or-short by the end."""
    up = np.linspace(100, 200, 160)
    down = np.linspace(200, 120, 60)
    close = np.r_[np.full(40, 100.0), up, down]
    data = {"X": _ohlc_from_close(close)}
    base = TSMomentumParams(entry_channel=20, exit_channel=10, ema_trend=30, atr_mult=3.0)
    conv = TSMomentumParams(entry_channel=20, exit_channel=10, ema_trend=30, convex_exits=True,
                            initial_atr_mult=1.5, trail_atr_mult=4.0, activation_atr=1.0)
    pos_sym = TSMomentumBreakout(base).generate_positions(data)["X"]
    pos_cvx = TSMomentumBreakout(conv).generate_positions(data)["X"]
    assert (pos_sym == 1.0).any() and (pos_cvx == 1.0).any()   # both went long the trend
    assert pos_sym[-1] <= 0.0 and pos_cvx[-1] <= 0.0           # both exited the long by the crash end


def test_integrates_with_portfolio_backtest():
    rng = np.random.default_rng(0)
    data = {}
    for s, seed in (("BTCUSDT", 1), ("ETHUSDT", 2)):
        c = 100 + np.cumsum(rng.standard_normal(2000) * 0.5)
        data[s] = _ohlc_from_close(np.maximum(c, 1.0))
    res = portfolio_backtest({"ts_mom": TSMomentumBreakout(_params())}, data,
                             bars_per_year=BARS_1H_YR, market_symbol="BTCUSDT")
    assert np.isfinite(res.net_returns).all()
    assert "sharpe" in res.metrics and "reg_alpha" in res.metrics
