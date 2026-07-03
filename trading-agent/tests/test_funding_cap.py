"""Funding-aware leverage cap — EMA smoothing, continuous ramp, sign-awareness, anti-whipsaw."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from backend.risk.funding_cap import (  # noqa: E402
    FundingCapConfig, FundingCapTracker, ema_funding, funding_scale,
    funding_scale_series, PRINTS_PER_YEAR,
)
from backend.strategies.combined_book import ts_momentum_target_weights  # noqa: E402


def test_ema_is_causal_and_converges():
    prints = np.concatenate([np.zeros(10), np.full(50, 0.001)])
    ema = ema_funding(prints, halflife_prints=9.0)
    assert ema[0] == 0.0
    assert ema[9] == 0.0                      # untouched by the future step-up
    assert 0 < ema[15] < 0.001                # converging, not jumping
    assert ema[-1] > 0.0009                   # near the new level after ~5 half-lives


def test_scale_ramp_monotonic_and_bounded():
    cfg = FundingCapConfig()
    lo, hi, fl = cfg.lo_annual, cfg.hi_annual, cfg.floor
    assert funding_scale(-1.0, lo, hi, fl) == 1.0        # receiving side: never capped
    assert funding_scale(0.0, lo, hi, fl) == 1.0
    assert funding_scale(lo, lo, hi, fl) == 1.0          # at lo: still full
    assert abs(funding_scale((lo + hi) / 2, lo, hi, fl) - (1 + fl) / 2) < 1e-9
    assert funding_scale(hi, lo, hi, fl) == fl           # at hi: floored
    assert funding_scale(10.0, lo, hi, fl) == fl         # beyond: stays floored, never 0
    grid = funding_scale(np.linspace(-1, 2, 200), lo, hi, fl)
    assert np.all(np.diff(grid) <= 1e-12)                # monotone non-increasing in cost


def test_sign_awareness_short_receives():
    """Positive funding: longs pay (capped), shorts receive (never capped) — and vice versa."""
    tr = FundingCapTracker()
    for _ in range(30):
        tr.update("BTCUSDT", 0.0005)          # strongly positive: 0.05%/8h ≈ 55%/yr
    assert tr.scale("BTCUSDT", +1) == tr.cfg.floor       # long pays → floored
    assert tr.scale("BTCUSDT", -1) == 1.0                # short receives → untouched
    tr2 = FundingCapTracker()
    for _ in range(30):
        tr2.update("ETHUSDT", -0.0005)
    assert tr2.scale("ETHUSDT", -1) == tr2.cfg.floor     # short pays negative funding
    assert tr2.scale("ETHUSDT", +1) == 1.0


def test_unknown_symbol_or_flat_position_uncapped():
    tr = FundingCapTracker()
    assert tr.scale("NEVERSEEN", +1) == 1.0
    tr.update("BTCUSDT", 0.001)
    assert tr.scale("BTCUSDT", 0) == 1.0


def test_anti_whipsaw_vs_raw_threshold():
    """Alternating extreme prints (the flip scenario): a raw threshold rule flips the cap
    every print; the EMA'd ramp holds it nearly constant."""
    rng = np.random.default_rng(0)
    prints = 0.0006 * np.where(np.arange(120) % 2 == 0, 1.0, -1.0)   # ±0.06%/8h alternating
    prints += 0.00005 * rng.standard_normal(120)
    signs = np.ones(120)                                              # constant long

    smoothed = funding_scale_series(prints, signs)
    raw_cost = np.sign(signs) * prints * PRINTS_PER_YEAR
    cfg = FundingCapConfig()
    raw_rule = np.where(raw_cost > cfg.lo_annual, cfg.floor, 1.0)     # naive threshold cap

    flips_raw = int(np.abs(np.diff(raw_rule) > 1e-9).sum())
    smooth_turn = float(np.abs(np.diff(smoothed)).sum())
    raw_turn = float(np.abs(np.diff(raw_rule)).sum())
    assert flips_raw > 30                                # the naive rule thrashes
    assert smooth_turn < raw_turn * 0.05                 # the smoothed ramp barely moves


def test_sustained_flip_does_repricecap_slowly():
    """A REAL regime flip (sustained sign change) does move the cap — over ~a half-life,
    not instantly."""
    prints = np.concatenate([np.full(30, 0.0006), np.full(30, -0.0006)])
    signs = np.ones(60)
    s = funding_scale_series(prints, signs)
    assert s[29] < 0.5                                   # paying side capped hard
    assert s[33] < 0.9                                   # 4 prints after flip: still cautious
    assert s[-1] == 1.0                                  # fully released once EMA crosses


def test_ts_sleeve_weights_apply_funding_cap():
    import pandas as pd
    rng = np.random.default_rng(1)
    n = 400
    bars = {}
    for s in ("AAAUSDT", "BBBUSDT"):
        close = 100 * np.cumprod(1 + 0.003 + 0.01 * rng.standard_normal(n))  # strong uptrend
        bars[s] = pd.DataFrame({"open": close, "high": close * 1.005, "low": close * 0.995,
                                "close": close, "volume": np.ones(n)})
    base = ts_momentum_target_weights(bars)
    capped = ts_momentum_target_weights(
        bars, funding_annual={"AAAUSDT": 0.60})          # AAA longs pay 60%/yr smoothed
    if base["AAAUSDT"] > 0:                              # sleeve is long the uptrend
        assert capped["AAAUSDT"] < base["AAAUSDT"]       # capped down…
        assert capped["AAAUSDT"] >= base["AAAUSDT"] * 0.25 - 1e-12   # …but only to the floor
    assert abs(capped["BBBUSDT"] - base["BBBUSDT"]) < 1e-12          # no funding data → untouched
