"""Daily multi-asset TSMOM sleeve — direction, vol scaling, causality, turnover guard."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from backend.strategies.tsmom_multiasset import (  # noqa: E402
    TSMOMMultiAsset, TSMOMParams, tsmom_target_weights,
)


def _df(close):
    close = np.asarray(close, float)
    return pd.DataFrame({
        "open": close, "high": close * 1.001, "low": close * 0.999,
        "close": close, "volume": np.ones_like(close),
        "timestamp": pd.date_range("2018-01-01", periods=len(close), freq="D"),
    })


def _trend(n, drift, vol=0.005, seed=0):
    rng = np.random.default_rng(seed)
    return 100.0 * np.cumprod(1.0 + drift + vol * rng.standard_normal(n))


def test_long_in_uptrend_short_in_downtrend():
    n = 600
    up, dn = _df(_trend(n, +0.002)), _df(_trend(n, -0.002))
    pos = TSMOMMultiAsset().generate_positions({"UP": up, "DN": dn})
    assert pos["UP"][-1] > 0.05
    assert pos["DN"][-1] < -0.05


def test_vol_scaling_shrinks_high_vol_asset():
    """Deterministic construction: same +0.3%/day drift, but the wild asset carries a
    ±3% zigzag → same trend sign, ~10× the daily vol → materially smaller weight."""
    n = 600
    t = np.arange(n)
    drift = np.cumprod(np.full(n, 1.003))
    calm = _df(100.0 * drift * (1.0 + 0.002 * np.sin(t)))          # tiny wiggle (vol ≈ 0.14%)
    wild = _df(100.0 * drift * (1.0 + 0.03 * ((-1.0) ** t)))        # ±3% zigzag (vol ≈ 3%)
    pos = TSMOMMultiAsset().generate_positions({"CALM": calm, "WILD": wild})
    assert pos["CALM"][-1] > 0
    assert pos["WILD"][-1] > 0
    assert abs(pos["WILD"][-1]) < abs(pos["CALM"][-1]) * 0.6


def test_causal_prefix_stable():
    """position[t] must not depend on data after t: truncating the series must
    reproduce the same prefix exactly."""
    n = 700
    full = _trend(n, +0.001, seed=3)
    strat = TSMOMMultiAsset()
    p_full = strat.generate_positions({"A": _df(full)})["A"]
    p_cut = strat.generate_positions({"A": _df(full[: n - 120])})["A"]
    assert np.allclose(p_full[: n - 120], p_cut)


def test_warmup_is_flat():
    n = 600
    pos = TSMOMMultiAsset().generate_positions({"A": _df(_trend(n, +0.002))})["A"]
    warm = int(max(TSMOMParams().lookbacks) * TSMOMParams().min_history_factor)
    assert np.allclose(pos[:warm], 0.0)
    # too-short history → all flat
    short = TSMOMMultiAsset().generate_positions({"A": _df(_trend(100, +0.002))})["A"]
    assert np.allclose(short, 0.0)


def test_dead_band_limits_rebalance_count():
    """The band's job is FEWER rebalances: without it the vol-scaled target drifts
    every bar; with it the position only moves on material drift or a sign flip."""
    n = 800
    # vol 2%/day → ann vol ~32% > the 15% per-asset target, so the vol scale is BELOW
    # the 1.0 clip and fluctuates bar to bar (otherwise the target is constant and
    # neither variant ever rebalances).
    close = _trend(n, +0.004, vol=0.02, seed=4)
    banded = TSMOMMultiAsset(TSMOMParams(dead_band=0.10)).generate_positions({"A": _df(close)})["A"]
    tight = TSMOMMultiAsset(TSMOMParams(dead_band=0.0)).generate_positions({"A": _df(close)})["A"]
    n_banded = int((np.diff(banded) != 0).sum())
    n_tight = int((np.diff(tight) != 0).sum())
    assert n_banded < n_tight * 0.3


def test_profitable_on_strong_synthetic_trends():
    """Long/short symmetric: profits on an up-trender AND a down-trender (gross)."""
    n = 900
    for drift, seed in ((+0.002, 5), (-0.002, 6)):
        close = _trend(n, drift, vol=0.008, seed=seed)
        pos = TSMOMMultiAsset().generate_positions({"A": _df(close)})["A"]
        ret = np.zeros(n)
        ret[1:] = close[1:] / close[:-1] - 1.0
        pnl = (pos[:-1] * ret[1:]).sum()
        assert pnl > 0, f"drift={drift} not monetised"


def test_target_weights_latest_bar():
    n = 600
    bars = {"UP": _df(_trend(n, +0.002, seed=7)), "DN": _df(_trend(n, -0.002, seed=8)),
            "SHORT_HIST": _df(_trend(50, +0.002, seed=9))}
    w = tsmom_target_weights(bars)
    assert set(w) == {"UP", "DN", "SHORT_HIST"}
    assert w["UP"] > 0 and w["DN"] < 0
    assert w["SHORT_HIST"] == 0.0              # not enough history → flat, but key present
    # equal-capital split across the 2 usable names → |w| ≤ 1/2
    assert abs(w["UP"]) <= 0.5 + 1e-9 and abs(w["DN"]) <= 0.5 + 1e-9
