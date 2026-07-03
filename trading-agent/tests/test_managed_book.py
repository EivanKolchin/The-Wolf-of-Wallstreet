"""Live target-weights tests: the book must go long assets in an uptrend, flat below trend, keep
gross bounded, and de-risk (shrink weights) when VIX is elevated."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from backend.strategies.managed_book import target_weights  # noqa: E402
from backend.strategies.managed_beta import ManagedBetaParams  # noqa: E402


def _series(close, start="2015-01-01"):
    c = np.asarray(close, float)
    ts = pd.date_range(start, periods=len(c), freq="1D")
    return pd.DataFrame({"timestamp": ts, "open": c, "high": c + 1, "low": c - 1, "close": c,
                         "volume": np.ones(len(c))})


def _universe(n=300):
    rng = np.random.default_rng(0)
    up = 100 + np.cumsum(np.abs(rng.standard_normal(n)) * 0.3 + 0.2)      # clear uptrend
    up2 = 100 + np.cumsum(np.abs(rng.standard_normal(n)) * 0.3 + 0.25)
    dn = 200 - np.cumsum(np.abs(rng.standard_normal(n)) * 0.3 + 0.2)      # clear downtrend
    dn2 = 200 - np.cumsum(np.abs(rng.standard_normal(n)) * 0.3 + 0.25)
    return {"A": _series(up), "B": _series(up2), "C": _series(np.maximum(dn, 1)),
            "D": _series(np.maximum(dn2, 1))}


def test_target_weights_long_uptrend_flat_downtrend():
    w = target_weights(_universe(), ManagedBetaParams(trend_ema=50), target_vol=0.15, max_leverage=2.0)
    assert w["A"] > 0 and w["B"] > 0            # uptrend assets are held long
    assert w["C"] == 0.0 and w["D"] == 0.0      # below-trend assets are flat (cash)
    gross = sum(abs(v) for v in w.values())
    assert 0 < gross <= 2.0 + 1e-6              # gross bounded by max leverage


def test_target_weights_vix_derisk_shrinks_gross():
    bars = _universe()
    n = len(bars["A"])
    lo = target_weights(bars, ManagedBetaParams(trend_ema=50), vix=np.full(n, 12.0))
    hi = target_weights(bars, ManagedBetaParams(trend_ema=50), vix=np.full(n, 45.0))
    assert sum(abs(v) for v in hi.values()) < sum(abs(v) for v in lo.values())   # high VIX de-risks


def test_target_weights_insufficient_universe_is_flat():
    w = target_weights({"A": _series(np.linspace(100, 110, 300))}, ManagedBetaParams(trend_ema=50))
    assert all(v == 0.0 for v in w.values())    # need >=2 aligned assets
