"""Macro risk-filter tests: the de-risk scalars must be bounded [floor,1], monotonic in stress
(higher VIX / higher correlation → more de-risking), and strictly causal (use only past data)."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from backend.signals.macro_regime import (  # noqa: E402
    vix_risk_scalar, correlation_risk_scalar, regime_scalar_from_proba,
    combined_macro_scalar, avg_pairwise_correlation,
)


def test_vix_scalar_monotone_and_bounded():
    vix = np.array([10, 18, 25, 32, 45], float)
    s = vix_risk_scalar(vix, lo=18, hi=32, floor=0.3, shift=0)
    assert s[0] == 1.0 and s[1] == 1.0          # at/below lo → full risk-on
    assert abs(s[3] - 0.3) < 1e-9 and abs(s[4] - 0.3) < 1e-9   # at/above hi → floor
    assert s[1] > s[2] > s[3]                   # higher VIX → more de-risk
    assert np.all((s >= 0.3) & (s <= 1.0))


def test_vix_scalar_is_causal():
    vix = np.array([40, 10, 10, 10], float)     # a spike only at t=0
    s = vix_risk_scalar(vix, lo=18, hi=32, floor=0.3, shift=1)
    assert s[0] == 1.0                           # bar 0 cannot use its own (or future) VIX
    assert abs(s[1] - 0.3) < 1e-9                # bar 1 sees t=0's spike → de-risked


def test_correlation_scalar_cuts_when_assets_move_together():
    rng = np.random.default_rng(0)
    n = 300
    a = rng.standard_normal(n) * 0.01
    indep = np.column_stack([a, rng.standard_normal(n) * 0.01, rng.standard_normal(n) * 0.01])
    together = np.column_stack([a, a + 1e-6, a - 1e-6])      # ~perfectly correlated
    s_indep = correlation_risk_scalar(indep, window=50, shift=0)
    s_together = correlation_risk_scalar(together, window=50, shift=0)
    assert np.nanmean(s_together[60:]) < np.nanmean(s_indep[60:])   # crisis-like corr → de-risk
    assert np.all((s_together[60:] >= 0.5 - 1e-9) & (s_together[60:] <= 1.0 + 1e-9))


def test_avg_pairwise_correlation_detects_comovement():
    rng = np.random.default_rng(1)
    a = rng.standard_normal(200) * 0.01
    together = np.column_stack([a, a, a])
    avg = avg_pairwise_correlation(together, window=50)
    assert np.nanmean(avg[60:]) > 0.95          # identical series → corr ~1


def test_regime_scalar_from_proba_maps_ends():
    p = np.array([0.0, 0.5, 1.0], float)
    s = regime_scalar_from_proba(p, floor=0.3, shift=0)
    assert s[0] == 1.0 and abs(s[2] - 0.3) < 1e-9 and abs(s[1] - 0.65) < 1e-9


def test_combined_macro_scalar_compounds_and_handles_missing():
    vix = np.full(100, 40.0)                     # stressed VIX
    rng = np.random.default_rng(2)
    rets = np.column_stack([rng.standard_normal(100) * 0.01 for _ in range(3)])
    both = combined_macro_scalar(vix=vix, returns=rets, floor=0.2)
    vonly = combined_macro_scalar(vix=vix, floor=0.2)
    ronly = combined_macro_scalar(returns=rets, floor=0.2)
    assert np.all((both >= 0.2) & (both <= 1.0))
    assert np.all(both <= vonly + 1e-9)          # adding a 2nd de-risk source can only cut further
    assert len(vonly) == 100 and len(ronly) == 100
