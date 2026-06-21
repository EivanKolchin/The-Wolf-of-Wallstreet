"""Unit tests for the honest evaluation metrics: regression alpha/beta separation and
the cost-aware net-alpha selection score. These are the numbers that gate promotion and
drive checkpoint selection, so they must behave predictably."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from backend.backtest.engine import regression_alpha_beta, net_alpha_score  # noqa: E402


def test_regression_pure_beta_has_zero_alpha():
    rng = np.random.default_rng(0)
    mkt = rng.standard_normal(2000) * 0.01
    strat = 1.3 * mkt                      # pure market exposure, no skill
    alpha, beta = regression_alpha_beta(strat, mkt, bars_per_year=1.0)
    assert abs(beta - 1.3) < 1e-6
    assert abs(alpha) < 1e-6               # no alpha when returns are pure beta


def test_regression_detects_alpha_above_beta():
    rng = np.random.default_rng(1)
    mkt = rng.standard_normal(5000) * 0.01
    edge = 0.0005                          # constant per-bar skill on top of beta
    strat = 0.5 * mkt + edge
    alpha, beta = regression_alpha_beta(strat, mkt, bars_per_year=1.0)
    assert abs(beta - 0.5) < 0.05
    assert abs(alpha - edge) < 1e-4        # intercept recovers the skill term


def test_net_alpha_rewards_correct_low_cost_signal():
    # Probabilities that always go long, with forward returns that are mostly up:
    n = 500
    p_long = np.full(n, 0.7); p_short = np.full(n, 0.1)
    fwd = np.full(n, 0.01)                 # +1% realized each time, dwarfs cost
    score = net_alpha_score(p_long, p_short, fwd, min_confidence=0.45, min_edge=0.05)
    assert score > 0                       # consistent profitable edge -> positive


def test_net_alpha_punishes_costly_wrong_signal():
    n = 500
    p_long = np.full(n, 0.7); p_short = np.full(n, 0.1)
    fwd = np.full(n, -0.01)                # always wrong (price falls when we go long)
    score = net_alpha_score(p_long, p_short, fwd)
    assert score < 0


def test_net_alpha_flat_when_gate_never_fires():
    n = 500
    # Below confidence/edge gate everywhere -> no trades -> score 0 (not NaN).
    p_long = np.full(n, 0.34); p_short = np.full(n, 0.33)
    fwd = np.random.default_rng(2).standard_normal(n) * 0.01
    score = net_alpha_score(p_long, p_short, fwd, min_confidence=0.45, min_edge=0.05)
    assert score == 0.0


def test_net_alpha_cost_flips_thin_edge():
    # A tiny realized edge smaller than round-trip cost must score negative.
    n = 500
    p_long = np.full(n, 0.7); p_short = np.full(n, 0.1)
    fwd = np.full(n, 0.0005)               # 5 bps move < 15 bps round-trip cost
    score = net_alpha_score(p_long, p_short, fwd, fee_bps=10.0, slippage_bps=5.0)
    assert score < 0
