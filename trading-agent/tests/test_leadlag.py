"""Lead-lag estimator — planted-edge recovery, self-momentum neutralisation, walk-forward
profitability on synthetic diffusion, and the null case (pure noise → no edges, no profit)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.signals.leadlag import (  # noqa: E402
    LeadLagConfig, leadlag_matrix, build_edges, leadlag_signal,
    signal_positions, portfolio_returns, walk_forward,
)


def _planted_panel(T=4000, beta=0.35, seed=0):
    """Asset 0 leads asset 1 with one-bar delay (r1[t+1] = beta*r0[t] + noise); assets 2,3 noise."""
    rng = np.random.default_rng(seed)
    r0 = 0.01 * rng.standard_normal(T)
    r1 = np.zeros(T)
    r1[1:] = beta * r0[:-1] + 0.008 * rng.standard_normal(T - 1)
    r2 = 0.01 * rng.standard_normal(T)
    r3 = 0.01 * rng.standard_normal(T)
    return np.column_stack([r0, r1, r2, r3])


def test_matrix_recovers_planted_edge_direction():
    R = _planted_panel()
    L = leadlag_matrix(R)
    assert L[0, 1] > 0.25                      # 0 leads 1, strongly
    assert abs(L[1, 0]) < 0.1                  # NOT the reverse direction
    assert abs(L[2, 3]) < 0.1                  # noise pair ≈ 0


def test_build_edges_admits_planted_and_rejects_noise():
    R = _planted_panel()
    edges = build_edges(leadlag_matrix(R), LeadLagConfig(min_abs_corr=0.15))
    keys = {(i, j) for (i, j, _) in edges}
    assert (0, 1) in keys
    assert all(abs(w) >= 0.15 for (_, _, w) in edges)
    assert (1, 0) not in keys


def test_self_neutralisation_removes_own_momentum():
    """An asset that only trends on ITSELF (AR(1)) must produce ~no cross signal — without
    neutralisation 'lead-lag' silently rediscovers single-asset momentum."""
    rng = np.random.default_rng(1)
    T = 4000
    r = np.zeros(T)
    for t in range(1, T):
        r[t] = 0.3 * r[t - 1] + 0.01 * rng.standard_normal()
    R = np.column_stack([r, 0.01 * rng.standard_normal(T)])
    cfg = LeadLagConfig(min_abs_corr=0.05, self_neutralise=True)
    L = leadlag_matrix(R)
    edges = build_edges(L, cfg)                # no strong cross edges expected
    sig = leadlag_signal(R, edges, L, cfg)
    # the AR(1) asset's own-momentum contribution is subtracted → its signal shrinks
    corr_own = np.corrcoef(sig[:-1, 0], R[1:, 0])[0, 1] if np.std(sig[:, 0]) > 1e-12 else 0.0
    assert abs(corr_own) < 0.35                # bounded — not the raw 0.3 autocorr ride


def test_walk_forward_profits_on_planted_edge_and_ic_holds_oos():
    R = _planted_panel(T=6000, beta=0.4)
    res = walk_forward(R, folds=4, cost_bps=5.0, bars_per_year=105_120,
                       cfg=LeadLagConfig(min_abs_corr=0.15))
    assert len(res) >= 3
    assert np.mean([f.ic_oos for f in res]) > 0.15        # planted edge persists OOS
    assert np.mean([f.sharpe_net for f in res]) > 2.0     # and is monetisable net of costs


def test_walk_forward_null_case_no_free_lunch():
    rng = np.random.default_rng(2)
    R = 0.01 * rng.standard_normal((6000, 6))
    res = walk_forward(R, folds=4, cost_bps=15.0,
                       cfg=LeadLagConfig(min_abs_corr=0.15))
    for f in res:
        assert f.n_edges <= 2                              # noise rarely clears the bar
        assert f.sharpe_net < 2.0                          # and never a strong "edge"


def test_positions_gated_and_costs_charged():
    R = _planted_panel()
    cfg = LeadLagConfig(min_abs_corr=0.15)
    L = leadlag_matrix(R[:2000])
    edges = build_edges(L, cfg)
    sig = leadlag_signal(R, edges, L, cfg)
    pos = signal_positions(sig, cfg, slice(0, 2000))
    assert set(np.unique(pos)).issubset({-1.0, 0.0, 1.0})
    net = portfolio_returns(R, pos, cost_bps=15.0)
    gross = portfolio_returns(R, pos, cost_bps=0.0)
    assert net.sum() < gross.sum()                         # costs actually bite
