"""Lead-lag network estimation — the measured version of "Coca-Cola moves Pepsi".

The hypothesis (well-documented in the literature as slow information diffusion): a move in a
LEADER asset propagates to a FOLLOWER with a delay, so the leader's realized return predicts
the follower's NEXT-bar return. This module measures whether that effect (a) exists in-sample,
(b) survives out-of-sample, and (c) survives COSTS — per timescale, because the a-priori
expectation is that fast lead-lag (5m/15m) is arbitraged by HFT latency racers while slow
lead-lag (4h/daily) may persist.

Estimator (deliberately the simple, robust literature baseline — a learned Temporal-GNN
version is only justified if THIS finds anything):
    edge score(i→j) = corr( r_i[t] , r_j[t+1] )  computed on a TRAIN window,
    minus nothing — but the follower's own momentum is neutralised in the STRATEGY test by
    subtracting the follower's autocorrelation contribution (see leadlag_signal).

Strategy test: at each bar the follower's signal is the edge-weighted sum of its leaders'
last-bar returns; positions ∝ sign(signal) gated by |signal|; walk-forward folds re-estimate
edges on each fold's train slice only. Costs charged on turnover. Everything pure numpy —
unit-tested with planted synthetic edges."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class LeadLagConfig:
    min_abs_corr: float = 0.03      # edge admission threshold on the TRAIN window
    max_edges_per_follower: int = 5  # strongest leaders only (sparsity — avoids noise averaging)
    self_neutralise: bool = True    # subtract the follower's own lag-1 autocorr contribution
    signal_gate: float = 0.5        # trade only when |signal| > gate × its train std
    allow_short: bool = True


def _standardise(R: np.ndarray) -> np.ndarray:
    """Column-standardised returns (zero mean, unit std, computed per column). NaN-safe."""
    R = np.asarray(R, dtype=np.float64)
    mu = np.nanmean(R, axis=0, keepdims=True)
    sd = np.nanstd(R, axis=0, keepdims=True)
    sd = np.where(sd > 1e-12, sd, 1.0)
    Z = (R - mu) / sd
    return np.nan_to_num(Z, nan=0.0)


def leadlag_matrix(R: np.ndarray, lag: int = 1) -> np.ndarray:
    """(N, N) directed predictive-correlation matrix on a returns panel R (T, N):

        L[i, j] = corr( r_i[t], r_j[t+lag] )        — i LEADS j.

    Diagonal = each asset's own lag-``lag`` autocorrelation (kept: it's the baseline any
    cross-edge must beat). Vectorised: one matrix product on standardised panels."""
    Z = _standardise(R)
    T = Z.shape[0]
    if T <= lag + 2:
        return np.zeros((Z.shape[1], Z.shape[1]))
    A = Z[:-lag]            # leaders at t
    B = Z[lag:]             # followers at t+lag
    n = A.shape[0]
    return (A.T @ B) / n    # (N, N)


def build_edges(L: np.ndarray, cfg: LeadLagConfig) -> List[Tuple[int, int, float]]:
    """[(leader, follower, weight)] — per follower, the strongest |corr| off-diagonal edges
    above the admission threshold, at most ``max_edges_per_follower``."""
    N = L.shape[0]
    edges: List[Tuple[int, int, float]] = []
    for j in range(N):
        col = L[:, j].copy()
        col[j] = 0.0                                     # self handled separately
        order = np.argsort(-np.abs(col))
        for i in order[: cfg.max_edges_per_follower]:
            w = float(col[i])
            if abs(w) >= cfg.min_abs_corr:
                edges.append((int(i), int(j), w))
    return edges


def leadlag_signal(R: np.ndarray, edges: List[Tuple[int, int, float]],
                   L_train: np.ndarray, cfg: LeadLagConfig) -> np.ndarray:
    """(T, N) follower signals: edge-weighted sum of leaders' CURRENT-bar standardised returns
    (predicting each follower's NEXT bar). With ``self_neutralise`` the follower's own lag-1
    autocorr times its own return is SUBTRACTED, so the signal is the CROSS-asset increment —
    otherwise "lead-lag" silently degenerates into single-asset momentum."""
    Z = _standardise(R)
    T, N = Z.shape
    sig = np.zeros((T, N))
    for (i, j, w) in edges:
        sig[:, j] += w * Z[:, i]
    if cfg.self_neutralise:
        for j in range(N):
            sig[:, j] -= float(L_train[j, j]) * Z[:, j]
    return sig


def signal_positions(sig: np.ndarray, cfg: LeadLagConfig,
                     train_slice: slice) -> np.ndarray:
    """Map signals → positions in {-1, 0, +1} (or {0,1} long-only), gated so only signals
    larger than ``signal_gate`` × the TRAIN-window signal std trade (noise floor)."""
    thr = np.nanstd(sig[train_slice], axis=0, keepdims=True) * cfg.signal_gate
    thr = np.where(thr > 1e-12, thr, np.inf)
    pos = np.where(sig > thr, 1.0, np.where(sig < -thr, -1.0, 0.0))
    if not cfg.allow_short:
        pos = np.clip(pos, 0.0, 1.0)
    return pos


def portfolio_returns(R: np.ndarray, pos: np.ndarray, cost_bps: float) -> np.ndarray:
    """Equal-dollar per-bar net book returns: position[t] earns R[t+1]; costs on turnover.
    Bars where nothing trades contribute 0 (cash)."""
    T, N = R.shape
    cost = cost_bps / 1e4
    Rn = np.nan_to_num(np.asarray(R, dtype=np.float64), nan=0.0)
    gross = np.zeros(T)
    gross[1:] = np.mean(pos[:-1] * Rn[1:], axis=1)
    turn = np.zeros(T)
    turn[0] = np.mean(np.abs(pos[0]))
    turn[1:] = np.mean(np.abs(pos[1:] - pos[:-1]), axis=1)
    return gross - turn * cost


@dataclass
class FoldResult:
    ic_insample: float          # mean |edge corr| admitted on train
    ic_oos: float               # same edges' mean signed corr realized on test
    sharpe_gross: float
    sharpe_net: float
    n_edges: int
    turnover: float


def walk_forward(R: np.ndarray, *, folds: int = 4, cost_bps: float = 15.0,
                 bars_per_year: float = 105_120.0,
                 cfg: Optional[LeadLagConfig] = None) -> List[FoldResult]:
    """K chronological folds: estimate the lead-lag network on each fold's TRAIN prefix,
    trade the follower book on the fold's TEST slice. No edge ever sees its test data."""
    cfg = cfg or LeadLagConfig()
    R = np.asarray(R, dtype=np.float64)
    T = R.shape[0]
    out: List[FoldResult] = []
    fold_len = T // (folds + 1)
    ann = float(np.sqrt(bars_per_year))
    for k in range(folds):
        tr_end = fold_len * (k + 1)
        te_end = min(T, fold_len * (k + 2))
        if te_end - tr_end < 50 or tr_end < 100:
            continue
        L = leadlag_matrix(R[:tr_end])
        edges = build_edges(L, cfg)
        sig = leadlag_signal(R, edges, L, cfg)
        pos = signal_positions(sig, cfg, slice(0, tr_end))
        pos[:tr_end] = 0.0                                # trade the test slice only
        pos[te_end:] = 0.0
        net = portfolio_returns(R, pos, cost_bps)[tr_end:te_end]
        gross = portfolio_returns(R, pos, 0.0)[tr_end:te_end]

        # OOS edge validity: do the SAME train-admitted edges still predict on test?
        Lte = leadlag_matrix(R[tr_end:te_end])
        ic_oos = float(np.mean([np.sign(w) * Lte[i, j] for (i, j, w) in edges])) if edges else 0.0
        ic_ins = float(np.mean([abs(w) for (_, _, w) in edges])) if edges else 0.0

        def _sh(x):
            sd = x.std()
            return float(x.mean() / sd * ann) if sd > 1e-12 else 0.0
        turn = float(np.mean(np.abs(np.diff(pos[tr_end:te_end], axis=0))))
        out.append(FoldResult(ic_insample=ic_ins, ic_oos=ic_oos,
                              sharpe_gross=_sh(gross), sharpe_net=_sh(net),
                              n_edges=len(edges), turnover=turn))
    return out
