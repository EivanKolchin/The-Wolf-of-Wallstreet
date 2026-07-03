"""QuantileTCN overlay trainer — core-function tests on tiny synthetic data (no network,
no parquet caches, CPU-fast). The heavy end-to-end path is exercised by the script itself."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import train_quantile_tcn as tq  # noqa: E402


def _synthetic_dataset(n=1200, seed=0):
    """Feature matrix whose first column IS the (noisy) future vol-normalized return —
    a learnable planted signal — plus close prices to drive the A/B."""
    rng = np.random.default_rng(seed)
    y = rng.standard_normal((n, len(tq.HORIZONS))).astype(np.float32)
    X = rng.standard_normal((n, tq.FEATURE_DIM)).astype(np.float32) * 0.1
    X[:, 0] = y[:, 0] + 0.1 * rng.standard_normal(n)          # planted signal @ H idx 0
    X[:, 1] = y[:, 1] + 0.1 * rng.standard_normal(n)          # planted signal @ H idx 1
    close = 100.0 * np.cumprod(1.0 + 0.0005 + 0.01 * rng.standard_normal(n))
    ts = pd.date_range("2024-01-01", periods=n, freq="5min").to_numpy()
    return dict(X=X, y=y, close=close, high=close * 1.001, low=close * 0.999, timestamps=ts)


def test_window_ends_and_split_are_chronological():
    d = _synthetic_dataset()
    d["y"][-5:] = np.nan                                     # tail mask
    ends = tq.window_ends(len(d["X"]), d["y"], stride=4)
    assert ends[0] >= tq.SEQ_LEN
    assert np.all(np.diff(ends) > 0)
    assert ends[-1] <= len(d["X"]) - 5                       # NaN tail dropped
    tr, va, te = tq.split_ends(ends, embargo=3)
    assert len(tr) and len(va) and len(te)
    assert tr[-1] < va[0] <= va[-1] < te[0]                  # strictly chronological


def test_edge_score_rewards_planted_signal():
    rng = np.random.default_rng(1)
    fwd = rng.standard_normal(4000)
    good = tq.edge_score(fwd + 0.1 * rng.standard_normal(4000), np.ones(4000), fwd)
    noise = tq.edge_score(rng.standard_normal(4000), np.ones(4000), fwd)
    assert good > 1.0
    assert abs(noise) < good / 2


def test_gated_positions_never_originates_and_bounds():
    #                  agree   agree   flat   disagree  disagree(short)
    pos = np.array([1.0, -1.0, 0.0, 1.0, -1.0])
    edge = np.array([2.0, -2.0, 5.0, -2.0, 2.0])
    unc = np.ones(5)
    g = tq.gated_positions(pos, edge, unc, ir_cap=1.0, floor=0.25)
    assert g[2] == 0.0                                       # flat stays flat (no origination)
    assert np.isclose(g[0], 1.0) and np.isclose(g[1], -1.0)  # agreement → full size
    assert np.isclose(g[3], 0.25) and np.isclose(g[4], -0.25)  # disagreement → floor, not flip
    assert np.all(np.abs(g) <= np.abs(pos) + 1e-12)          # can only shrink, never grow


def test_train_one_seed_learns_planted_signal():
    """A few epochs on a planted-signal dataset must yield a clearly positive val edge
    score (the trainer's selection metric) — guards the loss/selection wiring AND the
    last-step skip in QuantileTCN (without it, a last-bar signal was unlearnable at T=60)."""
    datasets = {"BTCUSDT": _synthetic_dataset(n=6000)}
    model, score = tq.train_one_seed(datasets, seed=0, epochs=4, stride=2, hidden=32,
                                     lr=3e-3, batch=128, device=torch.device("cpu"),
                                     log=lambda *a, **k: None)
    assert np.isfinite(score)
    assert score > 1.0, f"planted signal not learned (score {score})"


def test_sharpe_objective_learns_planted_signal():
    """DMN mode: plant the stride-matched vol-normalized forward return into feature 0;
    a few epochs must produce a clearly positive val NET Sharpe."""
    from backend.models.targets import vol_normalized_forward_return
    rng = np.random.default_rng(3)
    n, stride = 6000, 8
    close = 100.0 * np.cumprod(1.0 + 0.01 * rng.standard_normal(n))
    vnr = vol_normalized_forward_return(close, h=stride)
    X = rng.standard_normal((n, tq.FEATURE_DIM)).astype(np.float32) * 0.1
    X[:, 0] = np.nan_to_num(vnr) + 0.1 * rng.standard_normal(n)
    d = dict(X=X, y=np.zeros((n, len(tq.HORIZONS)), np.float32), close=close,
             high=close, low=close,
             timestamps=pd.date_range("2024-01-01", periods=n, freq="5min").to_numpy())
    model, score = tq.train_one_seed_sharpe({"BTCUSDT": d}, seed=0, epochs=4, stride=stride,
                                            hidden=32, lr=3e-3, block=64,
                                            device=torch.device("cpu"),
                                            log=lambda *a, **k: None)
    assert np.isfinite(score)
    assert score > 1.0, f"sharpe objective did not learn (val net sharpe {score})"
    # the head must emit bounded positions
    p = model.forward_position(torch.from_numpy(np.stack([X[i:i + tq.SEQ_LEN] for i in range(4)])),
                               torch.zeros(4, dtype=torch.long))
    assert torch.all(p.abs() <= 1.0)


def test_run_ab_plumbing_end_to_end():
    """A/B runs end-to-end on synthetic data and returns both books' stats."""
    datasets = {"BTCUSDT": _synthetic_dataset(n=4000, seed=2)}
    model = tq.QuantileTCN(input_size=tq.FEATURE_DIM, hidden=16,
                           num_symbols=25, num_horizons=len(tq.HORIZONS),
                           quantiles=tq.QUANTILES)
    out = tq.run_ab(datasets, [model], torch.device("cpu"), log=lambda *a, **k: None)
    assert {"base", "overlay", "control", "timing_skill"} <= set(out)
    for k in ("sharpe", "cagr", "maxdd"):
        assert np.isfinite(out["base"][k]) and np.isfinite(out["overlay"][k]) \
            and np.isfinite(out["control"][k])
    # constant scaling ~preserves Sharpe: control must sit near the baseline
    assert abs(out["control"]["sharpe"] - out["base"]["sharpe"]) < 0.35
    assert np.isfinite(out["timing_skill"])
