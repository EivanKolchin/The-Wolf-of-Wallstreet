"""Phase 18 v1 tests — MC-dropout predictive distribution + payload shape.

The old `_background_predictions_loop` projected price as `velocity * 0.85^i`
(deterministic ramp, no model uncertainty). Phase 18 v1 replaces that with
real MC-dropout sampling of per-horizon edges. These tests guard:

  - `PersistentTradingModel.infer_predictive_distribution(K)` returns the
    documented `{horizons, edge_samples}` shape and is non-degenerate under K>1.
  - The visualization payload constructed from the distribution has
    `median_close`, `p25_close`, `p75_close` per step, the band is non-zero
    in width, and `high >= median >= low`.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from backend.signals import feature_spec as fs


# --------------------------------------------------------- model-level
def test_infer_predictive_distribution_shape_and_horizons(tmp_path, monkeypatch):
    from backend.agents.nn_model import PersistentTradingModel
    from backend.agents.improved_model import SYMBOL_TO_ID, HORIZONS

    monkeypatch.setattr(PersistentTradingModel, "MODEL_PATH", tmp_path / "model.pt")
    monkeypatch.setattr(PersistentTradingModel, "CHECKPOINT_DIR", tmp_path / "ckpts")
    pm = PersistentTradingModel()

    seq = np.random.randn(pm.SEQUENCE_LENGTH, fs.INPUT).astype(np.float32)
    K = 8
    dist = pm.infer_predictive_distribution(seq, symbol_id=SYMBOL_TO_ID["BTCUSDT"], mc_samples=K)
    assert "horizons" in dist and "edge_samples" in dist
    assert list(dist["horizons"]) == list(HORIZONS)
    assert dist["edge_samples"].shape == (K, len(HORIZONS))
    assert dist["edge_samples"].dtype == np.float32
    assert np.all(np.isfinite(dist["edge_samples"]))
    # Every edge sits inside [-1, 1] because it's p_long - p_short with probs in [0, 1].
    assert dist["edge_samples"].min() >= -1.0 - 1e-6
    assert dist["edge_samples"].max() <= 1.0 + 1e-6


def test_infer_predictive_distribution_is_stochastic_under_mc(tmp_path, monkeypatch):
    """K MC-dropout samples should NOT all be identical (dropout actually fires)."""
    from backend.agents.nn_model import PersistentTradingModel
    from backend.agents.improved_model import SYMBOL_TO_ID

    monkeypatch.setattr(PersistentTradingModel, "MODEL_PATH", tmp_path / "model.pt")
    monkeypatch.setattr(PersistentTradingModel, "CHECKPOINT_DIR", tmp_path / "ckpts")
    pm = PersistentTradingModel()

    seq = np.random.randn(pm.SEQUENCE_LENGTH, fs.INPUT).astype(np.float32)
    dist = pm.infer_predictive_distribution(seq, symbol_id=0, mc_samples=12)
    # Std across samples > 0 at the primary horizon: dropout produces real variation.
    stds = dist["edge_samples"].std(axis=0)
    assert stds[0] > 1e-4, f"expected nonzero MC stddev, got {stds.tolist()}"


def test_infer_predictive_distribution_k_equals_one_is_deterministic(tmp_path, monkeypatch):
    from backend.agents.nn_model import PersistentTradingModel

    monkeypatch.setattr(PersistentTradingModel, "MODEL_PATH", tmp_path / "model.pt")
    monkeypatch.setattr(PersistentTradingModel, "CHECKPOINT_DIR", tmp_path / "ckpts")
    pm = PersistentTradingModel()

    seq = np.random.randn(pm.SEQUENCE_LENGTH, fs.INPUT).astype(np.float32)
    dist_a = pm.infer_predictive_distribution(seq, symbol_id=0, mc_samples=1)
    dist_b = pm.infer_predictive_distribution(seq, symbol_id=0, mc_samples=1)
    # Without dropout on (K=1), two calls should be identical.
    np.testing.assert_allclose(dist_a["edge_samples"], dist_b["edge_samples"], atol=1e-6)


# --------------------------------------------------------- payload shape
def _build_payload(current_price: float, df: pd.DataFrame, dist: dict, chart_steps: int = 12) -> list:
    """Mirror the projection logic now in nn_agent._background_predictions_loop
    so the test can validate the shape without spinning up the agent."""
    horizons = dist["horizons"]
    edge_samples = dist["edge_samples"]
    try:
        rets = df["close"].pct_change().tail(60).dropna()
        realized_step_vol = float(rets.std())
        if not np.isfinite(realized_step_vol) or realized_step_vol < 1e-5:
            realized_step_vol = 0.005
    except Exception:
        realized_step_vol = 0.005

    predictions = []
    prev_close = float(current_price)
    for i in range(1, chart_steps + 1):
        closest_h_idx = int(np.argmin([abs(h - i) for h in horizons]))
        edges_at_step = edge_samples[:, closest_h_idx]
        mag_i = realized_step_vol * float(np.sqrt(i))
        sample_prices = current_price * (1.0 + edges_at_step * mag_i)
        median_close = float(np.median(sample_prices))
        p25_close = float(np.percentile(sample_prices, 25))
        p75_close = float(np.percentile(sample_prices, 75))
        predictions.append({
            "step": i,
            "open": prev_close, "close": median_close,
            "high": p75_close, "low": p25_close,
            "median_close": median_close,
            "p25_close": p25_close, "p75_close": p75_close,
        })
        prev_close = median_close
    return predictions


def test_payload_band_shape_and_invariants():
    # Synthetic K MC samples — strong positive edge with some spread.
    K = 16
    edge_samples = np.random.default_rng(42).normal(loc=0.4, scale=0.2, size=(K, 3)).astype(np.float32)
    dist = {"horizons": [3, 12, 48], "edge_samples": edge_samples}
    df = pd.DataFrame({"close": np.linspace(100.0, 102.0, 60)})
    payload = _build_payload(current_price=102.0, df=df, dist=dist)

    assert len(payload) == 12
    for p in payload:
        assert {"median_close", "p25_close", "p75_close", "high", "low", "open", "close"} <= set(p.keys())
        # band invariants
        assert p["low"] <= p["median_close"] <= p["high"]
        assert p["p25_close"] <= p["median_close"] <= p["p75_close"]
        assert p["high"] == p["p75_close"] and p["low"] == p["p25_close"]
        # band has non-zero width (dropout produced real spread)
        assert (p["high"] - p["low"]) > 0.0
    # Bands grow over the projection horizon (sqrt-time scaling).
    width_1 = payload[0]["high"] - payload[0]["low"]
    width_12 = payload[-1]["high"] - payload[-1]["low"]
    assert width_12 > width_1


def test_payload_median_drifts_in_direction_of_edge_sign():
    """Strongly positive edge → median predicted price climbs above entry."""
    K = 16
    edge_pos = np.full((K, 3), 0.5, dtype=np.float32)
    df = pd.DataFrame({"close": np.linspace(100.0, 100.5, 60)})
    payload_up = _build_payload(100.0, df, {"horizons": [3, 12, 48], "edge_samples": edge_pos})
    assert payload_up[-1]["median_close"] > 100.0

    edge_neg = np.full((K, 3), -0.5, dtype=np.float32)
    payload_dn = _build_payload(100.0, df, {"horizons": [3, 12, 48], "edge_samples": edge_neg})
    assert payload_dn[-1]["median_close"] < 100.0


def test_payload_uses_zero_volatility_fallback_when_returns_missing():
    """A degenerate (constant-price) df should not produce NaNs / divide-by-zero."""
    edge_samples = np.full((4, 3), 0.2, dtype=np.float32)
    df = pd.DataFrame({"close": np.full(60, 100.0)})
    payload = _build_payload(100.0, df, {"horizons": [3, 12, 48], "edge_samples": edge_samples})
    for p in payload:
        assert np.isfinite(p["median_close"])
        assert np.isfinite(p["high"]) and np.isfinite(p["low"])
