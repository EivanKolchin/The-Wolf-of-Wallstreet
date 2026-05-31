"""Phase 17 tests — ATR-multiple exit heads.

The exit heads now emit ATR multiples (regime-invariant). The agent converts to
fractions at inference time via the live `atr_norm` slot. Same trained weights
must produce proportionally different *absolute* stop distances under low-vol vs
high-vol regimes — the core robustness property for the stock universe (AMD vs
BE) and crypto regimes alike.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from signals import feature_spec as fs  # noqa: E402
from agents.improved_model import (  # noqa: E402
    ImprovedTradingLSTM, SYMBOL_TO_ID,
    SL_ATR_MULT_RANGE, TP_ATR_MULT_RANGE, TRAIL_ATR_MULT_RANGE,
    SL_FRAC_RANGE, TP_FRAC_RANGE, TRAIL_FRAC_RANGE,
)


def test_feature_schema_version_bumped_to_v2_1():
    # Phase 3 bumped the schema to v2.2 (added the 16-dim NEWS_EMBED block).
    assert fs.VERSION == "v2.2", f"expected v2.2 after Phase 3 news-embed bump, got {fs.VERSION}"


def test_forward_emits_both_legacy_fraction_and_atr_mult_keys():
    model = ImprovedTradingLSTM()
    x = torch.randn(2, 60, fs.INPUT)
    sids = torch.tensor([0, 0], dtype=torch.long)
    _, _, _, exits, _ = model(x, sids)
    for k in ("sl", "tp", "trail", "sl_mult", "tp_mult", "trail_mult"):
        assert k in exits, f"missing exit head output '{k}'"
        assert exits[k].shape == (2, 1)


def test_atr_mult_outputs_land_inside_configured_ranges():
    model = ImprovedTradingLSTM()
    x = torch.randn(8, 60, fs.INPUT)
    sids = torch.zeros(8, dtype=torch.long)
    _, _, _, exits, _ = model(x, sids)
    for k, (lo, hi) in (
        ("sl_mult", SL_ATR_MULT_RANGE),
        ("tp_mult", TP_ATR_MULT_RANGE),
        ("trail_mult", TRAIL_ATR_MULT_RANGE),
    ):
        v = exits[k]
        assert torch.all(v >= lo - 1e-6) and torch.all(v <= hi + 1e-6), \
            f"{k} produced values outside {(lo, hi)}: min={float(v.min()):.4f} max={float(v.max()):.4f}"


def test_legacy_fraction_outputs_still_inside_v2_0_ranges():
    """Back-compat: the v2.0 fractional ranges remain valid for downstream code
    that hasn't been switched to mult yet (e.g. the AWR offline loss in
    `_awr_update` regresses against legacy `sl_taken`/`tp_taken` fractions)."""
    model = ImprovedTradingLSTM()
    x = torch.randn(8, 60, fs.INPUT)
    sids = torch.zeros(8, dtype=torch.long)
    _, _, _, exits, _ = model(x, sids)
    for k, (lo, hi) in (
        ("sl", SL_FRAC_RANGE), ("tp", TP_FRAC_RANGE), ("trail", TRAIL_FRAC_RANGE),
    ):
        v = exits[k]
        assert torch.all(v >= lo - 1e-6) and torch.all(v <= hi + 1e-6), \
            f"legacy {k} drifted outside v2.0 range {(lo, hi)}"


def test_same_model_produces_proportional_sl_in_low_vs_high_vol(tmp_path, monkeypatch):
    """The killer test — feed the SAME model two sequences that differ ONLY in
    the `atr_norm` slot; the resulting fractional SL should scale linearly with
    ATR (because `sl_frac = sl_mult * atr_pct`)."""
    from agents.nn_model import PersistentTradingModel

    monkeypatch.setattr(PersistentTradingModel, "MODEL_PATH", tmp_path / "m.pt")
    monkeypatch.setattr(PersistentTradingModel, "CHECKPOINT_DIR", tmp_path / "ck")
    pm = PersistentTradingModel()

    rng = np.random.default_rng(0)
    seq = rng.standard_normal((pm.SEQUENCE_LENGTH, fs.INPUT)).astype(np.float32)

    seq_low = seq.copy()
    seq_low[-1, fs.VOLATILITY.start] = 0.04   # atr_norm = 0.04 → atr_pct = 0.4%
    res_low = pm.infer(seq_low, symbol_id=SYMBOL_TO_ID["BTCUSDT"], mc_samples=1)

    seq_high = seq.copy()
    seq_high[-1, fs.VOLATILITY.start] = 0.40  # atr_norm = 0.40 → atr_pct = 4.0%
    res_high = pm.infer(seq_high, symbol_id=SYMBOL_TO_ID["BTCUSDT"], mc_samples=1)

    # Only thing that differs between the two infer() calls is the last-row
    # atr_norm slot, which scales the mult → fraction conversion 10×.
    # The SL/TP/TRAIL fractions must scale roughly 10× too.
    ratio_sl = res_high.sl / max(res_low.sl, 1e-9)
    ratio_tp = res_high.tp / max(res_low.tp, 1e-9)
    ratio_tr = res_high.trail / max(res_low.trail, 1e-9)
    # The model's mult output is virtually identical (sequences only differ in
    # one cell of the last row), so the ratio should be ~10. Allow 8..12 to
    # accommodate any minor weight-dropout drift.
    assert 8.0 <= ratio_sl <= 12.0, f"sl ratio {ratio_sl:.2f} not in [8, 12]"
    assert 8.0 <= ratio_tp <= 12.0, f"tp ratio {ratio_tp:.2f} not in [8, 12]"
    assert 8.0 <= ratio_tr <= 12.0, f"trail ratio {ratio_tr:.2f} not in [8, 12]"
    # Absolute floors: high-vol stop should be substantially wider than low-vol.
    assert res_high.sl > res_low.sl
    assert res_high.tp > res_low.tp


def test_zero_atr_norm_falls_back_to_safe_floor(tmp_path, monkeypatch):
    """Degenerate sequence with `atr_norm = 0` should not produce 0 stops —
    nn_model.infer clamps to a 0.1% minimum."""
    from agents.nn_model import PersistentTradingModel

    monkeypatch.setattr(PersistentTradingModel, "MODEL_PATH", tmp_path / "m.pt")
    monkeypatch.setattr(PersistentTradingModel, "CHECKPOINT_DIR", tmp_path / "ck")
    pm = PersistentTradingModel()

    seq = np.zeros((pm.SEQUENCE_LENGTH, fs.INPUT), dtype=np.float32)
    res = pm.infer(seq, symbol_id=0, mc_samples=1)
    # 0.1% floor × min ATR mult (0.5) = 0.05% absolute → SL ≥ 0.0005, TP ≥ 0.0005
    assert res.sl > 0.0 and res.sl >= 0.5 * 0.001 - 1e-9
    assert res.tp > 0.0 and res.tp >= 0.5 * 0.001 - 1e-9
    assert res.trail > 0.0
