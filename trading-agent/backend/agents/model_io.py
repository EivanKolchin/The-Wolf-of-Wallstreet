"""Checkpoint save/load with HARD schema validation.

The previous loader (``PersistentTradingModel._load_or_initialise``) silently did
a partial ``load_state_dict(strict=False)`` on shape mismatch, which masked the
fact that offline-trained weights had a different architecture/feature layout
than the live model. This module refuses to load a checkpoint whose embedded
feature schema does not match the running code, so a mismatch surfaces loudly
(and the caller can choose to cold-start instead of silently corrupting state).
"""
from __future__ import annotations

from pathlib import Path

import torch

try:  # works whether imported as backend.agents.model_io or with backend/ on path
    from signals import feature_spec as fs
except ImportError:  # pragma: no cover
    from backend.signals import feature_spec as fs


class CheckpointSchemaMismatch(Exception):
    """Raised when a checkpoint's embedded schema does not match the running code."""


def save_checkpoint(
    path,
    *,
    model,
    optimizer=None,
    scheduler=None,
    value_baseline=None,
    trade_count: int = 0,
    cumulative_pnl: float = 0.0,
    extra_meta: dict | None = None,
) -> dict:
    """Atomically save a checkpoint with an embedded FeatureSpec metadata block."""
    ckpt: dict = {
        "model_state_dict": model.state_dict(),
        "trade_count": int(trade_count),
        "cumulative_pnl": float(cumulative_pnl),
        "meta": fs.checkpoint_meta(**(extra_meta or {})),
    }
    if optimizer is not None:
        ckpt["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        ckpt["scheduler_state_dict"] = scheduler.state_dict()
    if value_baseline is not None:
        ckpt["value_baseline_state_dict"] = value_baseline.state_dict()

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(ckpt, tmp)
    tmp.replace(path)
    return ckpt


def load_checkpoint(
    path,
    *,
    expected_input_size: int = fs.INPUT,
    expected_version: str = fs.VERSION,
    strict_version: bool = True,
) -> dict:
    """Load a checkpoint, raising CheckpointSchemaMismatch on any schema drift."""
    ckpt = torch.load(path, weights_only=False)
    meta = ckpt.get("meta", {}) if isinstance(ckpt, dict) else {}
    in_size = meta.get("input_size")
    version = meta.get("feature_version")

    if in_size is None or version is None:
        raise CheckpointSchemaMismatch(
            f"checkpoint '{path}' has no FeatureSpec metadata (legacy / incompatible)"
        )
    if in_size != expected_input_size:
        raise CheckpointSchemaMismatch(
            f"checkpoint input_size {in_size} != expected {expected_input_size}"
        )
    if strict_version and version != expected_version:
        raise CheckpointSchemaMismatch(
            f"checkpoint feature_version '{version}' != expected '{expected_version}'"
        )
    return ckpt
