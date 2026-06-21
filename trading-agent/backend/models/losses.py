"""Quantile (pinball) loss for the hybrid's quantile heads.

Predicting quantiles {p10, p50, p90} of the vol-normalized forward return gives the model a
calibrated edge (p50) AND an uncertainty band (p90 - p10) in one shot — no MC-dropout needed.
The pinball loss is the proper scoring rule for quantile regression.
"""
from __future__ import annotations

import torch


def pinball_loss(preds: torch.Tensor, target: torch.Tensor, quantiles) -> torch.Tensor:
    """Mean pinball loss.

    preds   : (B, Q) predicted quantiles
    target  : (B,) regression target
    quantiles : iterable of Q levels in (0, 1), e.g. (0.1, 0.5, 0.9)

    For quantile q and error e = target - pred: loss = max(q*e, (q-1)*e). Under-prediction
    is penalised more for high q (and over-prediction more for low q), so the heads spread
    into a calibrated interval around the median.
    """
    q = torch.as_tensor(list(quantiles), dtype=preds.dtype, device=preds.device).view(1, -1)
    e = target.view(-1, 1) - preds                       # (B, Q)
    return torch.maximum(q * e, (q - 1.0) * e).mean()
