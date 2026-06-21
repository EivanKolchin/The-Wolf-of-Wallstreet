"""QuantileStacker — blends the TCN and GBM quantile predictions into one calibrated edge.

Stacking out-of-fold predictions of diverse models (a neural sequence model + a tabular
GBM) is a robust, low-variance way to combine them: the blend learns how much to trust each
model per quantile from held-out data, rather than a hand-picked weight. The default is a
simple non-negative least-squares blend on the median (with an equal-weight fallback), which
is hard to overfit and trivially interpretable.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np


class QuantileStacker:
    def __init__(self, n_models: int, quantiles=(0.1, 0.5, 0.9)):
        self.n_models = int(n_models)
        self.quantiles = tuple(quantiles)
        # Per-model weights (shared across quantiles). Start equal; fit() refines on the median.
        self.weights = np.full(self.n_models, 1.0 / self.n_models, dtype=np.float64)

    @staticmethod
    def _stack(preds_list: List[np.ndarray]) -> np.ndarray:
        """(M, N, Q) from a list of M (N, Q) per-model predictions."""
        arr = np.stack([np.asarray(p, dtype=np.float64) for p in preds_list], axis=0)
        if arr.ndim != 3:
            raise ValueError("each model prediction must be (N, Q)")
        return arr

    def fit(self, preds_list: List[np.ndarray], y: np.ndarray) -> "QuantileStacker":
        """Solve non-negative weights so the blended MEDIAN best fits y (out-of-fold preds).
        Falls back to equal weights if degenerate. Weights are normalised to sum to 1."""
        arr = self._stack(preds_list)                       # (M, N, Q)
        if arr.shape[0] != self.n_models:
            raise ValueError(f"expected {self.n_models} models, got {arr.shape[0]}")
        med_idx = arr.shape[2] // 2
        M = arr[:, :, med_idx].T                            # (N, M) median preds per model
        y = np.asarray(y, dtype=np.float64)
        mask = np.isfinite(y) & np.isfinite(M).all(axis=1)
        if mask.sum() >= self.n_models + 1:
            try:
                w, *_ = np.linalg.lstsq(M[mask], y[mask], rcond=None)
                w = np.clip(w, 0.0, None)                    # non-negative
                if w.sum() > 1e-9:
                    self.weights = w / w.sum()
            except Exception:
                pass
        return self

    def predict(self, preds_list: List[np.ndarray]) -> np.ndarray:
        """Weighted blend → (N, Q), re-sorted so blended quantiles never cross."""
        arr = self._stack(preds_list)                       # (M, N, Q)
        w = self.weights.reshape(-1, 1, 1)
        blended = (arr * w).sum(axis=0)                     # (N, Q)
        return np.sort(blended, axis=1)

    def edge_and_uncertainty(self, preds_list: List[np.ndarray]):
        p = self.predict(preds_list)
        return p[:, p.shape[1] // 2], p[:, -1] - p[:, 0]
