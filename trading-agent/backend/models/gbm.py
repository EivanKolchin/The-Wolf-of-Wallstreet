"""QuantileGBM — the LightGBM co-model of the hybrid.

Gradient boosting is the right baseline at this data scale and signal-to-noise (the owner's
own probe: logistic ≥ GBM ≥ MLP ≥ LSTM). It is also SCALE-INVARIANT — it needs no
normalisation, so it is immune to any residual feature-scaling skew, making it a robustness
anchor for the hybrid. It trains ~100× faster than the TCN, so it drives the fast feature
prune loop, and its importances are read directly.

One LightGBM regressor per quantile (objective='quantile', alpha=q) → (N, Q) predictions,
post-sorted so quantiles never cross. Operates on the tabular feature SNAPSHOT (the latest
bar of each window) rather than the full sequence.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

try:
    import lightgbm as lgb
    HAS_LGB = True
except Exception:  # pragma: no cover - guarded so the package imports without lightgbm
    HAS_LGB = False


class QuantileGBM:
    def __init__(self, quantiles=(0.1, 0.5, 0.9), n_estimators: int = 300,
                 learning_rate: float = 0.03, num_leaves: int = 31,
                 min_child_samples: int = 200, subsample: float = 0.8,
                 colsample_bytree: float = 0.8, reg_lambda: float = 1.0, random_state: int = 0):
        if not HAS_LGB:
            raise ImportError("lightgbm is required for QuantileGBM (pip install lightgbm)")
        self.quantiles = tuple(quantiles)
        self.n_estimators = int(n_estimators)
        # Native LightGBM params (avoids the sklearn wrapper, so no scikit-learn dependency).
        self.params = dict(
            objective="quantile", learning_rate=learning_rate, num_leaves=num_leaves,
            min_data_in_leaf=min_child_samples, bagging_fraction=subsample, bagging_freq=1,
            feature_fraction=colsample_bytree, lambda_l2=reg_lambda, seed=random_state,
            verbose=-1,
        )
        self.models: list = []

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None):
        """Train one quantile booster per level. Rows with non-finite y are dropped."""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        m = np.isfinite(y)
        X, y = X[m], y[m]
        sw = None if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)[m]
        self.models = []
        for q in self.quantiles:
            params = dict(self.params, alpha=float(q))
            ds = lgb.Dataset(X, label=y, weight=sw, free_raw_data=False)
            booster = lgb.train(params, ds, num_boost_round=self.n_estimators)
            self.models.append(booster)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """(N, Q) quantile predictions, sorted ascending so quantiles never cross."""
        if not self.models:
            raise RuntimeError("QuantileGBM.predict called before fit")
        X = np.asarray(X, dtype=np.float64)
        preds = np.column_stack([m.predict(X) for m in self.models])   # (N, Q)
        return np.sort(preds, axis=1)

    def edge_and_uncertainty(self, X: np.ndarray):
        """(edge=median quantile, uncertainty=highest-lowest) per row."""
        p = self.predict(X)
        med = p[:, p.shape[1] // 2]
        return med, p[:, -1] - p[:, 0]

    def feature_importance(self) -> np.ndarray:
        """Mean gain importance across the quantile boosters (drives the prune loop)."""
        if not self.models:
            raise RuntimeError("no fitted models")
        return np.mean([m.feature_importance(importance_type="gain") for m in self.models], axis=0)

    def save(self, path: str):
        """Persist via LightGBM's native model strings (no joblib/sklearn)."""
        import json
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        blob = {"quantiles": list(self.quantiles),
                "models": [m.model_to_string() for m in self.models]}
        Path(path).write_text(json.dumps(blob), encoding="utf-8")

    @classmethod
    def load(cls, path: str) -> "QuantileGBM":
        import json
        blob = json.loads(Path(path).read_text(encoding="utf-8"))
        obj = cls(quantiles=tuple(blob["quantiles"]))
        obj.models = [lgb.Booster(model_str=s) for s in blob["models"]]
        return obj
