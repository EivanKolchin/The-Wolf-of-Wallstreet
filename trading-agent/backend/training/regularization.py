"""Phase 13 — anti-overfitting helpers for the offline trainer.

* walk_forward_splits — forward-only CV splits (no future leakage).
* EarlyStopping — patience-based stop monitor (min or max mode).
* oos_gate — rejects candidates whose train/val metric gap is too large.

These are standalone helpers; the offline trainer (Phase 3 follow-up) plugs them
in so promoted checkpoints have to (a) keep improving on each held-out window,
and (b) not have ballooning train/val gaps.
"""
from __future__ import annotations

from dataclasses import dataclass


def walk_forward_splits(n_samples: int, n_folds: int = 5, train_frac: float = 0.6) -> list[tuple[range, range]]:
    """Generate (train_indices, val_indices) for forward-only walk-forward CV.

    Fold 0 trains on the first `train_frac * n_samples`, validates on the next
    `(1 - train_frac)/n_folds * n_samples` window. Each subsequent fold slides
    forward — training always uses everything BEFORE the val window (so the val
    window is strictly future-of-train, no leakage).
    """
    if n_samples < 10 or n_folds < 1:
        return []
    train_size = max(1, int(n_samples * float(train_frac)))
    if train_size >= n_samples:
        return []
    val_size = max(1, (n_samples - train_size) // int(n_folds))
    splits: list[tuple[range, range]] = []
    for k in range(n_folds):
        v_start = train_size + k * val_size
        v_end = min(v_start + val_size, n_samples)
        if v_start >= n_samples:
            break
        splits.append((range(0, v_start), range(v_start, v_end)))
    return splits


@dataclass
class EarlyStopping:
    """Patience-based stop monitor.

    `mode="min"` -> lower score is better (loss).
    `mode="max"` -> higher score is better (accuracy/expectancy).
    """
    patience: int = 5
    min_delta: float = 0.0
    mode: str = "min"
    best_score: float | None = None
    bad_count: int = 0

    def step(self, score: float) -> bool:
        """Feed a new validation score. Returns True when training should stop."""
        if self.best_score is None:
            self.best_score = float(score)
            self.bad_count = 0
            return False
        if self.mode == "min":
            improved = float(score) < self.best_score - self.min_delta
        else:
            improved = float(score) > self.best_score + self.min_delta
        if improved:
            self.best_score = float(score)
            self.bad_count = 0
            return False
        self.bad_count += 1
        return self.bad_count >= self.patience


def oos_gate(train_metric: float, val_metric: float, max_gap_pct: float = 0.20,
             mode: str = "min") -> bool:
    """Return True if the train/val gap exceeds `max_gap_pct` (overfit flag).

    mode='min' (loss-like): overfit when val is MUCH WORSE than train.
        gap = (val - train) / |train|.
    mode='max' (accuracy-like): overfit when train >> val.
        gap = (train - val) / |train|.
    """
    denom = max(abs(float(train_metric)), 1e-9)
    if mode == "min":
        gap = (float(val_metric) - float(train_metric)) / denom
    else:
        gap = (float(train_metric) - float(val_metric)) / denom
    return gap > float(max_gap_pct)
