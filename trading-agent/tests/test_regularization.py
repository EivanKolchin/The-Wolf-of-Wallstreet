"""Phase 13 anti-overfitting tests: walk-forward CV + EarlyStopping + oos_gate."""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from backend.training.regularization import walk_forward_splits, EarlyStopping, oos_gate


# ----------------------------------------------------- walk-forward CV
def test_walk_forward_splits_have_no_future_leakage():
    splits = walk_forward_splits(n_samples=1000, n_folds=5, train_frac=0.6)
    assert len(splits) == 5
    for tr, val in splits:
        # training set must end strictly before the validation window starts
        assert max(tr) < min(val)
        # validation windows are forward-only chunks
        assert val.start >= 600                # past the first 60% train block


def test_walk_forward_splits_small_returns_empty():
    assert walk_forward_splits(n_samples=5) == []
    assert walk_forward_splits(n_samples=100, n_folds=0) == []


# ----------------------------------------------------- early stopping
def test_early_stopping_stops_after_patience_no_improvement():
    es = EarlyStopping(patience=3, mode="min")
    # first call establishes baseline
    assert es.step(1.0) is False
    # 3 worse-or-equal scores -> stop on the 3rd
    assert es.step(1.0) is False
    assert es.step(1.1) is False
    assert es.step(1.2) is True
    assert es.best_score == 1.0


def test_early_stopping_resets_on_improvement():
    es = EarlyStopping(patience=2, mode="min")
    es.step(2.0)
    es.step(2.0)            # no improvement -> bad_count=1
    es.step(1.5)            # improvement -> bad_count=0, best=1.5
    assert es.best_score == 1.5
    assert es.bad_count == 0


def test_early_stopping_max_mode():
    es = EarlyStopping(patience=2, mode="max")
    es.step(0.5)
    es.step(0.6)            # improvement
    es.step(0.6)            # bad=1
    assert es.step(0.5) is True   # bad=2 -> stop
    assert es.best_score == 0.6


# ----------------------------------------------------- OOS gate
def test_oos_gate_flags_overfit_loss_mode():
    # val loss 2x worse than train -> 100% gap -> flagged
    assert oos_gate(train_metric=0.1, val_metric=0.3, max_gap_pct=0.5, mode="min") is True


def test_oos_gate_passes_well_aligned_loss():
    assert oos_gate(train_metric=0.1, val_metric=0.11, max_gap_pct=0.20, mode="min") is False


def test_oos_gate_flags_overfit_accuracy_mode():
    assert oos_gate(train_metric=0.95, val_metric=0.60, max_gap_pct=0.20, mode="max") is True
