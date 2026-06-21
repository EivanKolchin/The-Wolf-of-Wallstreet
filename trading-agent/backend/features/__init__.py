"""Canonical feature engineering — the SINGLE source of truth shared by the
offline trainer (``scripts/pretrain.py``), the backtest, and the live agent.

Historically the offline pipeline and the live pipeline were two independent
implementations that drifted (different formulas/scales for ~15 features, and a
1000-bar rolling z-score applied offline but never live). This package exists so
there is exactly ONE feature code path; train == serve by construction.
"""
from backend.features.pipeline import (  # noqa: F401
    HAS_PANDAS_TA,
    BASE_FEATURES,
    HTF_FEATURES,
    ZSCORE_WIN,
    _TA,
    ta,
    _safe,
    build_feature_matrix,
    detect_regime,
    build_htf_features,
    apply_rolling_zscore,
    assemble_matrix,
    momentum_features,
    build_hybrid_matrix,
    MOMENTUM_NAMES,
    HYBRID_FEATURE_NAMES,
)
from backend.features.store import FeatureStore  # noqa: F401
