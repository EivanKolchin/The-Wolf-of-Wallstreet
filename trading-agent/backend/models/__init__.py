"""Phase 2 hybrid model components.

  • targets.vol_normalized_forward_return — the regression target (edge in vol units).
  • losses.pinball_loss               — quantile-regression loss.
  • sequence.QuantileTCN              — neural headline model (causal TCN + quantile heads).
  • gbm.QuantileGBM                   — LightGBM co-model (scale-invariant, fast).
  • ensemble.QuantileStacker          — blends TCN + GBM into one calibrated edge.

Every model predicts {p10, p50, p90} of the vol-normalized forward return: edge = p50,
uncertainty = p90 - p10. Nothing is promoted unless it beats the incumbent on walk-forward
net alpha (scripts/evaluate.py).
"""
from backend.models.targets import vol_normalized_forward_return, rolling_bar_vol  # noqa: F401
from backend.models.losses import pinball_loss  # noqa: F401
from backend.models.sequence import QuantileTCN  # noqa: F401
from backend.models.ensemble import QuantileStacker  # noqa: F401

try:  # GBM is optional (needs lightgbm); the rest of the package imports without it
    from backend.models.gbm import QuantileGBM, HAS_LGB  # noqa: F401
except Exception:  # pragma: no cover
    HAS_LGB = False
