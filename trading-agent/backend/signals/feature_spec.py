"""Canonical feature layout for the trading model — the SINGLE SOURCE OF TRUTH.

Both the live feature builder (``backend/signals/features.py``) and the offline
pretraining pipeline (``scripts/pretrain.py``) import these constants so the two
can never drift out of alignment again. Previously the live builder used an
8-slot orderbook block + 6-class regime one-hot at index 43, while the offline
builder used a 10-slot orderbook block + 4-class regime one-hot at index 45 —
which silently made offline-trained weights invalid against live vectors.

BASE layout (62 indices, 0..61):
  0-2    price action: body, upper wick, lower wick
  3      volume ratio (current / 20-bar MA)
  4      spread
  5-10   moving averages: ema9/21/50/200 dist, golden_cross, vwap_dist
  11-16  momentum: rsi, macd_norm, macd_hist_norm, stoch_rsi, adx_norm, rsi_divergence
  17-19  volatility: atr_norm, bb_width_norm, bb_pct_b
  20-21  volume: volume_ratio(dup/momentum), obv_slope
  22-24  fibonacci: nearest_level_pct, distance, strength
  25-34  candlestick pattern flags (10)
  35-42  orderbook microstructure (8): book_imbalance, bid_depth_ratio,
         ask_depth_ratio, cvd_slope, cvd_divergence, whale_activity,
         bullish_sweep, bearish_sweep
  43-48  regime one-hot (6): uptrend, downtrend, ranging, high_volatility,
         news_driven, low_liquidity
  49-52  news: direction, magnitude, confidence, age
  53-56  macro: fear_greed, btc_dominance, funding_rate, oi_change
  57-60  time cyclical: sin/cos hour, sin/cos weekday
  61     regime confidence

HTF block (8 indices, appended at 62..69 by the agent / offline pipeline):
  62-65  1h: rsi_norm, ema21_dist, macd_hist_norm, atr_norm
  66-69  4h: rsi_norm, ema21_dist, trend_dir, atr_norm

NEWS_EMBED block (16 indices, appended at 70..85 by the agent / offline pipeline):
  70-85  learned/semantic news-text embedding (sentence-transformer or a
         deterministic hashing fallback), projected to NEWS_EMBED_DIM dims.
         Distinct from the 4 scalar news slots (49-52): those encode the LLM's
         structured direction/magnitude/confidence/age; this block carries the
         *semantic content* of the news so the NN can learn from what the news
         actually says, not just 4 hand-rolled scalars.

EARNINGS block (4 indices, appended at 86..89 by the agent / offline pipeline):
  86-89  earnings-calendar features (stocks; zeros for crypto): time-to-next-earnings
         proximity, pre-earnings flag, post-earnings drift proximity, last EPS surprise.
         Leakage-safe — see backend/signals/earnings.py.

INPUT = BASE + HTF + NEWS_EMBED + EARNINGS = 90  (the model's input_size)
"""
from __future__ import annotations

import numpy as np

VERSION = "v2.3"
# v2.0 → v2.1 (Phase 17): exit heads now emit ATR multiples, not raw fractions.
# v2.1 → v2.2 (Phase 3 news-embeddings): appended a 16-dim NEWS_EMBED block at
# 70..85, growing INPUT 70 → 86.
# v2.2 → v2.3 (Cycle 7 earnings): appended a 4-dim EARNINGS block at 86..89
# (time-to-next-earnings + pre-earnings flag + post-earnings drift + last surprise),
# growing INPUT 86 → 90. Older checkpoints cold-start (the model is being retrained).

BASE = 62
HTF = 8
NEWS_EMBED_DIM = 16
EARNINGS_DIM = 4
INPUT = BASE + HTF + NEWS_EMBED_DIM + EARNINGS_DIM  # 90

# --- single-index features ---
VOLUME_RATIO = 3
SPREAD = 4
REGIME_CONFIDENCE = 61

# --- contiguous regions over the BASE vector (start inclusive, stop exclusive) ---
PRICE = slice(0, 3)
MA = slice(5, 11)
MOMENTUM = slice(11, 17)
VOLATILITY = slice(17, 20)
VOLUME = slice(20, 22)
FIBONACCI = slice(22, 25)
PATTERNS = slice(25, 35)
ORDERBOOK = slice(35, 43)   # 8 slots
REGIME = slice(43, 49)      # 6-class one-hot
NEWS = slice(49, 53)
MACRO = slice(53, 57)
TIME = slice(57, 61)

ORDERBOOK_SLOTS = ORDERBOOK.stop - ORDERBOOK.start  # 8
REGIME_LABELS = [
    "uptrend", "downtrend", "ranging", "high_volatility", "news_driven", "low_liquidity",
]
REGIME_START = REGIME.start  # 43

# HTF sub-layout (relative to the full INPUT vector)
HTF_START = BASE          # 62
HTF_END = BASE + HTF      # 70  (HTF occupies [HTF_START:HTF_END])

# NEWS_EMBED sub-layout (appended after HTF, relative to the full INPUT vector)
NEWS_EMBED_START = HTF_END                            # 70
NEWS_EMBED_END = NEWS_EMBED_START + NEWS_EMBED_DIM     # 86
NEWS_EMBED = slice(NEWS_EMBED_START, NEWS_EMBED_END)   # [70:86] — 16-dim semantic news embedding

# EARNINGS sub-layout (appended after NEWS_EMBED, relative to the full INPUT vector)
#   86  time-to-next-earnings proximity  exp(-days_to_next/τ)   (anticipatory — scheduled date,
#                                                                known ahead, so no leakage)
#   87  pre-earnings window flag         1.0 within PRE_DAYS before the next report
#   88  post-earnings drift proximity    exp(-days_since_last/τ) (recently reported)
#   89  last earnings surprise           clipped EPS surprise, decayed since release
EARNINGS_START = NEWS_EMBED_END                       # 86
EARNINGS = slice(EARNINGS_START, INPUT)               # [86:90] — earnings-calendar features


def regime_index(name: str):
    """Absolute index of a regime label in the BASE vector, or None if unknown."""
    try:
        return REGIME_START + REGIME_LABELS.index(name)
    except ValueError:
        return None


def regime_onehot(name: str) -> list[float]:
    return [1.0 if r == name else 0.0 for r in REGIME_LABELS]


def validate(vec: np.ndarray, *, allow_htf: bool = False) -> None:
    """Raise ValueError if ``vec`` violates the canonical layout."""
    expected = INPUT if allow_htf else BASE
    if vec.shape[-1] != expected:
        raise ValueError(f"feature vector length {vec.shape[-1]} != expected {expected}")
    regime_sum = float(np.sum(vec[..., REGIME]))
    # one-hot is either all-zero (regime not yet set) or sums to ~1
    if regime_sum > 0 and not np.isclose(regime_sum, 1.0, atol=1e-4):
        raise ValueError(f"regime one-hot must sum to 0 or 1, got {regime_sum}")


def checkpoint_meta(**extra) -> dict:
    """Metadata block embedded in checkpoints so the loader can hard-validate."""
    meta = {
        "feature_version": VERSION,
        "input_size": INPUT,
        "base_features": BASE,
        "htf_features": HTF,
        "news_embed_dim": NEWS_EMBED_DIM,
        "earnings_dim": EARNINGS_DIM,
        "regime_labels": list(REGIME_LABELS),
        "orderbook_slots": ORDERBOOK_SLOTS,
    }
    meta.update(extra)
    return meta


def _self_check() -> None:
    """Fail fast on import if the regions don't tile 0..BASE-1 exactly."""
    covered: list[int] = [VOLUME_RATIO, SPREAD, REGIME_CONFIDENCE]
    for sl in (PRICE, MA, MOMENTUM, VOLATILITY, VOLUME, FIBONACCI,
               PATTERNS, ORDERBOOK, REGIME, NEWS, MACRO, TIME):
        covered.extend(range(sl.start, sl.stop))
    if sorted(covered) != list(range(BASE)):
        raise AssertionError("FeatureSpec regions do not tile 0..BASE-1 exactly")
    if len(REGIME_LABELS) != (REGIME.stop - REGIME.start):
        raise AssertionError("REGIME_LABELS length must equal the REGIME slice width")


_self_check()
