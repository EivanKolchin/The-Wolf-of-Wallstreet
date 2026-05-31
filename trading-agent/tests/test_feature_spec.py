"""Phase 0 guard tests: the canonical FeatureSpec must stay self-consistent so the
live builder and the offline pretraining pipeline can never drift apart again.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

# Mirror the live runtime: project root + backend/ both on the path so bare
# imports (``from signals import ...``) resolve exactly as they do in main.py.
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from signals import feature_spec as fs  # noqa: E402


def test_version_and_sizes():
    assert fs.VERSION == "v2.2"   # Phase 3 bump (16-dim NEWS_EMBED block)
    assert fs.BASE == 62
    assert fs.HTF == 8
    assert fs.NEWS_EMBED_DIM == 16
    assert fs.INPUT == 86


def test_news_embed_block_layout():
    # NEWS_EMBED occupies [70:86], immediately after the HTF block [62:70],
    # and must NOT overlap the 4 scalar news slots [49:53].
    assert fs.HTF_END == 70
    assert fs.NEWS_EMBED_START == 70
    assert fs.NEWS_EMBED.start == 70 and fs.NEWS_EMBED.stop == 86
    assert (fs.NEWS_EMBED.stop - fs.NEWS_EMBED.start) == fs.NEWS_EMBED_DIM
    assert fs.NEWS.stop <= fs.HTF_START  # scalar news slots are inside BASE


def test_regions_tile_base_exactly():
    covered = [fs.VOLUME_RATIO, fs.SPREAD, fs.REGIME_CONFIDENCE]
    for sl in (fs.PRICE, fs.MA, fs.MOMENTUM, fs.VOLATILITY, fs.VOLUME,
               fs.FIBONACCI, fs.PATTERNS, fs.ORDERBOOK, fs.REGIME,
               fs.NEWS, fs.MACRO, fs.TIME):
        covered.extend(range(sl.start, sl.stop))
    assert sorted(covered) == list(range(fs.BASE)), "regions must tile 0..BASE-1"


def test_orderbook_is_eight_slots():
    assert fs.ORDERBOOK_SLOTS == 8
    assert (fs.ORDERBOOK.stop - fs.ORDERBOOK.start) == 8


def test_regime_layout():
    assert len(fs.REGIME_LABELS) == (fs.REGIME.stop - fs.REGIME.start) == 6
    assert fs.regime_index("uptrend") == 43
    assert fs.regime_index("ranging") == 45
    assert fs.regime_index("low_liquidity") == 48
    assert fs.regime_index("does_not_exist") is None


def test_validate_accepts_base_and_onehot():
    v = np.zeros(fs.BASE, dtype=np.float32)
    fs.validate(v)  # all-zero regime is allowed (not yet set)
    v[fs.regime_index("ranging")] = 1.0
    fs.validate(v)


def test_validate_accepts_full_input_with_htf():
    fs.validate(np.zeros(fs.INPUT, dtype=np.float32), allow_htf=True)


def test_validate_rejects_wrong_length():
    with pytest.raises(ValueError):
        fs.validate(np.zeros(50, dtype=np.float32))


def test_validate_rejects_bad_regime_sum():
    v = np.zeros(fs.BASE, dtype=np.float32)
    v[fs.regime_index("uptrend")] = 1.0
    v[fs.regime_index("downtrend")] = 1.0  # two-hot -> sum 2.0
    with pytest.raises(ValueError):
        fs.validate(v)


def test_checkpoint_meta_roundtrip():
    meta = fs.checkpoint_meta(seq_len=60, hidden_size=256)
    assert meta["feature_version"] == fs.VERSION
    assert meta["input_size"] == fs.INPUT
    assert meta["orderbook_slots"] == 8
    assert meta["news_embed_dim"] == 16
    assert meta["seq_len"] == 60 and meta["hidden_size"] == 256
