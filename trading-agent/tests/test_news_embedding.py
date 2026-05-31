"""Phase 3 tests: semantic news embedding for the NEWS_EMBED feature block.

The non-negotiable property is determinism + dimension-stability: the same text
must map to the same vector across processes/instances, so the offline pretrain
alignment and the live agent never drift apart.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from signals import feature_spec as fs  # noqa: E402
from backend.signals.news_embedding import (  # noqa: E402
    NewsEmbedder, text_for_news_impact, DIM,
)


def test_dim_matches_feature_spec():
    assert DIM == fs.NEWS_EMBED_DIM == 16


def test_embedding_is_deterministic_across_instances():
    txt = "Bitcoin ETF approval sends BTC surging to new highs"
    a = NewsEmbedder(backend="hashing").embed_text(txt)
    b = NewsEmbedder(backend="hashing").embed_text(txt)
    assert a.shape == (DIM,)
    np.testing.assert_array_equal(a, b)


def test_empty_text_is_zero_vector():
    e = NewsEmbedder(backend="hashing")
    np.testing.assert_array_equal(e.embed_text(""), np.zeros(DIM, dtype=np.float32))
    np.testing.assert_array_equal(e.embed_text("   "), np.zeros(DIM, dtype=np.float32))


def test_distinct_texts_differ_and_are_unit_norm():
    e = NewsEmbedder(backend="hashing")
    a = e.embed_text("regulator approves spot bitcoin etf bullish")
    b = e.embed_text("exchange hacked funds stolen bearish crash")
    assert not np.allclose(a, b)
    # non-empty embeddings are L2-normalized
    assert abs(float(np.linalg.norm(a)) - 1.0) < 1e-5
    assert abs(float(np.linalg.norm(b)) - 1.0) < 1e-5


def test_text_for_news_impact_composes_fields():
    class _Impact:
        asset = "BTC"
        direction = "up"
        severity = "SIGNIFICANT"
        rationale = "Halving narrative drives demand"
        matched_keywords = {"BTCUSDT": ["halving", "demand"]}

    txt = text_for_news_impact(_Impact())
    assert "BTC" in txt and "up" in txt and "halving" in txt
    # embedding of a populated impact is non-zero
    e = NewsEmbedder(backend="hashing")
    assert float(np.linalg.norm(e.embed_news_impact(_Impact()))) > 0.0
