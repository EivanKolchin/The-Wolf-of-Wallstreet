"""Tests for the directional news → risk overlay (backend/risk/news_overlay.py)."""
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import pytest

from backend.risk.news_overlay import (assess_news_risk, NewsOverlayConfig, NewsRiskAction,
                                        _tighten_only)

NOW = datetime(2026, 6, 29, 12, 0, tzinfo=timezone.utc)


@dataclass
class FakeImpact:
    severity: str = "SIGNIFICANT"
    asset: str = "BTC-USD"
    direction: str = "down"
    magnitude_pct_low: float = 2.0
    magnitude_pct_high: float = 5.0
    confidence: float = 0.8
    t_min_minutes: int = 5
    t_max_minutes: int = 60
    trust_score: float = 0.9
    created_at: str = NOW.isoformat()
    symbol_relevance: dict | None = None


def test_no_news_allows_unchanged():
    a = assess_news_risk("BTC-USD", "long", [], now=NOW)
    assert a.action == "allow" and a.size_scale == 1.0 and not a.blocks_trade


def test_neutral_news_is_ignored():
    a = assess_news_risk("BTC-USD", "long", [FakeImpact(severity="NEUTRAL")], now=NOW)
    assert a.action == "allow" and a.size_scale == 1.0


def test_severe_bearish_against_long_halts_asset():
    imp = FakeImpact(severity="SEVERE", direction="down",
                     magnitude_pct_low=10, magnitude_pct_high=20, t_max_minutes=120)
    a = assess_news_risk("BTC-USD", "long", [imp], now=NOW)
    assert a.action == "halt_asset" and a.blocks_trade and a.halt_seconds >= 120 * 60
    assert a.signed_score < 0


def test_severe_bullish_with_long_does_not_block():
    """Directionality: a SEVERE *bullish* event must NOT halt a long — it aligns with the trade."""
    imp = FakeImpact(severity="SEVERE", direction="up",
                     magnitude_pct_low=10, magnitude_pct_high=20, t_max_minutes=120, confidence=0.9)
    a = assess_news_risk("BTC-USD", "long", [imp], now=NOW)
    assert not a.blocks_trade and a.signed_score > 0


def test_significant_against_long_degears():
    a = assess_news_risk("BTC-USD", "long", [FakeImpact(severity="SIGNIFICANT", direction="down")],
                         now=NOW)
    assert a.action == "resize" and 0.0 < a.size_scale < 1.0


def test_significant_bearish_helps_a_short():
    """Same bearish news that de-gears a long should be neutral/supportive for a short."""
    a = assess_news_risk("BTC-USD", "short", [FakeImpact(severity="SIGNIFICANT", direction="down")],
                         now=NOW)
    assert not a.blocks_trade and a.size_scale >= 1.0


def test_upsize_on_aligned_high_conf_near_term():
    imp = FakeImpact(severity="SEVERE", direction="up", confidence=0.9, t_max_minutes=30)
    a = assess_news_risk("BTC-USD", "long", [imp], now=NOW,
                         cfg=NewsOverlayConfig(allow_upsize=True))
    assert a.action == "resize" and a.size_scale > 1.0 and a.size_scale <= 1.25


def test_no_upsize_when_disabled():
    imp = FakeImpact(severity="SEVERE", direction="up", confidence=0.9, t_max_minutes=30)
    a = assess_news_risk("BTC-USD", "long", [imp], now=NOW,
                         cfg=NewsOverlayConfig(allow_upsize=False))
    assert a.size_scale <= 1.0


def test_stale_news_ignored():
    old = FakeImpact(severity="SEVERE", direction="down",
                     created_at=(NOW - timedelta(hours=24)).isoformat(), t_max_minutes=60)
    a = assess_news_risk("BTC-USD", "long", [old], now=NOW)
    assert a.action == "allow"


def test_irrelevant_asset_ignored():
    imp = FakeImpact(severity="SEVERE", direction="down", asset="ETH-USD")
    a = assess_news_risk("BTC-USD", "long", [imp], now=NOW)
    assert a.action == "allow"


def test_symbol_relevance_map_used():
    imp = FakeImpact(severity="SIGNIFICANT", direction="down", asset="MARKET",
                     symbol_relevance={"BTC-USD": 1.0})
    a = assess_news_risk("BTC-USD", "long", [imp], now=NOW)
    assert a.action == "resize" and a.size_scale < 1.0


def test_hold_with_severe_news_can_block():
    """A 'hold' has no direction; severe news is treated as against (caution)."""
    imp = FakeImpact(severity="SEVERE", direction="down",
                     magnitude_pct_high=15, t_max_minutes=90)
    a = assess_news_risk("BTC-USD", "hold", [imp], now=NOW)
    assert a.blocks_trade


def test_llm_verifier_can_only_tighten():
    # base = allow; LLM tries to veto → allowed (more conservative)
    base = NewsRiskAction("allow", 1.0, 0.0, 0.0, "base")
    tighter = NewsRiskAction("veto", 0.0, 0.0, 0.0, "llm caution")
    assert _tighten_only(base, tighter).action == "veto"
    # base = veto; LLM tries to allow → refused (cannot loosen)
    base2 = NewsRiskAction("veto", 0.0, 0.0, 0.0, "base")
    looser = NewsRiskAction("allow", 1.0, 0.0, 0.0, "llm over-eager")
    assert _tighten_only(base2, looser).blocks_trade


def test_llm_verifier_integration_tightens_size():
    def verifier(symbol, direction, impacts, base):
        return NewsRiskAction("resize", 0.1, 0.0, base.signed_score, "llm extra caution")
    a = assess_news_risk("BTC-USD", "long", [FakeImpact(severity="SIGNIFICANT", direction="down")],
                         now=NOW, llm_verifier=verifier)
    assert a.size_scale <= 0.1
