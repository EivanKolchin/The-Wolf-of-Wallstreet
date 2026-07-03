"""Tests for news feedback persistence and MILD severity calibration."""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))


def test_adjust_severity_supports_mild(tmp_path, monkeypatch):
    from backend.agents import news_feedback
    monkeypatch.setattr(news_feedback, "PROFILE_PATH", tmp_path / "news_feedback.json")
    assert news_feedback.adjust_severity("MILD", 0.6) == "SIGNIFICANT"
    assert news_feedback.adjust_severity("SIGNIFICANT", -0.6) == "MILD"
    assert news_feedback.adjust_severity("MILD", -0.6) == "NEUTRAL"


def test_user_ratings_persist_across_load(tmp_path, monkeypatch):
    from backend.agents import news_feedback
    monkeypatch.setattr(news_feedback, "PROFILE_PATH", tmp_path / "news_feedback.json")
    news_feedback.record_feedback(
        feedback_type="relevance",
        rating=1,
        headline="Bitcoin ETF approval rally",
        source_domain="coindesk.com",
        article_hash="abc123",
    )
    ratings = news_feedback.get_rating_map()
    assert ratings.get("abc123") == 1
    assert ratings.get("coindesk.com:Bitcoin ETF approval rally") == 1


def test_classifier_feedback_persists_named_bucket_and_insignificant_suppresses(tmp_path, monkeypatch):
    from backend.agents import news_feedback
    monkeypatch.setattr(news_feedback, "PROFILE_PATH", tmp_path / "news_feedback.json")
    news_feedback.record_feedback(
        feedback_type="severity",
        rating=0,
        selected_value="INSIGNIFICANT",
        headline="Minor analyst chatter with no real catalyst",
        source_domain="example.com",
        article_hash="xyz789",
        severity="MILD",
        asset="AMD",
    )
    ratings = news_feedback.get_rating_map()
    assert ratings.get("xyz789") == "INSIGNIFICANT"
    assert news_feedback.article_interest_score("Minor analyst chatter with no real catalyst", "example.com") < 0


def test_build_search_query_for_earnings_gap():
    from backend.agents.web_search import build_search_query
    q = build_search_query(
        "When does NVDA report earnings?",
        {"requested_symbols": ["NVDA"], "earnings": {"NVDA": {"available": True, "next": None}}},
    )
    assert q is not None
    assert "NVDA" in q
