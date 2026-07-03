"""Tests for the self-improving agent skill books (backend/agents/skills.py)."""
from datetime import timedelta

import pytest

from backend.agents.skills import SkillBook, _norm_key, _now


@pytest.fixture
def book(tmp_path):
    return SkillBook("test_agent", skills_dir=tmp_path)


def test_record_creates_lesson(book):
    les = book.record("avoid longing into a regulatory ban", tag="news")
    assert les.occurrences == 1 and les.importance == 1.0
    assert book.path.exists()


def test_recurrence_escalates_importance(book):
    book.record("avoid longing into a regulatory ban", tag="news")
    les = book.record("Avoid longing into a regulatory ban!", tag="news")  # same skeleton
    assert les.occurrences == 2
    assert les.importance > 1.0                       # escalated
    assert len(book.all_lessons()) == 1               # de-duplicated, not a new entry


def test_distinct_lessons_not_merged(book):
    book.record("breakout failed in chop", tag="momentum")
    book.record("stop too tight on high ATR", tag="risk")
    assert len(book.all_lessons()) == 2


def test_top_lessons_ranked_by_importance(book):
    book.record("minor one-off", tag="a")
    for _ in range(4):
        book.record("chronic failure mode", tag="b")  # escalates 4x
    top = book.top_lessons(2)
    assert top[0].text == "chronic failure mode"
    assert top[0].importance > top[1].importance


def test_persistence_roundtrip(tmp_path):
    b1 = SkillBook("agentX", skills_dir=tmp_path)
    b1.record("lesson one", tag="t1", importance=2.0)
    b1.record("lesson two", tag="t2")
    b2 = SkillBook("agentX", skills_dir=tmp_path)         # reload from disk
    texts = {l.text for l in b2.all_lessons()}
    assert texts == {"lesson one", "lesson two"}
    assert any(abs(l.importance - 2.0) < 1e-6 for l in b2.all_lessons())


def test_age_decay_sinks_stale_lessons(book):
    fresh = book.record("fresh lesson", tag="f")
    old = book.record("old lesson", tag="o")
    # manually age the 'old' lesson well past the half-life
    old.last_seen = (_now() - timedelta(days=120)).isoformat()
    eff_fresh = book.effective_importance(fresh)
    eff_old = book.effective_importance(old)
    assert eff_fresh > eff_old


def test_as_prompt_contains_top_lesson(book):
    book.record("size down before FOMC", tag="macro", importance=3.0)
    prompt = book.as_prompt(3)
    assert "size down before FOMC" in prompt and "test_agent" in prompt


def test_empty_prompt_when_no_lessons(book):
    assert book.as_prompt() == ""


def test_norm_key_ignores_punct_and_case():
    assert _norm_key("news", "Hello, World!") == _norm_key("news", "hello world")
    assert _norm_key("a", "x") != _norm_key("b", "x")
