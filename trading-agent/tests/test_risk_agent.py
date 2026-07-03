"""Tests for the LLM RiskAgent verifier (backend/agents/risk_agent.py)."""
import asyncio
import json
from dataclasses import dataclass

import pytest

from backend.agents.risk_agent import NewsRiskVerifier
from backend.agents.skills import SkillBook
from backend.risk.news_overlay import NewsRiskAction


@dataclass
class FakeImpact:
    severity: str = "SEVERE"
    asset: str = "BTC-USD"
    direction: str = "down"
    magnitude_pct_low: float = 8.0
    magnitude_pct_high: float = 15.0
    confidence: float = 0.8
    rationale: str = "exchange halt rumor"


class FakeLLM:
    def __init__(self, reply):
        self.reply = reply
        self.prompts = []

    async def generate_text(self, prompt, tier="haiku", max_tokens=200):
        self.prompts.append(prompt)
        return self.reply


def _base():
    return NewsRiskAction("resize", 0.6, 0.0, -0.4, "deterministic base")


def test_no_llm_abstains(tmp_path):
    v = NewsRiskVerifier(llm_service=None, skill_book=SkillBook("ra", tmp_path))
    out = asyncio.run(v.verdict("BTC-USD", "long", [FakeImpact()], _base()))
    assert out is None


def test_no_impacts_abstains(tmp_path):
    v = NewsRiskVerifier(llm_service=FakeLLM("{}"), skill_book=SkillBook("ra", tmp_path))
    assert asyncio.run(v.verdict("BTC-USD", "long", [], _base())) is None


def test_parses_veto(tmp_path):
    llm = FakeLLM(json.dumps({"action": "veto", "size_scale": 0.5, "reason": "ban"}))
    v = NewsRiskVerifier(llm_service=llm, skill_book=SkillBook("ra", tmp_path))
    out = asyncio.run(v.verdict("BTC-USD", "long", [FakeImpact()], _base()))
    assert out.action == "veto" and out.size_scale == 0.0      # veto forces scale 0


def test_parses_halt_with_minutes(tmp_path):
    llm = FakeLLM(json.dumps({"action": "halt_asset", "halt_minutes": 90, "reason": "hack"}))
    v = NewsRiskVerifier(llm_service=llm, skill_book=SkillBook("ra", tmp_path))
    out = asyncio.run(v.verdict("BTC-USD", "long", [FakeImpact()], _base()))
    assert out.action == "halt_asset" and out.halt_seconds == 90 * 60


def test_garbage_reply_abstains(tmp_path):
    v = NewsRiskVerifier(llm_service=FakeLLM("not json at all"), skill_book=SkillBook("ra", tmp_path))
    assert asyncio.run(v.verdict("BTC-USD", "long", [FakeImpact()], _base())) is None


def test_lessons_injected_into_prompt(tmp_path):
    sb = SkillBook("ra", tmp_path)
    sb.record("halt BTC on exchange-halt rumors", tag="news_risk", importance=3.0)
    llm = FakeLLM(json.dumps({"action": "allow", "size_scale": 1.0, "reason": "ok"}))
    v = NewsRiskVerifier(llm_service=llm, skill_book=sb)
    asyncio.run(v.verdict("BTC-USD", "long", [FakeImpact()], _base()))
    assert "halt BTC on exchange-halt rumors" in llm.prompts[0]


def test_record_outcome_escalates(tmp_path):
    sb = SkillBook("ra", tmp_path)
    v = NewsRiskVerifier(llm_service=None, skill_book=sb)
    v.record_outcome(symbol="BTC-USD", predicted="allow", realized_bad=True, note="missed a hack")
    v.record_outcome(symbol="BTC-USD", predicted="allow", realized_bad=True, note="missed a hack")
    lessons = sb.all_lessons()
    assert len(lessons) == 1 and lessons[0].occurrences == 2 and lessons[0].importance > 1.0


def test_record_outcome_noop_when_good(tmp_path):
    sb = SkillBook("ra", tmp_path)
    v = NewsRiskVerifier(llm_service=None, skill_book=sb)
    v.record_outcome(symbol="BTC-USD", predicted="veto", realized_bad=False, note="correctly vetoed")
    assert sb.all_lessons() == []
