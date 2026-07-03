"""RiskAgent — the LLM cross-check between the trading engine's decision and the news.

This is the owner's idea: before a proposed trade goes out, an LLM looks at (a) what the engine wants
to do, (b) the relevant news, and (c) the agent's own accumulated lessons, and decides whether the
news materially threatens the trade. Its verdict can VETO the signal, shrink it, or temporarily halt
the asset on a severe outlier — but, by construction, it can only ever make the action MORE cautious
(the news overlay's ``_tighten_only`` rail enforces this), so an LLM mistake can never enlarge risk.

It plugs in as the StrategyAgent's ``news_verifier`` (async ``verdict(...)``). When realised PnL later
contradicts a call, ``record_outcome`` writes a lesson into the agent's SkillBook, escalating the
importance of recurring mistakes — and those lessons are fed back into the next prompt, so the agent
learns from its own history without any retraining.

Degrades gracefully: with no LLM service (or any error / unparseable reply) ``verdict`` returns None,
and the deterministic news overlay stands alone. Pure-ish: the only side effects are the LLM call and
the SkillBook file; both are injected, so this is testable with fakes.
"""
from __future__ import annotations

import json
import re
from typing import Any, List, Optional

try:
    import structlog
    logger = structlog.get_logger("risk_agent")
except Exception:  # pragma: no cover
    import logging
    logger = logging.getLogger("risk_agent")

from backend.risk.news_overlay import NewsRiskAction
from backend.agents.skills import SkillBook

VERIFY_PROMPT = """You are the RISK CROSS-CHECK for a live trading system. The engine proposes a trade.
Your ONLY job is to judge whether the NEWS makes this trade more dangerous, and if so, how much to
pull back. You can never make the trade bigger or safer than the engine already decided — only equal
or more cautious.

Proposed trade: {direction} {symbol}
Engine's pre-news stance: action={base_action}, size_scale={base_scale:.2f}
Recent news for {symbol}:
{news_block}
{lessons_block}
Decide. Respond in raw JSON only, no markdown:
{{
  "action": "allow" | "resize" | "veto" | "halt_asset",
  "size_scale": 0.0-1.0,        // multiply the engine size by this (<= the engine's, or smaller)
  "halt_minutes": 0,            // >0 only for halt_asset (how long to stand aside on this asset)
  "reason": "one short sentence"
}}
Guidance: news that OPPOSES the trade direction → resize down or veto; a severe opposing outlier
(hack, ban, depeg, fraud, halt) likely to move the asset hard → halt_asset. News that AGREES with the
trade → "allow" (do NOT upsize; that is not your job). Neutral/irrelevant/stale news → "allow"."""


def _news_block(impacts: List[Any], limit: int = 6) -> str:
    lines = []
    for imp in (impacts or [])[:limit]:
        sev = getattr(imp, "severity", "?")
        dirn = getattr(imp, "direction", "?")
        lo = getattr(imp, "magnitude_pct_low", 0.0)
        hi = getattr(imp, "magnitude_pct_high", 0.0)
        rat = str(getattr(imp, "rationale", "") or "")[:140]
        lines.append(f"- [{sev}/{dirn}] ~{lo:.0f}-{hi:.0f}% : {rat}")
    return "\n".join(lines) if lines else "- (none)"


def _extract_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text or "", re.DOTALL)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}


class NewsRiskVerifier:
    """LLM verifier usable as StrategyAgent.news_verifier. Async ``verdict`` returns a NewsRiskAction
    (the news overlay then keeps whichever of {engine, this} is MORE cautious) or None to abstain."""

    def __init__(self, llm_service: Any = None, skill_book: Optional[SkillBook] = None, *,
                 tier: str = "haiku", max_tokens: int = 200, lessons_in_prompt: int = 4):
        self.llm = llm_service
        self.skills = skill_book or SkillBook("risk_agent")
        self.tier = tier
        self.max_tokens = max_tokens
        self.lessons_in_prompt = lessons_in_prompt

    async def verdict(self, symbol: str, direction: str, impacts: List[Any],
                      base_action: NewsRiskAction) -> Optional[NewsRiskAction]:
        if self.llm is None or not impacts:
            return None
        lessons = self.skills.as_prompt(self.lessons_in_prompt)
        lessons_block = f"\nLessons from past mistakes — weigh these:\n{lessons}\n" if lessons else ""
        prompt = VERIFY_PROMPT.format(
            symbol=symbol, direction=direction, base_action=base_action.action,
            base_scale=base_action.size_scale, news_block=_news_block(impacts),
            lessons_block=lessons_block)
        try:
            text = await self.llm.generate_text(prompt, tier=self.tier, max_tokens=self.max_tokens)
        except Exception as e:
            logger.warning("risk_agent_llm_error", symbol=symbol, error=str(e)[:120])
            return None
        data = _extract_json(text)
        if not data:
            return None
        action = str(data.get("action", "allow")).lower()
        if action not in ("allow", "resize", "veto", "halt_asset"):
            return None
        try:
            scale = max(0.0, min(1.0, float(data.get("size_scale", 1.0))))
        except (TypeError, ValueError):
            scale = 1.0
        if action in ("veto", "halt_asset"):
            scale = 0.0
        halt_s = max(0.0, float(data.get("halt_minutes", 0) or 0)) * 60.0
        reason = str(data.get("reason", "llm cross-check"))[:160]
        return NewsRiskAction(action=action, size_scale=scale, halt_seconds=halt_s,
                              signed_score=base_action.signed_score, reason=reason)

    # ── self-improvement: record when a call didn't play out, escalating repeat mistakes ──
    def record_outcome(self, *, symbol: str, predicted: str, realized_bad: bool, note: str,
                       tag: str = "news_risk", importance: float = 1.0) -> None:
        """Call from the settlement/outcome path. ``realized_bad`` True = the trade went against the
        verifier's call → write/escalate a lesson so the agent avoids the same mistake next time."""
        if not realized_bad:
            return
        lesson = f"{symbol}: {note} (predicted {predicted})"
        les = self.skills.record(lesson, tag=tag, importance=importance)
        logger.info("risk_agent_lesson_recorded", symbol=symbol, importance=les.importance,
                    occurrences=les.occurrences)
