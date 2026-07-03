"""Self-improving agent skill books — a per-agent markdown file of LESSONS that the agent edits
autonomously as its predictions play out.

The idea (operator's): every agent keeps a human-readable ``.md`` of things it has learned. When an
agent makes a call that does NOT go as predicted, it records the lesson; if the SAME kind of mistake
recurs, the lesson's importance escalates (so persistent failure modes float to the top). The top
lessons are injected back into the agent's LLM prompt, so the agent literally gets better at the
things it keeps getting wrong — a cheap, transparent, no-retraining feedback loop.

Design choices:
  * Markdown, not a binary store → the owner can read/edit/delete lessons by hand; it doubles as an
    audit log of what each agent has learned and why.
  * De-duplication by a normalised KEY (tag + lowercased text), so recurring mistakes increment one
    lesson's ``occurrences`` + ``importance`` instead of spawning near-duplicates.
  * Importance escalates with each recurrence and decays slowly with age, so stale one-off lessons
    sink and chronic ones rise. ``top_lessons`` returns the prompt-ready shortlist.
  * Pure file I/O + parsing, no LLM / redis / network → fully unit-testable. Concurrency-safe enough
    for the single-writer-per-agent live loop (atomic replace on save).

This module only STORES and RANKS lessons; deciding when to call ``record`` is the agent's job
(typically in its outcome/settlement callback when realised PnL contradicts the prediction).
"""
from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

DEFAULT_SKILLS_DIR = Path(__file__).resolve().parent / "skills"
_ENTRY_RE = re.compile(r"^- \[(?P<imp>[0-9.]+)\]\s*\((?P<meta>\{.*?\})\)\s*(?P<text>.*)$")


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _norm_key(tag: str, text: str) -> str:
    """Stable de-dup key: tag + the lowercased alnum skeleton of the text (whitespace/punct-insensitive)."""
    skel = re.sub(r"[^a-z0-9 ]+", "", (text or "").lower())
    skel = re.sub(r"\s+", " ", skel).strip()
    return f"{(tag or 'general').lower()}::{skel}"


@dataclass
class Lesson:
    text: str
    tag: str = "general"
    importance: float = 1.0          # current rank weight (escalates on recurrence, decays with age)
    occurrences: int = 1             # how many times this lesson was recorded
    created_at: str = ""
    last_seen: str = ""

    def key(self) -> str:
        return _norm_key(self.tag, self.text)


class SkillBook:
    """One markdown lesson file per agent. ``record`` adds/escalates a lesson; ``top_lessons``
    returns the prompt-ready shortlist (highest effective importance first)."""

    def __init__(self, agent: str, skills_dir: Optional[Path] = None, *,
                 escalation: float = 0.75, half_life_days: float = 30.0, max_lessons: int = 200):
        self.agent = agent
        self.dir = Path(skills_dir or DEFAULT_SKILLS_DIR)
        self.path = self.dir / f"{agent}.md"
        self.escalation = escalation          # importance added per recurrence
        self.half_life_days = half_life_days  # age decay applied in effective_importance
        self.max_lessons = max_lessons
        self._lessons: Dict[str, Lesson] = {}
        self._load()

    # ───────────────────────── public API ─────────────────────────
    def record(self, text: str, *, tag: str = "general", importance: float = 1.0) -> Lesson:
        """Record a lesson. If an equivalent lesson exists (same tag + text skeleton), escalate its
        importance and bump occurrences instead of duplicating. Returns the (new/updated) lesson."""
        key = _norm_key(tag, text)
        now = _now().isoformat()
        if key in self._lessons:
            les = self._lessons[key]
            les.occurrences += 1
            les.importance = round(les.importance + self.escalation * max(0.1, importance), 3)
            les.last_seen = now
        else:
            les = Lesson(text=text.strip(), tag=tag, importance=float(importance),
                         occurrences=1, created_at=now, last_seen=now)
            self._lessons[key] = les
        self._save()
        return les

    def effective_importance(self, les: Lesson, *, now: Optional[datetime] = None) -> float:
        """Importance discounted by age since last seen (exponential half-life). Chronic lessons
        (recently re-seen, high occurrences) stay high; one-off old lessons decay toward 0."""
        now = now or _now()
        try:
            age_days = (now - datetime.fromisoformat(les.last_seen)).total_seconds() / 86400.0
        except Exception:
            age_days = 0.0
        decay = 0.5 ** (max(0.0, age_days) / max(1e-6, self.half_life_days))
        return les.importance * decay

    def top_lessons(self, n: int = 5, *, min_importance: float = 0.0,
                    now: Optional[datetime] = None) -> List[Lesson]:
        ranked = sorted(self._lessons.values(),
                        key=lambda l: self.effective_importance(l, now=now), reverse=True)
        out = [l for l in ranked if self.effective_importance(l, now=now) >= min_importance]
        return out[:n]

    def as_prompt(self, n: int = 5, **kw) -> str:
        """Render the top lessons as a compact block to inject into an LLM system prompt."""
        top = self.top_lessons(n, **kw)
        if not top:
            return ""
        lines = [f"Lessons learned by {self.agent} (heed the highest-importance first):"]
        for l in top:
            lines.append(f"- (importance {l.importance:.1f}, seen {l.occurrences}x) [{l.tag}] {l.text}")
        return "\n".join(lines)

    def all_lessons(self) -> List[Lesson]:
        return list(self._lessons.values())

    # ───────────────────────── persistence (markdown) ─────────────────────────
    def _load(self) -> None:
        self._lessons.clear()
        if not self.path.exists():
            return
        for line in self.path.read_text(encoding="utf-8").splitlines():
            m = _ENTRY_RE.match(line.strip())
            if not m:
                continue
            try:
                meta = json.loads(m.group("meta"))
                les = Lesson(text=m.group("text").strip(), tag=meta.get("tag", "general"),
                             importance=float(m.group("imp")), occurrences=int(meta.get("n", 1)),
                             created_at=meta.get("c", ""), last_seen=meta.get("s", ""))
                self._lessons[les.key()] = les
            except Exception:
                continue

    def _save(self) -> None:
        # prune to the most-important max_lessons before writing
        if len(self._lessons) > self.max_lessons:
            keep = sorted(self._lessons.values(), key=self.effective_importance, reverse=True)
            self._lessons = {l.key(): l for l in keep[:self.max_lessons]}
        self.dir.mkdir(parents=True, exist_ok=True)
        lines = [f"# Skill book — {self.agent}", "",
                 "_Autonomously maintained. Each line: `- [importance] (meta) lesson`._", ""]
        for l in sorted(self._lessons.values(), key=self.effective_importance, reverse=True):
            meta = json.dumps({"tag": l.tag, "n": l.occurrences, "c": l.created_at, "s": l.last_seen},
                              separators=(",", ":"))
            lines.append(f"- [{l.importance:g}] ({meta}) {l.text}")
        text = "\n".join(lines) + "\n"
        # atomic replace so a crash mid-write can't corrupt the file
        fd, tmp = tempfile.mkstemp(dir=str(self.dir), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(text)
            os.replace(tmp, self.path)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
