#!/usr/bin/env python
"""Periodic post-mortem over the trade journal — classify FIRST, learn SECOND.

The external systems that "fix something after every loss" are fitting noise: a correct +EV
book loses ~45% of its bets, and this project already measured what per-outcome adaptation
does (online AWR = noise; meta-label in-sample 0.90 AUC → OOS 0.50). So this job runs on a
WINDOW (default 7 days), and every concern is classified before anything learns from it:

  PROCESS  — bugs/violations (slippage far above the assumed model, journal gaps, gross
             exposure beyond cap, blocked-order pileups). ALWAYS actionable → lessons + report.
  NOISE    — drawdown/losses within statistical expectation for the book's vol. Explicitly
             NOT actionable: the report says so, and no lesson is written.
  REGIME   — sustained decay (hit-rate or book return degrading across the window). Actionable
             at the ALLOCATION level only (the overlay's conviction-shrink and the de-gear
             already respond mechanically); a lesson records the observation.

Outputs:
  1. ``statements/postmortem_<date>.json`` — the full aggregate report. THIS is the artifact
     the owner hands to the strong reviewing model (Claude) for deep review + code changes.
  2. Lessons appended to the strategy agent's SkillBook (LLM-phrased when an LLM is
     configured, deterministic phrasing otherwise) — the local model inherits the distilled
     judgment either way (skills/prompts only; this job NEVER edits code or parameters).

Usage:
    python scripts/postmortem.py                 # last 7 days
    python scripts/postmortem.py --days 30
    python scripts/postmortem.py --no-llm        # deterministic lessons only
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from backend.agents.trade_journal import TradeJournal          # noqa: E402
from backend.agents.skills import SkillBook                    # noqa: E402
from backend.execution.slippage_ledger import SlippageLedger   # noqa: E402


# ─────────────────────────── pure aggregation ───────────────────────────
def aggregate(rows: List[dict], slippage_summary: dict) -> dict:
    """Window aggregates from journal rows — the raw material for classification."""
    marks = [r for r in rows if r.get("kind") == "mark"]
    outcomes = [r for r in rows if r.get("kind") == "outcome"]
    rebs = [r for r in rows if r.get("kind") == "rebalance"]
    orders = [r for r in rows if r.get("kind") == "order"]

    rets = [float(m.get("book_return", 0.0)) for m in marks]
    n = len(rets)
    mean_r = sum(rets) / n if n else 0.0
    var = sum((r - mean_r) ** 2 for r in rets) / max(n - 1, 1) if n > 1 else 0.0
    vol = math.sqrt(var)
    total_ret = 1.0
    for r in rets:
        total_ret *= (1.0 + r)
    total_ret -= 1.0

    per_symbol: Dict[str, dict] = {}
    for o in outcomes:
        s = o.get("symbol")
        d = per_symbol.setdefault(s, {"n": 0, "wins": 0, "contribution": 0.0})
        c = float(o.get("contribution", 0.0))
        d["n"] += 1
        d["wins"] += 1 if c > 0 else 0
        d["contribution"] += c
    for d in per_symbol.values():
        d["hit_rate"] = round(d["wins"] / d["n"], 3) if d["n"] else 0.0
        d["contribution"] = round(d["contribution"], 5)

    blocked = [r for r in rebs if r.get("blocked")]
    max_gross = max((float(r.get("gross", 0.0)) for r in rebs), default=0.0)
    worst_dd = min((float(m.get("drawdown", 0.0)) for m in marks), default=0.0)

    # journal-gap detection: marks should arrive ~every rebalance interval
    gaps = 0
    ts = sorted(float(m.get("ts", 0)) for m in marks)
    if len(ts) > 2:
        diffs = [b - a for a, b in zip(ts, ts[1:])]
        med = sorted(diffs)[len(diffs) // 2]
        gaps = sum(1 for d in diffs if med > 0 and d > 3.0 * med)

    return {
        "n_marks": n, "n_rebalances": len(rebs), "n_orders": len(orders),
        "window_return": round(total_ret, 5), "per_mark_vol": round(vol, 5),
        "worst_drawdown": round(worst_dd, 4), "max_gross": round(max_gross, 3),
        "n_blocked_rebalances": len(blocked),
        "journal_gaps": gaps,
        "per_symbol": per_symbol,
        "slippage": slippage_summary or {},
    }


def classify(agg: dict, *, gross_cap: float = 2.0, slip_tolerance_bps: float = 5.0,
             regime_hit_floor: float = 0.35) -> List[dict]:
    """[{class, code, detail}] — PROCESS concerns are actionable; NOISE explicitly is not."""
    concerns: List[dict] = []

    # ---- PROCESS: violations of how the system is SUPPOSED to behave ----
    for key, s in (agg.get("slippage") or {}).items():
        if float(s.get("vs_assumed_bps", 0.0)) > slip_tolerance_bps:
            concerns.append({"class": "process", "code": "slippage_above_model",
                             "detail": f"{key}: realized {s.get('mean_bps')}bps vs assumed "
                                       f"(+{s.get('vs_assumed_bps')}bps over) on n={s.get('n')}"})
    if agg.get("max_gross", 0.0) > gross_cap:
        concerns.append({"class": "process", "code": "gross_exposure_over_cap",
                         "detail": f"max gross {agg['max_gross']} > cap {gross_cap}"})
    if agg.get("journal_gaps", 0) > 0:
        concerns.append({"class": "process", "code": "data_or_uptime_gap",
                         "detail": f"{agg['journal_gaps']} mark gap(s) >3× the median interval "
                                   f"(agent down or data feed stalled)"})
    if agg.get("n_blocked_rebalances", 0) >= 3:
        concerns.append({"class": "process", "code": "repeated_risk_blocks",
                         "detail": f"{agg['n_blocked_rebalances']} rebalances fully blocked — "
                                   f"check halt state / data sufficiency"})

    # ---- NOISE vs REGIME on the window PnL: is the loss within expectation? ----
    n, vol, ret = agg.get("n_marks", 0), agg.get("per_mark_vol", 0.0), agg.get("window_return", 0.0)
    if n >= 5 and ret < 0:
        # z-score of the window's compounded return against its own per-mark vol. A
        # zero-variance losing stream (vol == 0) is a deterministic bleed — the most
        # "beyond expectation" case there is, so it must NOT slip through a vol>0 guard.
        z = ret / (vol * math.sqrt(n)) if vol > 0 else float("-inf")
        if z > -2.0:
            concerns.append({"class": "noise", "code": "loss_within_expectation",
                             "detail": f"window return {ret:+.2%} is z={z:+.2f} of own vol — "
                                       f"statistically unremarkable; DO NOT change the system"})
        else:
            concerns.append({"class": "regime", "code": "loss_beyond_expectation",
                             "detail": f"window return {ret:+.2%} at z={z:+.2f} (<-2σ) — possible "
                                       f"regime shift; de-gear/shrink respond mechanically, "
                                       f"review sleeve allocation"})
    # sustained per-symbol decay (many symbols with hit-rate below floor over enough marks)
    weak = [s for s, d in (agg.get("per_symbol") or {}).items()
            if d.get("n", 0) >= 10 and d.get("hit_rate", 1.0) < regime_hit_floor]
    if len(weak) >= 3:
        concerns.append({"class": "regime", "code": "broad_hit_rate_decay",
                         "detail": f"hit-rate < {regime_hit_floor:.0%} on {len(weak)} symbols "
                                   f"({', '.join(weak[:5])}…) — trend regime may be fading"})
    return concerns


def deterministic_lessons(concerns: List[dict]) -> List[dict]:
    """Fallback lesson texts when no LLM is available. PROCESS/REGIME only — noise never
    generates a lesson (that would be the per-loss-mutation trap)."""
    out = []
    for c in concerns:
        if c["class"] == "noise":
            continue
        out.append({"tag": c["code"],
                    "text": f"[{c['class'].upper()}] {c['detail']}"})
    return out


# ─────────────────────────── LLM-optional lesson writer ───────────────────────────
async def llm_lessons(agg: dict, concerns: List[dict]) -> List[dict]:
    """Ask the configured LLM to phrase ≤3 lessons from the classified concerns. The LLM can
    only PHRASE — the classification (and the never-learn-from-noise rule) is deterministic."""
    from backend.core.config import settings
    from backend.agents.llm import LLMService
    actionable = [c for c in concerns if c["class"] != "noise"]
    if not actionable:
        return []
    svc = LLMService(provider=settings.AI_PROVIDER, anthropic_key=settings.ANTHROPIC_API_KEY,
                     gemini_key=settings.GEMINI_API_KEY,
                     ollama_model=getattr(settings, "OLLAMA_MODEL", "llama3"))
    prompt = (
        "You are the post-mortem writer for a systematic trading book. From the classified "
        "concerns below, write AT MOST 3 short lessons (max 25 words each) for the trading "
        "agent's skill book. Rules: only PROCESS or REGIME concerns become lessons; never "
        "suggest reacting to statistical noise; never suggest code or parameter changes — "
        "lessons describe what to watch or verify. Return ONLY a JSON array of "
        '{"tag": "<concern code>", "text": "<lesson>"}.\n\n'
        f"AGGREGATES: {json.dumps({k: v for k, v in agg.items() if k != 'per_symbol'})}\n"
        f"CONCERNS: {json.dumps(actionable)}"
    )
    try:
        raw = await svc.generate_text(prompt, tier="haiku", max_tokens=400, json_mode=True)
        import re
        m = re.search(r"\[.*\]", raw or "", re.DOTALL)
        items = json.loads(m.group(0)) if m else []
        out = []
        for it in items[:3]:
            if isinstance(it, dict) and it.get("text"):
                out.append({"tag": str(it.get("tag", "postmortem"))[:40],
                            "text": str(it["text"])[:200]})
        return out or deterministic_lessons(concerns)
    except Exception:
        return deterministic_lessons(concerns)


# ─────────────────────────── reusable entry ───────────────────────────
def run_once(days: float = 7.0, *, use_llm: bool = True, journal_path=None,
             write_report: bool = True) -> Optional[dict]:
    """Run one post-mortem over the last ``days``: aggregate → classify → write lessons to the
    strategy_agent SkillBook → (optionally) persist a report JSON. Returns the report dict, or
    None when there are no journal rows. Importable so the backend can SCHEDULE it, not just CLI.
    ``use_llm=False`` uses deterministic lessons only (no LLM dependency / no network)."""
    journal = TradeJournal(path=Path(journal_path)) if journal_path else TradeJournal()
    since = time.time() - days * 86400.0
    rows = journal.read(since_ts=since)
    if not rows:
        return None

    slip = SlippageLedger().summary()
    agg = aggregate(rows, slip)
    concerns = classify(agg)

    if not use_llm:
        lessons = deterministic_lessons(concerns)
    else:
        import asyncio
        try:
            lessons = asyncio.run(llm_lessons(agg, concerns))
        except RuntimeError:                       # already inside an event loop → deterministic
            lessons = deterministic_lessons(concerns)

    book = SkillBook("strategy_agent")
    for l in lessons:
        book.record(l["text"], tag=l["tag"])

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "window_days": days,
        "aggregates": agg,
        "concerns": concerns,
        "lessons_written": lessons,
        "note": ("Hand this file (plus training_data/trade_journal.jsonl and "
                 "training_data/slippage_log.jsonl) to the reviewing model for deep "
                 "post-mortem and code-level fixes."),
    }
    if write_report:
        out_dir = ROOT / "statements"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"postmortem_{datetime.now(timezone.utc).strftime('%Y%m%d')}.json"
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        report["report_path"] = str(out_path)
    return report


# ─────────────────────────── main ───────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--days", type=float, default=7.0)
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--journal", default=None, help="override journal path (tests)")
    args = ap.parse_args()

    report = run_once(args.days, use_llm=not args.no_llm, journal_path=args.journal)
    if report is None:
        print(f"No journal rows in the last {args.days:g} days — nothing to review.")
        return
    agg, concerns, lessons = report["aggregates"], report["concerns"], report["lessons_written"]
    print(f"=== POST-MORTEM — last {args.days:g} days "
          f"({agg['n_marks']} marks, {agg['n_orders']} orders) ===")
    print(f"window return {agg['window_return']:+.2%}   worst DD {agg['worst_drawdown']:+.1%}   "
          f"max gross {agg['max_gross']:.2f}")
    for c in concerns:
        print(f"  [{c['class'].upper():7s}] {c['code']}: {c['detail']}")
    if not concerns:
        print("  no concerns — book behaving within specification")
    print(f"lessons written to skill book: {len(lessons)}")
    print(f"report: {report.get('report_path')}")


if __name__ == "__main__":
    main()
