"""User feedback memory for the live news scanner.

FastAPI records button clicks here and the background news agent reads the same
profile before deciding what to send to the LLM and how to calibrate severity.
It is intentionally file-backed so it works across the API/news-agent processes
and survives restarts without needing a schema migration.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

from backend.core.config import BASE_DIR

PROFILE_PATH = BASE_DIR / "training_data" / "news_feedback.json"

_TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_\-]{2,}")
_STOPWORDS = {
    "about", "after", "again", "against", "also", "amid", "been", "before",
    "being", "between", "from", "have", "into", "market", "markets", "more",
    "news", "over", "report", "says", "than", "that", "their", "this",
    "under", "with", "will", "would",
}


SEVERITY_LEVELS = ["NEUTRAL", "MILD", "SIGNIFICANT", "SEVERE"]


def _default_profile() -> dict[str, Any]:
    return {
        "source_weights": {},
        "keyword_weights": {},
        "severity_weights": {},
        "user_ratings": {},
        "events": [],
    }


def rating_keys(
    *,
    article_hash: str | None = None,
    prediction_id: str | None = None,
    headline: str | None = None,
    source_domain: str | None = None,
) -> list[str]:
    """All lookup keys used by the frontend widgets for one news item."""
    keys: list[str] = []
    if article_hash:
        keys.append(str(article_hash))
    if prediction_id:
        keys.append(str(prediction_id))
    src = (source_domain or "").strip()
    head = (headline or "").strip()
    if src or head:
        keys.append(f"{src}:{head}")
    return list(dict.fromkeys(k for k in keys if k))


def _tokens(text: str | None) -> list[str]:
    raw = (text or "").lower()
    out: list[str] = []
    for token in _TOKEN_RE.findall(raw):
        if token in _STOPWORDS:
            continue
        if token.isdigit():
            continue
        out.append(token[:40])
    return list(dict.fromkeys(out))[:16]


def load_profile() -> dict[str, Any]:
    try:
        if PROFILE_PATH.exists():
            data = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
            base = _default_profile()
            base.update(data if isinstance(data, dict) else {})
            return base
    except Exception:
        pass
    return _default_profile()


def save_profile(profile: dict[str, Any]) -> None:
    PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROFILE_PATH.write_text(json.dumps(profile, indent=2, sort_keys=True), encoding="utf-8")


def _bump(bucket: dict[str, Any], key: str, delta: float) -> None:
    if not key:
        return
    current = float(bucket.get(key, 0.0) or 0.0)
    bucket[key] = round(max(-3.0, min(3.0, current + delta)), 4)


def record_feedback(
    *,
    feedback_type: str,
    rating: int,
    selected_value: str | int | None = None,
    headline: str | None = None,
    source_domain: str | None = None,
    asset: str | None = None,
    severity: str | None = None,
    article_hash: str | None = None,
    prediction_id: str | None = None,
) -> dict[str, Any]:
    """Persist one feedback event and update the learnable profile.

    ``feedback_type=relevance``: +1 means scan/analyse more similar articles,
    -1 means suppress similar articles.

    ``feedback_type=severity``: +1 means similar articles should be treated as
    more severe, -1 means similar articles should be treated as less severe.
    """
    rating = 1 if int(rating or 0) > 0 else -1
    ftype = (feedback_type or "").strip().lower()
    if ftype not in {"relevance", "severity"}:
        raise ValueError("feedback_type must be 'relevance' or 'severity'")

    profile = load_profile()
    source = (source_domain or "").strip().lower()
    words = _tokens(headline)

    if ftype == "relevance":
        _bump(profile["source_weights"], source, 0.35 * rating)
        for word in words:
            _bump(profile["keyword_weights"], word, 0.22 * rating)
    else:
        chosen = str(selected_value or "").strip().upper()
        current_bucket = str(severity or "NEUTRAL").strip().upper()
        target_bucket = "NEUTRAL" if chosen == "INSIGNIFICANT" else (chosen or current_bucket)
        if target_bucket not in SEVERITY_LEVELS:
            target_bucket = current_bucket if current_bucket in SEVERITY_LEVELS else "NEUTRAL"
        current_idx = SEVERITY_LEVELS.index(current_bucket) if current_bucket in SEVERITY_LEVELS else 0
        target_idx = SEVERITY_LEVELS.index(target_bucket)
        step_delta = target_idx - current_idx
        if step_delta == 0:
            step_delta = 1 if target_idx >= current_idx else -1
        magnitude = max(1, abs(step_delta))
        direction = 1 if step_delta > 0 else -1
        _bump(profile["severity_weights"], source, 0.26 * direction * magnitude)
        if asset:
            _bump(profile["severity_weights"], f"asset:{asset.upper()}", 0.22 * direction * magnitude)
        if severity:
            _bump(profile["severity_weights"], f"bucket:{current_bucket}", 0.10 * direction * magnitude)
        for word in words:
            _bump(profile["severity_weights"], word, 0.18 * direction * magnitude)
        if chosen == "INSIGNIFICANT":
            _bump(profile["source_weights"], source, -0.30)
            for word in words:
                _bump(profile["keyword_weights"], word, -0.20)

    event = {
        "ts": time.time(),
        "type": ftype,
        "rating": rating,
        "headline": (headline or "")[:500],
        "source_domain": source,
        "asset": asset,
        "severity": severity,
        "article_hash": article_hash,
        "prediction_id": prediction_id,
        "tokens": words,
    }
    profile["events"] = [event] + list(profile.get("events") or [])[:199]
    stored = {
        "feedback_type": ftype,
        "rating": rating,
        "selected_value": selected_value,
        "headline": (headline or "")[:500],
        "source_domain": source,
        "asset": asset,
        "severity": severity,
        "article_hash": article_hash,
        "prediction_id": prediction_id,
        "ts": time.time(),
    }
    ratings = profile.setdefault("user_ratings", {})
    for key in rating_keys(
        article_hash=article_hash,
        prediction_id=prediction_id,
        headline=headline,
        source_domain=source,
    ):
        ratings[key] = stored
    save_profile(profile)
    return summarize_profile(profile)


def get_rating_map() -> dict[str, int]:
    """Flat key → selected UI value for frontend hydration after page reload."""
    profile = load_profile()
    out: dict[str, Any] = {}
    for key, payload in (profile.get("user_ratings") or {}).items():
        try:
            selected = (payload or {}).get("selected_value")
            out[str(key)] = selected if selected not in (None, "") else int((payload or {}).get("rating", 0) or 0)
        except Exception:
            continue
    return out


def summarize_profile(profile: dict[str, Any] | None = None) -> dict[str, Any]:
    profile = profile or load_profile()

    def top(bucket: str, sign: int) -> list[dict[str, Any]]:
        rows = []
        for key, value in (profile.get(bucket) or {}).items():
            val = float(value or 0.0)
            if (sign > 0 and val > 0) or (sign < 0 and val < 0):
                rows.append({"key": key, "weight": round(val, 3)})
        rows.sort(key=lambda r: abs(r["weight"]), reverse=True)
        return rows[:8]

    return {
        "positive_sources": top("source_weights", 1),
        "negative_sources": top("source_weights", -1),
        "positive_keywords": top("keyword_weights", 1),
        "negative_keywords": top("keyword_weights", -1),
        "severity_up": top("severity_weights", 1),
        "severity_down": top("severity_weights", -1),
        "insignificant_count": sum(
            1 for e in (profile.get("events") or [])
            if str(e.get("selected_value", "")).upper() == "INSIGNIFICANT"
        ),
        "events": len(profile.get("events") or []),
    }


def article_interest_score(headline: str | None, source_domain: str | None) -> float:
    profile = load_profile()
    score = float((profile.get("source_weights") or {}).get((source_domain or "").lower(), 0.0) or 0.0)
    weights = profile.get("keyword_weights") or {}
    for word in _tokens(headline):
        score += float(weights.get(word, 0.0) or 0.0)
    return max(-3.0, min(3.0, score))


def severity_bias(headline: str | None, source_domain: str | None, asset: str | None = None) -> float:
    profile = load_profile()
    weights = profile.get("severity_weights") or {}
    score = float(weights.get((source_domain or "").lower(), 0.0) or 0.0)
    if asset:
        score += float(weights.get(f"asset:{asset.upper()}", 0.0) or 0.0)
    for word in _tokens(headline):
        score += float(weights.get(word, 0.0) or 0.0)
    return max(-3.0, min(3.0, score))


def adjust_severity(severity: str, bias: float) -> str:
    levels = SEVERITY_LEVELS
    sev = (severity or "NEUTRAL").upper()
    if sev not in levels:
        sev = "NEUTRAL"
    idx = levels.index(sev)
    if bias >= 0.55 and idx < len(levels) - 1:
        idx += 1
    elif bias <= -0.55 and idx > 0:
        idx -= 1
    return levels[idx]


def boosted_keywords() -> list[str]:
    """Keywords the user wants more of — merged into the RSS pre-filter."""
    profile = load_profile()
    weights = profile.get("keyword_weights") or {}
    boosted = [k for k, v in weights.items() if float(v or 0.0) > 0.15]
    boosted.sort(key=lambda k: float(weights.get(k, 0.0)), reverse=True)
    return boosted[:24]


def suppressed_keywords() -> list[str]:
    """Keywords the user thumbs-downed — used to deprioritize in the RSS pre-filter."""
    profile = load_profile()
    weights = profile.get("keyword_weights") or {}
    suppressed = [k for k, v in weights.items() if float(v or 0.0) < -0.15]
    suppressed.sort(key=lambda k: float(weights.get(k, 0.0)))
    return suppressed[:24]


def guidance_text() -> str:
    summary = summarize_profile()
    if not summary["events"]:
        return "No user feedback has been recorded yet."

    def fmt(rows: list[dict[str, Any]]) -> str:
        return ", ".join(f"{r['key']} ({r['weight']:+.2f})" for r in rows[:5]) or "none"

    return (
        "User feedback memory:\n"
        f"- Look for more news matching: {fmt(summary['positive_keywords'])}; sources: {fmt(summary['positive_sources'])}\n"
        f"- Deprioritize news matching: {fmt(summary['negative_keywords'])}; sources: {fmt(summary['negative_sources'])}\n"
        f"- Raise severity for similar items: {fmt(summary['severity_up'])}\n"
        f"- Lower severity for similar items: {fmt(summary['severity_down'])}"
    )
