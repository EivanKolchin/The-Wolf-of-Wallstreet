"""Lightweight web search for the market copilot (no API key required).

Uses DuckDuckGo HTML + instant-answer API as fallbacks so the chat agent can
answer questions that are not present in local portfolio/market state.
"""
from __future__ import annotations

import html
import json
import re
import urllib.parse
import urllib.request
from typing import Any


def search_web(query: str, max_results: int = 5) -> list[dict[str, str]]:
    query = (query or "").strip()
    if not query:
        return []
    results: list[dict[str, str]] = []
    results.extend(_ddg_instant_answer(query))
    if len(results) < max_results:
        results.extend(_ddg_html_search(query, max_results - len(results)))
    # De-dupe by snippet prefix
    seen: set[str] = set()
    out: list[dict[str, str]] = []
    for row in results:
        key = (row.get("title") or "")[:80]
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
        if len(out) >= max_results:
            break
    return out


def _ddg_instant_answer(query: str) -> list[dict[str, str]]:
    try:
        qs = urllib.parse.urlencode({"q": query, "format": "json", "no_redirect": 1, "skip_disambig": 1})
        url = f"https://api.duckduckgo.com/?{qs}"
        req = urllib.request.Request(url, headers={"User-Agent": "TradingAgentCopilot/1.0"})
        with urllib.request.urlopen(req, timeout=6) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="ignore"))
        rows: list[dict[str, str]] = []
        abstract = (data or {}).get("AbstractText") or ""
        if abstract:
            rows.append({
                "title": (data.get("Heading") or query)[:200],
                "snippet": abstract[:600],
                "url": (data.get("AbstractURL") or "")[:500],
                "source": "duckduckgo_instant",
            })
        for topic in (data or {}).get("RelatedTopics") or []:
            if isinstance(topic, dict) and topic.get("Text"):
                rows.append({
                    "title": (topic.get("Text") or "")[:120],
                    "snippet": (topic.get("Text") or "")[:600],
                    "url": (topic.get("FirstURL") or "")[:500],
                    "source": "duckduckgo_related",
                })
            if len(rows) >= 3:
                break
        return rows
    except Exception:
        return []


def _ddg_html_search(query: str, max_results: int) -> list[dict[str, str]]:
    try:
        body = urllib.parse.urlencode({"q": query}).encode("utf-8")
        req = urllib.request.Request(
            "https://html.duckduckgo.com/html/",
            data=body,
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; TradingAgentCopilot/1.0)",
                "Content-Type": "application/x-www-form-urlencoded",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=8) as resp:
            text = resp.read().decode("utf-8", errors="ignore")
        rows: list[dict[str, str]] = []
        for m in re.finditer(
            r'class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
            text,
            re.I | re.S,
        ):
            href = html.unescape(m.group(1))
            title = re.sub(r"<[^>]+>", "", m.group(2))
            title = html.unescape(title).strip()
            if not title or "duckduckgo.com/y.js" in href:
                continue
            rows.append({"title": title[:200], "snippet": "", "url": href[:500], "source": "duckduckgo_html"})
            if len(rows) >= max_results:
                break
        # Attach snippets when the simplified layout includes them.
        snippets = re.findall(r'class="result__snippet"[^>]*>(.*?)</(?:a|td|div)>', text, re.I | re.S)
        for i, snip in enumerate(snippets):
            if i >= len(rows):
                break
            clean = re.sub(r"<[^>]+>", "", snip)
            rows[i]["snippet"] = html.unescape(clean).strip()[:600]
        return rows
    except Exception:
        return []


def build_search_query(message: str, context: dict[str, Any] | None = None) -> str | None:
    """Heuristic: return a search query when local context is unlikely to suffice."""
    msg = (message or "").strip()
    if not msg:
        return None
    lower = msg.lower()
    context = context or {}

    # Explicit external-fact questions
    external_triggers = (
        "when does", "when is", "earnings", "report date", "ipo",
        "latest news", "what happened", "why isn't", "why is",
        "not working", "search for", "look up", "google",
        "who is", "what is the price of", "current news",
    )
    if any(t in lower for t in external_triggers):
        return msg

    symbols = list(context.get("requested_symbols") or [])
    earnings = context.get("earnings") or {}
    if symbols and any(k in lower for k in ("earnings", "report", "when")):
        missing = [s for s in symbols if not (earnings.get(s) or {}).get("next")]
        if missing:
            return f"{missing[0]} next earnings report date"

    market = context.get("market") or []
    if symbols and not market and any(k in lower for k in ("price", "trading at", "worth")):
        return f"{symbols[0]} stock price today"

    agents = context.get("agents") or {}
    if any(k in lower for k in ("why", "broken", "stuck", "offline", "not working")):
        if not agents.get("nn_trading_agent_alive") or not agents.get("llm_news_agent_alive"):
            return f"troubleshoot {msg}"

    mode = context.get("mode") or {}
    if "paper trading" in lower or "live trading" in lower:
        if mode.get("paper_trading") is None:
            return msg

    return None
