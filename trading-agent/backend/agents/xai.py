"""Phase 13 — Explainable AI (XAI).

Builds a structured rationale dict for every order: indicators fired, news
triggers, edge / R:R / target / ETA, the risk floor applied, and a short
human-readable summary. Persisted on the Trade row (JSON column) and surfaced
in the audit page so every trade is auditable.
"""
from __future__ import annotations

from typing import Optional


def build_rationale(decision, extras: Optional[dict] = None) -> dict:
    """Build a JSON-serialisable rationale dict from a TradeDecision.

    `extras` may include `{recent_vol, last_close, correlation_notes, ...}`.
    Use `render(...)` for a multi-line readable text version.
    """
    extras = extras or {}

    direction = str(getattr(decision, "direction", "hold"))
    probs = dict(getattr(decision, "nn_probs", {}) or {})
    edge_mean = float(getattr(decision, "edge_mean", 0.0) or 0.0)
    edge_std = float(getattr(decision, "edge_std", 0.0) or 0.0)
    sl = float(getattr(decision, "sl", 0.0) or 0.0)
    tp = float(getattr(decision, "tp", 0.0) or 0.0)
    trail = float(getattr(decision, "trail", 0.0) or 0.0)
    target = float(getattr(decision, "target_price", 0.0) or 0.0)
    eta = float(getattr(decision, "expected_execution_ts", 0.0) or 0.0)
    size = float(getattr(decision, "size_pct", 0.0) or 0.0)
    regime = str(getattr(decision, "regime", "") or "")

    news_summary = None
    news = getattr(decision, "active_news", None)
    if news is not None:
        try:
            news_summary = {
                "severity": str(getattr(news, "severity", "")),
                "direction": str(getattr(news, "direction", "")),
                "confidence": float(getattr(news, "confidence", 0.0) or 0.0),
                "rationale": str(getattr(news, "rationale", ""))[:160],
                "source": str(getattr(news, "source_domain", "")),
            }
        except Exception:
            news_summary = {"summary": str(news)[:120]}

    rr = (tp / sl) if sl > 0 else 0.0
    rationale = {
        "direction": direction,
        "size_pct": round(size, 5),
        "probs": {k: round(float(v), 4) for k, v in probs.items()},
        "nn_confidence": round(float(getattr(decision, "nn_confidence", 0.0) or 0.0), 4),
        "edge_mean": round(edge_mean, 5),
        "edge_std": round(edge_std, 5),
        "sl_pct": round(sl, 5),
        "tp_pct": round(tp, 5),
        "trail_pct": round(trail, 5),
        "rr_ratio": round(rr, 3),
        "target_price": round(target, 6) if target else 0.0,
        "expected_execution_ts": eta,
        "regime": regime,
        "news": news_summary,
        "recent_vol": round(float(extras.get("recent_vol", 0.0) or 0.0), 6),
        "last_close": round(float(extras.get("last_close", 0.0) or 0.0), 6),
        "summary": _summary(direction, size, probs, edge_mean, edge_std, sl, tp, rr, target, regime, news_summary),
    }
    return rationale


def _summary(direction, size, probs, edge_mean, edge_std, sl, tp, rr, target, regime, news_summary) -> str:
    conf = (max(probs.values()) if probs else 0.0)
    parts = []
    if direction == "hold":
        parts.append(f"HOLD ({regime or 'no regime'})")
    else:
        parts.append(f"{direction.upper()} {size * 100:.1f}% of capital @ {conf * 100:.0f}% conf")
    parts.append(f"edge={edge_mean:+.3f}±{edge_std:.3f}")
    parts.append(f"SL {sl * 100:.2f}% / TP {tp * 100:.2f}% (R:R {rr:.2f})")
    if target:
        parts.append(f"target {target:.4f}")
    if regime:
        parts.append(f"regime={regime}")
    if news_summary:
        sev = news_summary.get("severity", "")
        src = news_summary.get("source", "")
        if sev or src:
            parts.append(f"news={sev}({src})")
    return " | ".join(parts)


def render(rationale_dict: Optional[dict]) -> str:
    """Render the structured rationale as a multi-line readable text block."""
    if not rationale_dict:
        return ""
    out = [rationale_dict.get("summary", "")]
    out.append(f"  direction       : {rationale_dict.get('direction')}")
    out.append(f"  size_pct        : {rationale_dict.get('size_pct')}")
    out.append(f"  nn_confidence   : {rationale_dict.get('nn_confidence')}")
    out.append(f"  probs           : {rationale_dict.get('probs')}")
    out.append(f"  edge            : {rationale_dict.get('edge_mean')} ± {rationale_dict.get('edge_std')}")
    out.append(f"  SL / TP / trail : {rationale_dict.get('sl_pct')} / {rationale_dict.get('tp_pct')} / {rationale_dict.get('trail_pct')}")
    out.append(f"  R:R             : {rationale_dict.get('rr_ratio')}")
    out.append(f"  target_price    : {rationale_dict.get('target_price')}")
    out.append(f"  expected_ts     : {rationale_dict.get('expected_execution_ts')}")
    out.append(f"  regime          : {rationale_dict.get('regime')}")
    out.append(f"  recent_vol      : {rationale_dict.get('recent_vol')}")
    if rationale_dict.get("news"):
        out.append(f"  news            : {rationale_dict['news']}")
    return "\n".join(out)


# ----------------------------------------------------------------- post-trade
def post_trade_error(target_price: float, exit_price: float, direction: str) -> dict:
    """Phase 13 post-trade error: signed deviation of realized exit from the
    forecast target. Positive `error_pct` means we overshot the target in the
    favourable direction; negative means we fell short / went the wrong way."""
    if not target_price or target_price <= 0:
        return {"error_pct": 0.0, "abs_error_pct": 0.0, "target_price": float(target_price or 0.0)}
    deviation = (float(exit_price) - float(target_price)) / float(target_price)
    signed = deviation if direction == "long" else -deviation
    return {
        "error_pct": round(float(signed), 6),
        "abs_error_pct": round(abs(float(deviation)), 6),
        "target_price": float(target_price),
        "exit_price": float(exit_price),
    }
