"""Directional news → risk action — the architecturally-honest way news enters the system.

The net never learned from news (the offline training data stubs all 20 news slots to zero — see
feature_spec._STUB_OFFLINE), and rare fat-tailed events can't be learned from sparse history anyway.
So news does NOT predict direction here; it acts as a RISK OVERLAY that can veto, de-gear, halt, or
(cautiously) upsize a trade the primary engine already proposed.

The decision is DIRECTIONAL and POSITION-AWARE — this is the whole point:
  * SEVERE news AGAINST the position (e.g. a long into a hack/regulatory-ban headline) → veto, and
    for a high-magnitude outlier, HALT that asset for the news window (don't keep re-entering a knife).
  * SEVERE/ SIGNIFICANT news WITH the position → allow, and optionally upsize a little (capped) when
    it's high-confidence and near-term ("very bullish soon").
  * SIGNIFICANT against → de-gear (shrink size) rather than fully veto.
  * No / neutral / stale news → pass through unchanged (size_scale 1.0).

Pure + deterministic so it is unit-testable with plain fakes (it duck-types the NewsImpact fields,
so it never imports redis). An optional ``llm_verifier`` can be injected to add LLM judgement, but it
may only make the action MORE conservative (tighten size / escalate to veto/halt) — never looser —
so an LLM hallucination can never upsize the book into a loss. Nothing here executes a trade.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, List, Optional, Sequence

# severity → base weight (how much this class of news is allowed to move risk)
SEVERITY_WEIGHT = {"SEVERE": 1.0, "SIGNIFICANT": 0.5, "MILD": 0.2, "NEUTRAL": 0.0}

_UP = {"up", "bull", "bullish", "long", "positive", "buy", "pump"}
_DOWN = {"down", "bear", "bearish", "short", "negative", "sell", "dump"}
_LONG = {"long", "buy", "up", "bull", "bullish"}
_SHORT = {"short", "sell", "down", "bear", "bearish"}


@dataclass
class NewsOverlayConfig:
    """All thresholds in one place (operator-tunable, conservative defaults)."""
    significant_degear_k: float = 0.8     # size_scale = 1 - k·|opposing score| for SIGNIFICANT-against
    degear_floor: float = 0.2             # never shrink below this (still let a tiny position through)
    veto_score: float = 0.55              # opposing signed-score at/above this → veto (size 0)
    halt_magnitude_pct: float = 8.0       # a SEVERE-against event of at least this expected move → halt asset
    allow_upsize: bool = True             # cautiously upsize on aligned news ("bullish soon")
    upsize_k: float = 0.3                 # size_scale = 1 + k·aligned score …
    upsize_cap: float = 1.25              # … capped here (upsizing is the risky direction → small cap)
    upsize_min_confidence: float = 0.7    # only upsize on high-confidence news …
    upsize_max_horizon_min: float = 240.0 # … that is near-term ("soon")
    stale_after_min: float = 360.0        # news older than its window + this is ignored


@dataclass
class NewsRiskAction:
    action: str            # "allow" | "resize" | "veto" | "halt_asset"
    size_scale: float      # multiply the engine's intended size by this (0 = no trade)
    halt_seconds: float    # >0 → caller should suppress NEW trades on this asset this long
    signed_score: float    # net news pressure: >0 bullish, <0 bearish (asset-frame, not trade-frame)
    reason: str

    @property
    def blocks_trade(self) -> bool:
        return self.action in ("veto", "halt_asset") or self.size_scale <= 0.0


def _sign_from(text: str, up: set, down: set) -> int:
    t = (text or "").strip().lower()
    if t in up:
        return 1
    if t in down:
        return -1
    return 0


def _age_minutes(created_at: Any, now: datetime) -> float:
    """Minutes since the impact was created; 0 if unparseable (treat as fresh)."""
    if not created_at:
        return 0.0
    try:
        ts = datetime.fromisoformat(str(created_at).replace("Z", "+00:00"))
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return max(0.0, (now - ts).total_seconds() / 60.0)
    except Exception:
        return 0.0


def _impact_components(impact: Any, asset: str, now: datetime, cfg: NewsOverlayConfig):
    """Return (signed_score, severity, magnitude_pct, horizon_min, confidence) for one impact, or
    None if it is irrelevant/stale/neutral. signed_score is in the ASSET frame (+bullish/−bearish)."""
    severity = str(getattr(impact, "severity", "NEUTRAL") or "NEUTRAL").upper()
    sev_w = SEVERITY_WEIGHT.get(severity, 0.0)
    if sev_w <= 0.0:
        return None
    sign = _sign_from(getattr(impact, "direction", ""), _UP, _DOWN)
    if sign == 0:
        return None
    conf = max(0.0, min(1.0, float(getattr(impact, "confidence", 0.0) or 0.0)))
    trust = max(0.0, min(1.0, float(getattr(impact, "trust_score", 0.0) or 0.0)))
    # per-asset relevance (default 1.0 when the impact is explicitly about this asset)
    rel_map = getattr(impact, "symbol_relevance", None)
    if isinstance(rel_map, dict) and rel_map:
        rel = float(rel_map.get(asset, rel_map.get(str(asset).upper(), 0.0)) or 0.0)
    else:
        rel = 1.0 if str(getattr(impact, "asset", "")).upper() == str(asset).upper() else 0.0
    if rel <= 0.0:
        return None
    lo = float(getattr(impact, "magnitude_pct_low", 0.0) or 0.0)
    hi = float(getattr(impact, "magnitude_pct_high", 0.0) or 0.0)
    magnitude = max(lo, hi)
    horizon = float(getattr(impact, "t_max_minutes", 0) or 0.0)
    # recency: full weight inside the event window, linear decay over stale_after_min past it
    age = _age_minutes(getattr(impact, "created_at", None), now)
    past = max(0.0, age - horizon)
    recency = max(0.0, 1.0 - past / max(1e-6, cfg.stale_after_min))
    if recency <= 0.0:
        return None
    score = sign * sev_w * conf * trust * rel * recency
    return score, severity, magnitude, horizon, conf


def assess_news_risk(symbol: str, trade_direction: str, impacts: Sequence[Any], *,
                     now: Optional[datetime] = None, cfg: Optional[NewsOverlayConfig] = None,
                     llm_verifier: Optional[Callable[..., Optional["NewsRiskAction"]]] = None,
                     ) -> NewsRiskAction:
    """Decide what the news means for a proposed trade on ``symbol``.

    ``trade_direction``: "long"/"short" (or "hold"). ``impacts``: NewsImpact-like objects (duck-typed).
    Returns a NewsRiskAction. Deterministic; the optional ``llm_verifier(symbol, trade_direction,
    impacts, base_action) -> NewsRiskAction|None`` may only TIGHTEN the result (never loosen)."""
    cfg = cfg or NewsOverlayConfig()
    now = now or datetime.now(timezone.utc)
    trade_sign = _sign_from(trade_direction, _LONG, _SHORT)

    comps = [c for c in (_impact_components(i, symbol, now, cfg) for i in (impacts or [])) if c]
    if not comps:
        return NewsRiskAction("allow", 1.0, 0.0, 0.0, "no relevant news")

    signed = sum(c[0] for c in comps)                       # net asset-frame pressure
    signed = max(-1.5, min(1.5, signed))                    # bound the aggregate
    # the worst SEVERE event opposing the trade drives halt/veto
    severe_against = [c for c in comps
                      if c[1] == "SEVERE" and trade_sign != 0 and (c[0] * trade_sign) < 0]
    worst_mag = max((c[2] for c in severe_against), default=0.0)
    worst_hzn = max((c[4 - 1] for c in severe_against), default=0.0)  # horizon of severe-against

    # alignment of net news vs the trade
    aligned = (signed * trade_sign) if trade_sign != 0 else -abs(signed)  # hold: any news is "against"
    against_mag = abs(min(0.0, aligned))                     # how strongly news opposes the trade
    with_mag = max(0.0, aligned)                             # how strongly news supports the trade

    base: NewsRiskAction
    if severe_against and worst_mag >= cfg.halt_magnitude_pct:
        halt_s = max(60.0, worst_hzn * 60.0)
        base = NewsRiskAction("halt_asset", 0.0, halt_s, signed,
                              f"SEVERE news against {trade_direction} (~{worst_mag:.0f}% move) "
                              f"→ halt {symbol} {halt_s/60:.0f}m")
    elif against_mag >= cfg.veto_score:
        base = NewsRiskAction("veto", 0.0, 0.0, signed,
                              f"news opposes {trade_direction} (score {against_mag:.2f}) → veto")
    elif against_mag > 0.0:
        scale = max(cfg.degear_floor, 1.0 - cfg.significant_degear_k * against_mag)
        base = NewsRiskAction("resize", scale, 0.0, signed,
                              f"news leans against {trade_direction} → de-gear ×{scale:.2f}")
    elif (with_mag > 0.0 and cfg.allow_upsize and trade_sign != 0):
        near_term = worst_hzn  # reuse: but compute aligned horizon/conf below
        # only upsize on high-confidence, near-term aligned news
        aligned_comps = [c for c in comps if (c[0] * trade_sign) > 0]
        best_conf = max((c[4] for c in aligned_comps), default=0.0)
        best_hzn = min((c[3] for c in aligned_comps if c[3] > 0), default=1e9)
        if best_conf >= cfg.upsize_min_confidence and best_hzn <= cfg.upsize_max_horizon_min:
            scale = min(cfg.upsize_cap, 1.0 + cfg.upsize_k * with_mag)
            base = NewsRiskAction("resize", scale, 0.0, signed,
                                  f"news supports {trade_direction} (conf {best_conf:.2f}, "
                                  f"{best_hzn:.0f}m) → upsize ×{scale:.2f}")
        else:
            base = NewsRiskAction("allow", 1.0, 0.0, signed, "aligned news, but not strong/near enough to upsize")
    else:
        base = NewsRiskAction("allow", 1.0, 0.0, signed, "news net-neutral for this trade")

    if llm_verifier is not None:
        try:
            override = llm_verifier(symbol, trade_direction, list(impacts), base)
        except Exception:
            override = None
        if override is not None:
            base = _tighten_only(base, override)
    return base


def _tighten_only(base: NewsRiskAction, override: NewsRiskAction) -> NewsRiskAction:
    """Apply an LLM override but ONLY if it is more conservative than the deterministic action — a
    hard safety rail so the LLM can add caution (veto/halt/smaller size) but never enlarge risk."""
    rank = {"allow": 0, "resize": 1, "veto": 2, "halt_asset": 3}
    base_block = base.blocks_trade
    over_block = override.blocks_trade
    if over_block and not base_block:
        return override                                     # escalating to veto/halt is allowed
    if not over_block and base_block:
        return base                                         # LLM tried to UN-block → refuse
    # both same block-status: keep the smaller size, the longer halt, the higher action rank
    return NewsRiskAction(
        action=base.action if rank[base.action] >= rank[override.action] else override.action,
        size_scale=min(base.size_scale, override.size_scale),
        halt_seconds=max(base.halt_seconds, override.halt_seconds),
        signed_score=base.signed_score,
        reason=f"{base.reason} | llm: {override.reason}",
    )
