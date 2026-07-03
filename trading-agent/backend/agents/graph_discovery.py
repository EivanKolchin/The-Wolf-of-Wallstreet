"""LLM edge discovery for the entity graph — the LLM PROPOSES, the correlation validator DISPOSES.

The static curated edge list (BTC→MSTR, NVDA→TSM, …) goes stale as narratives rotate and misses
brand-new relationships (a fresh spot-BTC ETF should map into the crypto-proxy network without a
code change). This agent reads recent headlines and proposes CANDIDATE edges as structured JSON.

The safety property that makes this sound: a proposed edge enters the graph with only a small
PRIOR — its LIVE weight is still `prior × ramp(rolling return correlation)` (EntityGraph.refresh).
So the LLM can only NOMINATE a relationship; it earns weight solely by correlating in real data,
and a hallucinated or dead edge sits at weight 0 forever. The LLM cannot size a trade — it can
only put a candidate in front of the statistical validator. Degrades to a no-op without an LLM.

Persistence: discovered edges are appended to a JSON file so they survive restarts and accumulate;
duplicates (same src→dst) refresh their prior instead of stacking."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

try:
    import structlog
    logger = structlog.get_logger("graph_discovery")
except Exception:  # pragma: no cover
    import logging
    logger = logging.getLogger("graph_discovery")

from backend.signals.entity_graph import Edge, EntityGraph

# Destinations the LLM is allowed to map onto — the tradable universe. Anything else is dropped
# (the graph only matters for symbols we can act on).
DEFAULT_DST_WHITELIST = ("MSTR", "COIN", "TSLA", "PLTR", "NVDA", "AMD", "MU", "TSM", "SMCI",
                         "SNDK", "BE", "ETHUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT")
MAX_DISCOVERED_PRIOR = 0.5      # a DISCOVERED edge can never start stronger than a curated one


_PROMPT = """You map financial news to CROSS-ASSET price relationships for a trading system.
Given recent headlines, list asset-proxy relationships where a move in a SOURCE asset would
mechanically move a DESTINATION ticker (issuer/holdings proxy, supply chain, or same sector).

Return ONLY a JSON array, each item:
  {{"src": "<source asset symbol, e.g. BTC or NVDA>",
    "dst": "<destination ticker from this list: {dst}>",
    "kind": "proxy|supply_chain|sector",
    "confidence": <0.0-1.0>,
    "why": "<max 12 words>"}}
Only include relationships strongly implied by the headlines. Destinations MUST be from the list.
Empty array if none. No prose outside the JSON.

HEADLINES:
{headlines}"""


@dataclass
class DiscoveredEdge:
    src: str
    dst: str
    kind: str
    prior: float
    why: str = ""


def _parse_edges(raw: str, dst_whitelist) -> List[DiscoveredEdge]:
    """Tolerantly extract the JSON array from an LLM response and validate each edge."""
    m = re.search(r"\[.*\]", raw or "", re.DOTALL)
    if not m:
        return []
    try:
        items = json.loads(m.group(0))
    except (json.JSONDecodeError, ValueError):
        return []
    wl = {d.upper() for d in dst_whitelist}
    out: List[DiscoveredEdge] = []
    for it in items if isinstance(items, list) else []:
        try:
            src = str(it["src"]).upper().strip()
            dst = str(it["dst"]).upper().strip()
            kind = str(it.get("kind", "sector")).lower().strip()
            conf = float(it.get("confidence", 0.0))
        except (KeyError, TypeError, ValueError):
            continue
        if not src or dst not in wl or src == dst or kind not in ("proxy", "supply_chain", "sector"):
            continue
        prior = max(0.0, min(MAX_DISCOVERED_PRIOR, conf * MAX_DISCOVERED_PRIOR / 1.0))
        if prior <= 0:
            continue
        out.append(DiscoveredEdge(src, dst, kind, prior, str(it.get("why", ""))[:60]))
    return out


class GraphDiscoveryAgent:
    """Proposes candidate edges from headlines; merges them into an EntityGraph (weights still
    earned via correlation). LLM optional — no LLM → no-op."""

    def __init__(self, llm_service=None, *, store_path: Optional[str] = None,
                 dst_whitelist=DEFAULT_DST_WHITELIST):
        self.llm = llm_service
        self.store_path = Path(store_path or "training_data/discovered_edges.json")
        self.dst_whitelist = tuple(dst_whitelist)

    # ----------------------------------------------------------- persistence
    def _load(self) -> List[DiscoveredEdge]:
        if not self.store_path.exists():
            return []
        try:
            rows = json.loads(self.store_path.read_text(encoding="utf-8"))
            return [DiscoveredEdge(**r) for r in rows]
        except Exception:
            return []

    def _save(self, edges: List[DiscoveredEdge]) -> None:
        try:
            self.store_path.parent.mkdir(parents=True, exist_ok=True)
            self.store_path.write_text(
                json.dumps([e.__dict__ for e in edges], indent=2), encoding="utf-8")
        except OSError as e:
            logger.warning("discovered_edges_save_failed", error=str(e)[:100])

    def _merge(self, existing: List[DiscoveredEdge],
               new: List[DiscoveredEdge]) -> List[DiscoveredEdge]:
        by_key = {(e.src, e.dst): e for e in existing}
        for e in new:
            by_key[(e.src, e.dst)] = e          # refresh prior/kind (dedupe, don't stack)
        return list(by_key.values())

    # ----------------------------------------------------------- discovery
    async def discover(self, headlines: List[str]) -> List[DiscoveredEdge]:
        """Ask the LLM for candidate edges from headlines; persist the merged set. Returns the
        newly proposed edges (possibly empty). No LLM / any error → []."""
        if self.llm is None or not headlines:
            return []
        prompt = _PROMPT.format(dst=", ".join(self.dst_whitelist),
                                headlines="\n".join(f"- {h}" for h in headlines[:40]))
        try:
            raw = await self.llm.generate_text(prompt, tier="haiku", max_tokens=500, json_mode=True)
        except Exception as e:
            logger.warning("graph_discovery_llm_failed", error=str(e)[:120])
            return []
        proposed = _parse_edges(raw, self.dst_whitelist)
        if not proposed:
            return []
        merged = self._merge(self._load(), proposed)
        self._save(merged)
        logger.info("graph_edges_discovered", n_new=len(proposed), n_total=len(merged))
        return proposed

    def apply_to(self, graph: EntityGraph) -> int:
        """Add all persisted discovered edges to a live EntityGraph (as priors — the graph's
        correlation validator still decides their live weight). Returns the count added."""
        existing = {(e.src.upper(), e.dst.upper()) for e in graph.edges}
        added = 0
        for de in self._load():
            key = (de.src.upper(), de.dst.upper())
            if key in existing:
                continue
            graph.edges.append(Edge(de.src, de.dst, de.kind, de.prior))
            existing.add(key)
            added += 1
        if added:
            logger.info("discovered_edges_applied", n=added)
        return added
