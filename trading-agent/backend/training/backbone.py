from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np
from backend.core.config import settings

# Keyword bank backbone (editable by user for tuning)
CRYPTO_KEYWORD_BANK: dict[str, list[str]] = {
    "BTCUSDT": [
        "bitcoin", "btc", "etf", "blackrock", "fidelity", "miner", "hashrate",
        "halving", "treasury", "michael saylor", "microstrategy", "mt gox"
    ],
    "ETHUSDT": [
        "ethereum", "eth", "gas fees", "layer 2", "l2", "rollup", "eip",
        "staking", "validator", "defi", "uniswap", "arbitrum", "optimism"
    ],
    "SOLUSDT": [
        "solana", "sol", "validator outage", "jito", "memecoin", "firedancer"
    ],
    "XRPUSDT": [
        "xrp", "ripple", "sec lawsuit", "xrp ledger", "on-demand liquidity"
    ],
    "ADAUSDT": [
        "cardano", "ada", "charles hoskinson", "hydra", "midnight"
    ],
    "DOGEUSDT": [
        "dogecoin", "doge", "elon musk", "x payments", "meme coin"
    ],
    "AAVEUSDT": [
        "aave", "lending protocol", "collateral", "liquidation", "governance vote"
    ],
    "XLMUSDT": [
        "stellar", "xlm", "stellar development foundation", "cross-border payments"
    ],
    "RENDERUSDT": [
        "render network", "rndr", "render token", "gpu rendering", "octane"
    ],
    "NEARUSDT": [
        "near protocol", "near foundation", "nightshade", "sharding", "aurora near"
    ],
}

# Stock keyword bank — company name + distinctive ticker + key people/products/sector terms.
# Matching is SUBSTRING (see extract_symbol_relevance), so ambiguous 2-letter tokens ("be", "mu",
# bare "coin") are DELIBERATELY excluded — they'd false-match ("before", "much", "bitcoin").
STOCK_KEYWORD_BANK: dict[str, list[str]] = {
    "NVDA": ["nvidia", "nvda", "jensen huang", "blackwell", "h100", "h200", "cuda",
             "ai chip", "data center gpu"],
    "AMD":  ["advanced micro devices", "amd", "lisa su", "ryzen", "epyc", "radeon",
             "mi300", "instinct"],
    "MU":   ["micron", "dram", "hbm", "nand flash", "memory chips"],
    "TSM":  ["tsmc", "taiwan semiconductor", "chip foundry", "3nm", "2nm", "arizona fab"],
    "SMCI": ["supermicro", "super micro", "smci", "ai server", "liquid cooling"],
    "SNDK": ["sandisk", "sndk", "flash storage", "solid state drive"],
    "BE":   ["bloom energy", "fuel cell", "solid oxide", "hydrogen power"],
    "TSLA": ["tesla", "tsla", "elon musk", "cybertruck", "model 3", "model y",
             "full self-driving", "robotaxi", "gigafactory", "optimus"],
    "MSTR": ["microstrategy", "mstr", "michael saylor", "bitcoin treasury", "strategy inc"],
    "COIN": ["coinbase", "brian armstrong", "crypto exchange", "base chain"],
    "PLTR": ["palantir", "pltr", "alex karp", "foundry platform", "gotham", "aip"],
    "GOOGL": ["alphabet", "google", "googl", "sundar pichai", "gemini ai", "deepmind",
              "waymo", "google cloud", "android"],
    "MSFT": ["microsoft", "msft", "satya nadella", "azure", "copilot", "openai",
             "windows", "xbox"],
    "RKLB": ["rocket lab", "rklb", "peter beck", "electron rocket", "neutron rocket",
             "space launch"],
    "RGTI": ["rigetti", "rgti", "quantum computing", "qubit", "quantum processor",
             "superconducting qubit"],
}

# The combined bank the news pipeline actually searches (crypto + stocks).
KEYWORD_BANK: dict[str, list[str]] = {**CRYPTO_KEYWORD_BANK, **STOCK_KEYWORD_BANK}


def map_asset_to_symbol(asset: str | None) -> str | None:
    if not asset:
        return None
    normalized = asset.upper().replace("-", "").replace("/", "")
    aliases = {
        "BTCUSD": "BTCUSDT", "BTCUSDT": "BTCUSDT",
        "ETHUSD": "ETHUSDT", "ETHUSDT": "ETHUSDT",
        "SOLUSD": "SOLUSDT", "XRPUSD": "XRPUSDT",
        "ADAUSD": "ADAUSDT", "DOGEUSD": "DOGEUSDT",
        "AAVEUSD": "AAVEUSDT", "XLMUSD": "XLMUSDT",
        "RENDERUSD": "RENDERUSDT", "RNDRUSD": "RENDERUSDT", "RNDR": "RENDERUSDT",
        "NEARUSD": "NEARUSDT",
    }
    if normalized in aliases:
        return aliases[normalized]
    # Stocks map to themselves (the LLM often declares the bare ticker).
    from backend.core.universe import STOCK_UNDERLYINGS
    if normalized in STOCK_UNDERLYINGS:
        return normalized
    return None


def extract_symbol_relevance(
    text: str,
    keyword_bank: dict[str, list[str]] | None = None,
    weights: dict | None = None,
) -> tuple[dict[str, float], dict[str, list[str]]]:
    """Per-symbol relevance from keyword hits. If `weights` (a {(symbol, keyword_lower):
    weight} map from the learnable KeywordWeight table) is provided, hits are weighted so
    the bank adapts over time; otherwise falls back to a flat hit count."""
    bank = keyword_bank or KEYWORD_BANK      # crypto + stocks (was crypto-only)
    lowered = text.lower()
    scores: dict[str, float] = {}
    matches: dict[str, list[str]] = {}

    for symbol, keywords in bank.items():
        hit_terms = [kw for kw in keywords if kw.lower() in lowered]
        if not hit_terms:
            continue
        denom = max(4.0, float(len(keywords)))
        if weights:
            wsum = sum(float(weights.get((symbol, kw.lower()), 1.0)) for kw in hit_terms)
            score = min(1.0, wsum / denom)
        else:
            score = min(1.0, len(hit_terms) / denom)
        scores[symbol] = float(score)
        matches[symbol] = hit_terms

    return scores, matches


class TrainingBackbone:
    """Simple JSONL recorder for decisions/outcomes to bootstrap offline training datasets."""

    def __init__(self, output_dir: str = "training_data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.decisions_path = self.output_dir / "decision_log.jsonl"
        self.outcomes_path = self.output_dir / "outcome_log.jsonl"
        self._lock = Lock()

    def _write_jsonl(self, path: Path, payload: dict[str, Any]) -> None:
        line = json.dumps(payload, ensure_ascii=False)
        with self._lock:
            with path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")

    def record_decision(
        self,
        symbol: str,
        sequence: np.ndarray,
        decision: str,
        size_pct: float,
        probs: dict[str, float],
        regime: str,
        approved: bool,
        reason: str,
        news_impact: Any = None,
    ) -> None:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "feature_schema_version": settings.FEATURE_SCHEMA_VERSION,
            "symbol": symbol,
            "decision": decision,
            "size_pct": float(size_pct),
            "probs": probs,
            "regime": regime,
            "approved": bool(approved),
            "reason": reason,
            "news_impact": asdict(news_impact) if news_impact else None,
            "sequence_length": int(sequence.shape[0]),
            "feature_count": int(sequence.shape[1]) if sequence.ndim == 2 else 0,
            "features": sequence.tolist(),
        }
        self._write_jsonl(self.decisions_path, payload)

    def record_outcome(self, symbol: str, pnl_pct: float, trade_id: str | None = None) -> None:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbol": symbol,
            "trade_id": trade_id,
            "pnl_pct": float(pnl_pct),
        }
        self._write_jsonl(self.outcomes_path, payload)