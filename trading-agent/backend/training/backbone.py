from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np

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
}


def map_asset_to_symbol(asset: str | None) -> str | None:
    if not asset:
        return None
    normalized = asset.upper().replace("-", "").replace("/", "")
    aliases = {
        "BTCUSD": "BTCUSDT",
        "BTCUSDT": "BTCUSDT",
        "ETHUSD": "ETHUSDT",
        "ETHUSDT": "ETHUSDT",
        "SOLUSD": "SOLUSDT",
        "XRPUSD": "XRPUSDT",
        "ADAUSD": "ADAUSDT",
        "DOGEUSD": "DOGEUSDT",
        "AAVEUSD": "AAVEUSDT",
        "XLMUSD": "XLMUSDT",
    }
    return aliases.get(normalized)


def extract_symbol_relevance(text: str, keyword_bank: dict[str, list[str]] | None = None) -> tuple[dict[str, float], dict[str, list[str]]]:
    bank = keyword_bank or CRYPTO_KEYWORD_BANK
    lowered = text.lower()
    scores: dict[str, float] = {}
    matches: dict[str, list[str]] = {}

    for symbol, keywords in bank.items():
        hit_terms = [kw for kw in keywords if kw.lower() in lowered]
        if not hit_terms:
            continue
        # simple normalized hit score; clipped to 1.0
        score = min(1.0, len(hit_terms) / max(4, len(keywords)))
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