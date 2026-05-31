"""Phase 3 tests: the online self-labeling news accumulator.

Verifies that matured pending records are realized into JSONL training samples
with the correct forward-return label, and that not-yet-due records are kept.
"""
import json
import sys
import types
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))


def _make_stub():
    from backend.agents.nn_agent import NNTradingAgent  # noqa
    stub = types.SimpleNamespace()
    stub._pending_news_labels = deque(maxlen=5000)
    stub._last_price = {}
    return stub, NNTradingAgent


def test_drain_realizes_matured_label(tmp_path, monkeypatch):
    from backend.core.config import settings
    stub, Agent = _make_stub()

    log_path = tmp_path / "news_labels.jsonl"
    monkeypatch.setattr(settings, "NN_NEWS_LABEL_LOG", str(log_path), raising=False)
    monkeypatch.setattr(settings, "NN_NEWS_LABEL_HORIZON_MIN", 60, raising=False)

    past = datetime.now(timezone.utc).timestamp() - 10  # already due
    stub._pending_news_labels.append({
        "symbol": "BTCUSDT",
        "embedding": [0.1] * 16,
        "price_then": 100.0,
        "logged_at": "2026-01-01T00:00:00+00:00",
        "due_at": past,
    })
    stub._last_price["BTCUSDT"] = 105.0  # +5% forward return

    Agent._drain_matured_news_labels(stub)

    assert log_path.exists()
    rec = json.loads(log_path.read_text(encoding="utf-8").strip())
    assert rec["symbol"] == "BTCUSDT"
    assert abs(rec["forward_return"] - 0.05) < 1e-9
    assert rec["label"] == 0          # positive forward return -> "long"
    assert len(stub._pending_news_labels) == 0


def test_drain_keeps_not_yet_due(tmp_path, monkeypatch):
    from backend.core.config import settings
    stub, Agent = _make_stub()
    monkeypatch.setattr(settings, "NN_NEWS_LABEL_LOG", str(tmp_path / "nl.jsonl"), raising=False)

    future = datetime.now(timezone.utc).timestamp() + 3600
    stub._pending_news_labels.append({
        "symbol": "ETHUSDT", "embedding": [0.0] * 16,
        "price_then": 50.0, "logged_at": "x", "due_at": future,
    })
    stub._last_price["ETHUSDT"] = 60.0

    Agent._drain_matured_news_labels(stub)
    # still pending, nothing written
    assert len(stub._pending_news_labels) == 1
    assert not (tmp_path / "nl.jsonl").exists()
