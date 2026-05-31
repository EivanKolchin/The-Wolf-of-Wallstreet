"""Phase 13 tests: XAI rationale + post-trade error feedback + ledger auto-error."""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from backend.agents.xai import build_rationale, render, post_trade_error
from backend.core import ledger


# ---------------------------------------------------------------- build_rationale
def _fake_decision(**kw):
    base = dict(
        direction="long", size_pct=0.1, nn_confidence=0.65,
        nn_probs={"long": 0.65, "short": 0.20, "hold": 0.15},
        edge_mean=0.12, edge_std=0.04, sl=0.02, tp=0.04, trail=0.02,
        target_price=104.0, expected_execution_ts=1719000000.0,
        regime="ranging", active_news=None,
    )
    base.update(kw)
    return SimpleNamespace(**base)


def test_build_rationale_long_has_expected_fields_and_summary():
    r = build_rationale(_fake_decision(), extras={"recent_vol": 0.008, "last_close": 100.0})
    for key in ("direction", "size_pct", "probs", "edge_mean", "edge_std",
                "sl_pct", "tp_pct", "rr_ratio", "target_price",
                "expected_execution_ts", "regime", "recent_vol", "last_close", "summary"):
        assert key in r
    assert r["direction"] == "long"
    assert r["rr_ratio"] == pytest.approx(2.0)         # tp 0.04 / sl 0.02
    assert "LONG" in r["summary"] and "R:R" in r["summary"]


def test_build_rationale_hold_summary():
    r = build_rationale(_fake_decision(direction="hold"))
    assert r["summary"].startswith("HOLD")


def test_render_produces_multiline_text():
    r = build_rationale(_fake_decision())
    txt = render(r)
    assert "\n" in txt and "direction" in txt and "SL" in txt


def test_render_empty_dict_safe():
    assert render(None) == "" and render({}) == ""


# ---------------------------------------------------------------- post-trade error
def test_post_trade_error_long_favourable():
    e = post_trade_error(target_price=100.0, exit_price=102.0, direction="long")
    assert e["error_pct"] == pytest.approx(0.02)
    assert e["abs_error_pct"] == pytest.approx(0.02)


def test_post_trade_error_short_favourable():
    e = post_trade_error(target_price=100.0, exit_price=98.0, direction="short")
    assert e["error_pct"] == pytest.approx(0.02)       # favourable for short -> positive
    assert e["abs_error_pct"] == pytest.approx(0.02)


def test_post_trade_error_zero_target_safe():
    e = post_trade_error(target_price=0.0, exit_price=100.0, direction="long")
    assert e["error_pct"] == 0.0 and e["abs_error_pct"] == 0.0


# ---------------------------------------------------------------- ledger auto-error
def test_ledger_auto_computes_error_fields(tmp_path, monkeypatch):
    # Redirect the statements dir to a temp folder.
    monkeypatch.setattr(ledger, "STATEMENTS_DIR", tmp_path)
    monkeypatch.setattr(ledger, "TRANSACTIONS_CSV", tmp_path / "transactions.csv")
    monkeypatch.setattr(ledger, "TRANSACTIONS_JSONL", tmp_path / "transactions.jsonl")

    ledger.record_transaction({
        "trade_id": "t1", "symbol": "AMD", "direction": "long",
        "entry_price": 100.0, "exit_price": 105.0, "target_price": 104.0,
        "pnl_usd": 5.0, "pnl_pct": 0.05, "fee_paid": 0.1,
        "broker": "alpaca", "asset_class": "us_stock", "quote_asset": "USD",
        "exit_reason": "take_profit", "paper": True, "size_usd": 100.0,
    })

    csv_path = tmp_path / "transactions.csv"
    body = csv_path.read_text(encoding="utf-8")
    # auto-computed: (105-104)/104 = +0.00961... (long, favourable)
    assert "error_pct" in body
    assert "abs_error_pct" in body
    # The row should have a non-zero error_pct
    import csv as _csv
    with csv_path.open() as f:
        rows = list(_csv.DictReader(f))
    assert rows and float(rows[-1]["error_pct"]) > 0.0
