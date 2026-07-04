"""Trade journal, post-mortem classification, anomaly scanner, GBM arm — the 2026-07-04 builds.

Design invariant under test throughout: the journal RECORDS but never decides; the post-mortem
learns from AGGREGATES with noise explicitly excluded (per-loss mutation is the measured
online-AWR failure mode); the scanner de-gears only on LIQUIDITY flags and fails open."""
import asyncio
import sys
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from backend.agents.trade_journal import TradeJournal  # noqa: E402
from backend.signals.anomaly_scanner import (  # noqa: E402
    ScannerConfig, Flag, classify_anomalies, degear_from_flags,
)


# ─────────────────────────── trade journal ───────────────────────────
def test_journal_roundtrip_and_filters(tmp_path):
    j = TradeJournal(path=tmp_path / "j.jsonl")
    j.record("note", msg="hello")
    j.record_rebalance({"orders": [{"symbol": "BTCUSDT", "target_weight": 0.1, "delta": 0.1}],
                        "skipped": ["SPY"], "targets": {"BTCUSDT": 0.1}, "news": {}, "overlay": {},
                        "blocked": None},
                       equity=100_000.0, prices={"BTCUSDT": 60_000.0})
    j.record_mark(equity=100_500.0, book_return=0.005, drawdown=0.0,
                  contributions={"BTCUSDT": 0.005}, weights={"BTCUSDT": 0.1})
    rows = j.read()
    kinds = [r["kind"] for r in rows]
    assert kinds == ["note", "rebalance", "order", "mark", "outcome"]
    order = next(r for r in rows if r["kind"] == "order")
    assert order["symbol"] == "BTCUSDT" and order["decision_price"] == 60_000.0
    assert order["notional"] == pytest.approx(10_000.0)
    # kind + time filters
    assert len(j.read(kinds=["mark"])) == 1
    assert j.read(since_ts=time.time() + 10) == []


def test_journal_tolerates_torn_line(tmp_path):
    p = tmp_path / "j.jsonl"
    j = TradeJournal(path=p)
    j.record("note", msg="ok")
    with p.open("a", encoding="utf-8") as f:
        f.write('{"ts": 1, "kind": "mark", TRUNCATED')     # crash mid-write
    assert len(j.read()) == 1                              # torn tail skipped, no raise


def test_agent_wires_journal_and_overlay_outcomes(tmp_path):
    """PaperBook mark → _after_mark journals mark+outcomes AND feeds overlay.record_outcome."""
    from backend.agents.strategy_agent import StrategyAgent, PaperBook

    class SpyOverlay:
        def __init__(self): self.calls = []
        def record_outcome(self, sym, ret): self.calls.append((sym, round(ret, 6)))

    j = TradeJournal(path=tmp_path / "j.jsonl")
    book = PaperBook(equity=100_000.0, cost_bps=0.0)
    agent = StrategyAgent(universe=["X"], bar_provider=None, portfolio=book,
                          journal=j, overlay_gate=SpyOverlay())
    asyncio.get_event_loop_policy().new_event_loop().run_until_complete(
        book.apply("BTCUSDT", 0.5, 0.5))
    book.mark_to_market({"BTCUSDT": 100.0}, time.time())           # seed
    ret = book.mark_to_market({"BTCUSDT": 110.0}, time.time() + 1)  # +10% → book +5%
    agent._after_mark(ret)
    marks = j.read(kinds=["mark"])
    outs = j.read(kinds=["outcome"])
    assert len(marks) == 1 and marks[0]["book_return"] == pytest.approx(0.05)
    assert outs[0]["symbol"] == "BTCUSDT" and outs[0]["contribution"] == pytest.approx(0.05)
    assert agent.overlay_gate.calls == [("BTCUSDT", 0.1)]          # per-symbol return 10%


# ─────────────────────────── post-mortem classify ───────────────────────────
def _mark(ts, ret, dd=0.0):
    return {"kind": "mark", "ts": ts, "book_return": ret, "drawdown": dd}


def test_postmortem_noise_vs_regime_and_process():
    import postmortem as pm
    # 20 marks: tiny negative drift (−0.05%/mark) inside 1%/mark vol → z ≈ −0.2 → NOISE
    rows = [_mark(i * 3600, -0.0005 + 0.01 * ((-1) ** i)) for i in range(20)]
    agg = pm.aggregate(rows, {})
    cs = pm.classify(agg)
    assert any(c["code"] == "loss_within_expectation" for c in cs) or agg["window_return"] >= 0
    assert not any(c["class"] == "process" for c in cs)
    # noise NEVER becomes a lesson
    assert pm.deterministic_lessons(cs) == [] or all(
        l["tag"] != "loss_within_expectation" for l in pm.deterministic_lessons(cs))

    # slippage far above the model + gross over cap → PROCESS concerns
    agg2 = pm.aggregate(rows, {"binance/passive": {"n": 40, "mean_bps": 20.0,
                                                   "worst_bps": 60.0, "vs_assumed_bps": 12.5}})
    agg2["max_gross"] = 2.7
    cs2 = pm.classify(agg2)
    codes = {c["code"] for c in cs2}
    assert "slippage_above_model" in codes and "gross_exposure_over_cap" in codes
    lessons = pm.deterministic_lessons(cs2)
    assert any("slippage" in l["tag"] for l in lessons)


def test_postmortem_regime_on_extreme_loss():
    import postmortem as pm
    rows = [_mark(i * 3600, -0.01) for i in range(20)]      # relentless −1%/mark
    agg = pm.aggregate(rows, {})
    cs = pm.classify(agg)
    assert any(c["code"] == "loss_beyond_expectation" and c["class"] == "regime" for c in cs)


def test_postmortem_gap_detection():
    import postmortem as pm
    ts = [0, 3600, 7200, 7200 + 5 * 3600, 7200 + 6 * 3600]   # one 5h gap in an hourly cadence
    rows = [_mark(t, 0.001) for t in ts]
    agg = pm.aggregate(rows, {})
    assert agg["journal_gaps"] >= 1
    assert any(c["code"] == "data_or_uptime_gap" for c in pm.classify(agg))


# ─────────────────────────── anomaly scanner ───────────────────────────
def _stats_row(sym, pct, qv):
    return {"symbol": sym, "priceChangePercent": pct, "quoteVolume": qv}


def test_scanner_flags_and_degear():
    cfg = ScannerConfig(min_quote_volume=1e6)
    stats = [_stats_row(f"S{i}USDT", 0.5, 5e7) for i in range(20)]
    stats.append(_stats_row("MOVERUSDT", 25.0, 5e7))         # huge move vs cross-section
    stats.append(_stats_row("THINUSDT", 0.4, 1e5))           # illiquid
    books = {"WIDEUSDT": {"bid": 100.0, "ask": 100.6},       # 60bps spread → severity 1
             "S0USDT": {"bid": 100.0, "ask": 100.01}}        # 1bp → clean
    flags = classify_anomalies(stats, books, cfg)
    assert any(f.kind == "abnormal_move" for f in flags["MOVERUSDT"])
    assert any(f.kind == "illiquid" for f in flags["THINUSDT"])
    assert any(f.kind == "wide_spread" for f in flags["WIDEUSDT"])
    assert "S0USDT" not in flags                             # clean symbol unflagged

    # de-gear: liquidity flags scale down to the floor; attention flags do NOT
    assert degear_from_flags(flags["WIDEUSDT"], cfg) == pytest.approx(cfg.degear_floor)
    assert degear_from_flags(flags["THINUSDT"], cfg) == pytest.approx(cfg.degear_floor)
    assert degear_from_flags(flags["MOVERUSDT"], cfg) == 1.0
    assert degear_from_flags([], cfg) == 1.0                 # fail open


def test_scanner_small_universe_no_zscores():
    flags = classify_anomalies([_stats_row("AUSDT", 50.0, 1e9)], {}, ScannerConfig())
    assert not any(f.kind == "abnormal_move" for f in flags.get("AUSDT", []))  # n<8 → no z flags


# ─────────────────────────── GBM ensemble arm ───────────────────────────
def test_gbm_arm_learns_and_ensembles(tmp_path):
    lgb = pytest.importorskip("lightgbm")
    import train_quantile_tcn as tq
    rng = np.random.default_rng(0)
    n = 4000
    y = rng.standard_normal((n, len(tq.HORIZONS))).astype(np.float32)
    X = rng.standard_normal((n, tq.FEATURE_DIM)).astype(np.float32) * 0.1
    X[:, 3] = y[:, 1] + 0.1 * rng.standard_normal(n)         # planted signal @ H+48 arm target
    import pandas as pd
    close = 100 * np.cumprod(1 + 0.001 * rng.standard_normal(n))
    d = dict(X=X, y=y, close=close, high=close, low=close,
             timestamps=pd.date_range("2024-01-01", periods=n, freq="5min").to_numpy())
    old = tq.GBM_PATH
    tq.GBM_PATH = tmp_path / "gbm.json"
    try:
        arm = tq.train_gbm_arm({"BTCUSDT": d}, stride=4, log=lambda *a, **k: None)
        ends = tq.window_ends(n, y, 4)
        _, va, _ = tq.split_ends(ends, 3)
        e, u = arm.predict_edge_unc_rows(X[va - 1])
        corr = np.corrcoef(e, y[va - 1, 1])[0, 1]
        assert corr > 0.5                                    # planted signal learned
        assert np.all(u > 0)
        # duck-typed into the ensemble path
        import torch
        E, U = tq.ensemble_edge_unc([tq.GBMArm.load(tq.GBM_PATH)], X, va[:64], 0,
                                    torch.device("cpu"))
        assert np.corrcoef(E, y[va[:64] - 1, 1])[0, 1] > 0.4
    finally:
        tq.GBM_PATH = old
