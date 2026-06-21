"""A3: the prebuilt-feature cache. The ~50s/symbol feature+label build must run ONCE and be
reused across reruns / separate experiment processes — a cache hit needs no raw data at all."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

import importlib.util

_spec = importlib.util.spec_from_file_location("pre", str(ROOT / "scripts" / "pretrain.py"))
pre = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pre)


def test_assemble_with_cache_builds_once_then_reuses(tmp_path, monkeypatch):
    monkeypatch.setenv("PRETRAIN_FEATURE_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("PRETRAIN_FEATURE_CACHE_DISABLE", raising=False)
    monkeypatch.setenv("PRETRAIN_NEWS_ALIGN", "0")
    monkeypatch.setenv("PRETRAIN_EARNINGS_ALIGN", "0")

    n = 300
    calls = {"load": 0, "assemble": 0, "labels": 0}
    ts = pd.date_range("2023-01-01", periods=n, freq="5min")

    def fake_load(sym, y, m, sk):
        calls["load"] += 1
        df = pd.DataFrame({"timestamp": ts, "open": 1.0, "high": 1.0, "low": 1.0,
                           "close": 1.0, "volume": 1.0})
        return {"5m": df, "1h": df, "4h": df}

    def fake_assemble(df5, df1, df4, sym):
        calls["assemble"] += 1
        return np.random.RandomState(0).randn(n, pre.INPUT_SIZE).astype(np.float32)

    def fake_labels(df, horizons, k):
        calls["labels"] += 1
        H = len(horizons)
        return np.full((n, H), 2, np.int64), np.zeros((n, H), np.float32)

    monkeypatch.setattr(pre, "load_full_history", fake_load)
    monkeypatch.setattr(pre, "assemble_feature_matrix", fake_assemble)
    monkeypatch.setattr(pre, "triple_barrier_labels", fake_labels)

    # 1st call → builds + writes the cache
    c1, l1, r1, t1 = pre.assemble_with_cache("BTCUSDT", 2023, 1, True)
    assert calls == {"load": 1, "assemble": 1, "labels": 1}
    assert len(list(tmp_path.glob("*.npz"))) == 1

    # 2nd call → cache HIT: none of the expensive builders run again
    c2, l2, r2, t2 = pre.assemble_with_cache("BTCUSDT", 2023, 1, True)
    assert calls == {"load": 1, "assemble": 1, "labels": 1}      # unchanged
    assert np.array_equal(c1, c2) and np.array_equal(l1, l2) and np.array_equal(r1, r2)
    assert np.array_equal(t1, t2)


def test_cache_key_separates_news_on_from_news_off(tmp_path, monkeypatch):
    """news-on and news-off produce DIFFERENT matrices → must be different cache files."""
    monkeypatch.setenv("PRETRAIN_FEATURE_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("PRETRAIN_NEWS_ALIGN", "0")
    monkeypatch.setenv("PRETRAIN_EARNINGS_ALIGN", "0")
    p_off = pre._feature_cache_path("BTCUSDT", 2023, 1)
    monkeypatch.setenv("PRETRAIN_NEWS_ALIGN", "1")
    p_on = pre._feature_cache_path("BTCUSDT", 2023, 1)
    assert p_off != p_on


def test_extra_stocks_env_extends_routing(monkeypatch):
    """PRETRAIN_EXTRA_STOCKS lets the audit route arbitrary tickers to Alpaca (broad universe)
    without code edits; crypto must still route to Binance."""
    monkeypatch.delenv("PRETRAIN_EXTRA_STOCKS", raising=False)
    assert pre._is_stock_symbol("NVDA")           # base set
    assert not pre._is_stock_symbol("AVGO")       # not registered yet
    assert not pre._is_stock_symbol("BTCUSDT")    # crypto
    monkeypatch.setenv("PRETRAIN_EXTRA_STOCKS", "AVGO, QCOM  INTC")
    assert pre._is_stock_symbol("AVGO") and pre._is_stock_symbol("qcom") and pre._is_stock_symbol("INTC")
    assert pre._is_stock_symbol("NVDA")           # base still works
    assert not pre._is_stock_symbol("BTCUSDT")    # crypto unaffected


def test_cache_disable_env_bypasses(tmp_path, monkeypatch):
    monkeypatch.setenv("PRETRAIN_FEATURE_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("PRETRAIN_FEATURE_CACHE_DISABLE", "1")
    monkeypatch.setenv("PRETRAIN_NEWS_ALIGN", "0")
    monkeypatch.setenv("PRETRAIN_EARNINGS_ALIGN", "0")
    n = 120
    ts = pd.date_range("2023-01-01", periods=n, freq="5min")
    monkeypatch.setattr(pre, "load_full_history", lambda *a: {
        "5m": pd.DataFrame({"timestamp": ts, "open": 1.0, "high": 1.0, "low": 1.0,
                            "close": 1.0, "volume": 1.0})} | {"1h": None, "4h": None})
    monkeypatch.setattr(pre, "assemble_feature_matrix",
                        lambda *a: np.zeros((n, pre.INPUT_SIZE), np.float32))
    monkeypatch.setattr(pre, "triple_barrier_labels",
                        lambda *a: (np.full((n, len(pre.HORIZONS)), 2, np.int64),
                                    np.zeros((n, len(pre.HORIZONS)), np.float32)))
    pre.assemble_with_cache("ETHUSDT", 2023, 1, True)
    assert len(list(tmp_path.glob("*.npz"))) == 0       # disabled → nothing written
