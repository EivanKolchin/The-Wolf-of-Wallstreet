"""Tests for the two-sleeve live target-weights (backend/strategies/combined_book.py)."""
import numpy as np
import pandas as pd
import pytest

from backend.strategies.combined_book import (ts_momentum_target_weights, combined_target_weights)


def _ohlc(closes):
    c = np.asarray(closes, float)
    ts = pd.date_range("2023-01-01", periods=len(c), freq="4h")
    return pd.DataFrame({"timestamp": ts, "open": c, "high": c * 1.002,
                         "low": c * 0.998, "close": c, "volume": np.ones(len(c)) * 1000})


def _trend(n=400, start=100.0, drift=0.004):
    rng = np.random.default_rng(0)
    steps = drift + rng.normal(0, 0.005, n)
    return start * np.exp(np.cumsum(steps))


def _flat(n=400, val=100.0):
    return np.full(n, val) + np.random.default_rng(1).normal(0, 0.01, n)


def test_ts_weights_keys_and_bounds():
    bars = {"BTCUSDT": _ohlc(_trend()), "ETHUSDT": _ohlc(_trend(drift=0.003))}
    w = ts_momentum_target_weights(bars)
    assert set(w) == {"BTCUSDT", "ETHUSDT"}
    assert all(np.isfinite(v) for v in w.values())
    assert sum(abs(v) for v in w.values()) <= 2.0 + 1e-6        # gross within max_leverage


def test_ts_weights_too_few_symbols_returns_zeros():
    w = ts_momentum_target_weights({"BTCUSDT": _ohlc(_trend())})
    assert w == {"BTCUSDT": 0.0}


def test_ts_weights_uptrend_goes_long():
    bars = {"BTCUSDT": _ohlc(_trend(drift=0.006)), "ETHUSDT": _ohlc(_trend(drift=0.006))}
    w = ts_momentum_target_weights(bars)
    assert sum(w.values()) > 0                                  # net long in a strong uptrend


def test_combined_merges_and_scales():
    ts_bars = {"BTCUSDT": _ohlc(_trend(drift=0.006)), "ETHUSDT": _ohlc(_trend(drift=0.006))}
    # fake managed sleeve via monkeypatch-free path: pass empty managed, only TS contributes
    out = combined_target_weights({}, ts_bars, w_managed=0.5, w_ts=0.5)
    raw = ts_momentum_target_weights(ts_bars)
    for s in raw:
        assert out[s] == pytest.approx(0.5 * raw[s])


def test_combined_sums_overlapping_symbol(monkeypatch):
    import backend.strategies.combined_book as cb
    monkeypatch.setattr(cb, "managed_target_weights", lambda bars, params, **kw: {"BTCUSDT": 0.4, "SPY": 0.2})
    ts_bars = {"BTCUSDT": _ohlc(_trend(drift=0.006)), "ETHUSDT": _ohlc(_trend(drift=0.006))}
    out = cb.combined_target_weights({"x": _ohlc(_trend())}, ts_bars, w_managed=0.5, w_ts=0.5)
    ts = cb.ts_momentum_target_weights(ts_bars)
    # SPY only from managed; BTC from both → summed
    assert out["SPY"] == pytest.approx(0.5 * 0.2)
    assert out["BTCUSDT"] == pytest.approx(0.5 * 0.4 + 0.5 * ts["BTCUSDT"])


def test_combined_empty_both():
    assert combined_target_weights({}, {}) == {}
