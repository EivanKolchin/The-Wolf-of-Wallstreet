"""Verify the pure-pandas `ta` shim replaces pandas_ta cleanly and the offline
feature pipeline still runs + produces finite, correctly-shaped features."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

_spec = importlib.util.spec_from_file_location("pretrain_ta_mod", str(ROOT / "scripts" / "pretrain.py"))
pt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pt)

from signals import feature_spec as fs  # noqa: E402


def _ohlcv(n=600, seed=0):
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.standard_normal(n))
    close = np.maximum(close, 1.0)
    high = close + np.abs(rng.standard_normal(n))
    low = close - np.abs(rng.standard_normal(n))
    open_ = close + rng.standard_normal(n) * 0.5
    vol = rng.uniform(10, 1000, n)
    ts = pd.date_range("2022-01-01", periods=n, freq="5min")
    return pd.DataFrame({"timestamp": ts, "open": open_, "high": high, "low": low,
                         "close": close, "volume": vol})


def test_shim_has_expected_functions():
    for fn in ("ema", "rsi", "macd", "stochrsi", "adx", "atr", "bbands", "obv"):
        assert hasattr(pt.ta, fn)
    assert pt.HAS_PANDAS_TA is True


def test_shim_output_shapes_and_order():
    df = _ohlcv()
    c = df["close"]
    assert isinstance(pt.ta.rsi(c), pd.Series)
    assert isinstance(pt.ta.ema(c, 21), pd.Series)
    assert isinstance(pt.ta.obv(c, df["volume"]), pd.Series)
    macd = pt.ta.macd(c)
    assert list(macd.columns)[:2] == ["MACD", "MACDh"]          # macd, histogram (positional)
    bb = pt.ta.bbands(c, length=20)
    # pandas_ta order: lower, mid, upper -> iloc 0,1,2
    last = len(df) - 1
    assert bb.iloc[last, 0] < bb.iloc[last, 1] < bb.iloc[last, 2]
    adx = pt.ta.adx(df["high"], df["low"], df["close"])
    assert list(adx.columns)[0] == "ADX"


def test_feature_matrix_pipeline_runs():
    df = _ohlcv()
    base = pt.build_feature_matrix(df)               # (N, 62) — exercises all ta.* calls
    assert base.shape == (len(df), fs.BASE)
    # The indicator block (slots 11-21: rsi, macd, stochrsi, adx, atr, bbands,
    # volume, obv) is what the `ta` shim feeds — it must be fully finite.
    # (Slots 22-24 are Fibonacci rolling-window warm-up NaNs, pre-existing and
    # handled downstream by z-scoring/sequence-building — not the shim's concern.)
    assert np.isfinite(base[:, 11:22]).all()
    reg = pt.detect_regime(df, base)                 # uses ta.adx/atr
    assert reg.shape == (len(df), fs.BASE)
    # RSI feature (slot 11) is in [-1, 1] after the /50 normalisation
    assert base[:, 11].min() >= -1.0001 and base[:, 11].max() <= 1.0001
