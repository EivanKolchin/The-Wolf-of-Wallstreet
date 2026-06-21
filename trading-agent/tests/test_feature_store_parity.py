"""Train/serve parity: the FeatureStore (live) must produce byte-identical features
to the offline trainer's assembly for the SAME input bars — including when fed the
live websocket frame format (ms-epoch 'timestamp' + DatetimeIndex). This is the
regression guard for the train/serve skew that motivated the whole rebuild.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

# Offline trainer (a script) — loaded the same way the other pretrain tests do.
_spec = importlib.util.spec_from_file_location("pretrain_store_mod", str(ROOT / "scripts" / "pretrain.py"))
pre = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pre)

from backend.features.store import FeatureStore  # noqa: E402
from signals import feature_spec as fs  # noqa: E402


def _ohlcv_5m(n=1400, seed=7):
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.standard_normal(n) * 0.5)
    close = np.maximum(close, 1.0)
    high = close + np.abs(rng.standard_normal(n)) * 0.3
    low = close - np.abs(rng.standard_normal(n)) * 0.3
    open_ = close + rng.standard_normal(n) * 0.2
    vol = rng.uniform(10, 1000, n)
    ts = pd.date_range("2022-01-01", periods=n, freq="5min")
    return pd.DataFrame({"timestamp": ts, "open": open_, "high": high,
                         "low": low, "close": close, "volume": vol})


def _resample(df5, rule):
    d = df5.set_index("timestamp")
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    return d.resample(rule).agg(agg).dropna(subset=["close"]).reset_index()


def test_featurestore_matches_offline_assembly():
    df5 = _ohlcv_5m()
    df1h, df4h = _resample(df5, "1h"), _resample(df5, "4h")

    offline = pre.assemble_feature_matrix(df5.copy(), df1h.copy(), df4h.copy(), "BTCUSDT")
    store = FeatureStore(seq_len=60)
    seq = store.build_sequence(df5.copy(), df1h.copy(), df4h.copy())

    assert offline.shape[1] == fs.INPUT
    assert seq.shape == (60, fs.INPUT)
    # Identical math + identical input window => byte-identical features.
    np.testing.assert_allclose(seq, offline[-60:], rtol=0, atol=1e-5)


def test_featurestore_handles_live_feed_format():
    """The live websocket frame stores ms-epoch ints in 'timestamp' + a DatetimeIndex.
    Features built from THAT must match the offline datetime-keyed assembly exactly —
    otherwise the live model silently sees a different distribution than it trained on."""
    df5 = _ohlcv_5m()
    df1h, df4h = _resample(df5, "1h"), _resample(df5, "4h")
    offline = pre.assemble_feature_matrix(df5.copy(), df1h.copy(), df4h.copy(), "BTCUSDT")

    def _to_live(df):
        # Mirror the Binance feed: integer MILLISECOND epochs (k["t"]). Force ms
        # resolution explicitly so this is correct regardless of pandas' default
        # datetime unit (pandas 3.0 uses us; .astype('int64') alone would not be ms).
        live = df.copy()
        live["timestamp"] = df["timestamp"].values.astype("datetime64[ms]").astype("int64")
        live.index = pd.to_datetime(live["timestamp"], unit="ms")
        return live

    store = FeatureStore(seq_len=60)
    seq = store.build_sequence(_to_live(df5), _to_live(df1h), _to_live(df4h))
    np.testing.assert_allclose(seq, offline[-60:], rtol=0, atol=1e-5)


def test_build_sequence_insufficient_returns_none():
    df5 = _ohlcv_5m(n=40)
    df1h, df4h = _resample(df5, "1h"), _resample(df5, "4h")
    store = FeatureStore(seq_len=60)
    assert store.build_sequence(df5, df1h, df4h) is None


def test_min_bars_required_covers_zscore_window():
    store = FeatureStore(seq_len=60, zscore_window=1000)
    # Must demand at least the z-score window + a full sequence so the last row's
    # normalisation matches offline (the live buffer must be sized to this).
    assert store.min_bars_required >= 1000 + 60


def test_resample_htf_shapes():
    df5 = _ohlcv_5m()
    df1h, df4h = FeatureStore.resample_htf(df5)
    assert {"open", "high", "low", "close", "volume", "timestamp"} <= set(df1h.columns)
    assert len(df1h) > len(df4h) > 0
