"""FeatureStore — the single entry point for building the model's input sequence.

The whole point is to KILL train/serve skew: research (history from Parquet) and
live (hot OHLCV from the websocket buffer / Redis) both call the *same* assembly
(``backend.features.pipeline.assemble_matrix``), so the (seq_len, INPUT) tensor the
model sees offline is byte-for-byte what it sees live — given the same input bars.

The store is deliberately SOURCE-AGNOSTIC: callers pass OHLCV DataFrames, whether
those came from a Parquet partition (backtest) or the live buffer (agent). That is
the "identical API over both backends" the plan calls for — the backend is just
where the DataFrame is read from; the feature math is shared and lives in one place.

Critical live constraint (see ``min_bars_required``): the offline pipeline z-scores
each feature over a trailing ``zscore_window`` (1000) bars. For the LAST row's
normalisation to match offline, the live OHLCV buffer must hold at least
``zscore_window + seq_len`` bars (plus EMA-200 warm-up). A shorter buffer silently
re-introduces skew — the live wiring must size the buffer accordingly.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from backend.features.pipeline import assemble_matrix, ZSCORE_WIN

try:
    from backend.signals import feature_spec as fs
except ImportError:  # pragma: no cover
    from signals import feature_spec as fs

SEQ_LEN_DEFAULT = 60          # matches PersistentTradingModel.SEQUENCE_LENGTH
EMA_WARMUP = 200              # longest indicator look-back (EMA-200) before features stabilise


class FeatureStore:
    """Builds the canonical (seq_len, fs.INPUT) feature sequence from OHLCV.

    Stateless w.r.t. data (the caller owns the buffers); holds only the geometry
    (sequence length + z-score window) so offline and live agree on shape and
    normalisation."""

    def __init__(self, seq_len: int = SEQ_LEN_DEFAULT, zscore_window: int = ZSCORE_WIN):
        self.seq_len = int(seq_len)
        self.zscore_window = int(zscore_window)

    @property
    def min_bars_required(self) -> int:
        """Minimum 5m bars the live buffer must hold for the last row's z-score to
        match offline (full trailing window) plus a complete sequence + EMA warm-up."""
        return self.zscore_window + self.seq_len + EMA_WARMUP

    @staticmethod
    def _ensure_dt(df: pd.DataFrame) -> pd.DataFrame:
        """Guarantee a datetime ``timestamp`` column (the pipeline reads
        ``df['timestamp'].dt``). The live feed stores ms-epoch ints + a datetime
        index; offline/backtest already use datetime. Idempotent."""
        if "timestamp" in df.columns and np.issubdtype(df["timestamp"].dtype, np.datetime64):
            return df
        out = df.copy()
        if "timestamp" in out.columns and not np.issubdtype(out["timestamp"].dtype, np.datetime64):
            # ms epoch (live feed) -> datetime; auto-detect µs vs ms by magnitude
            ts = pd.to_numeric(out["timestamp"], errors="coerce")
            unit = "us" if (len(ts) and float(ts.iloc[0]) >= 1e14) else "ms"
            out["timestamp"] = pd.to_datetime(ts.astype("int64"), unit=unit)
        elif isinstance(out.index, pd.DatetimeIndex):
            out["timestamp"] = out.index
        else:
            raise ValueError("OHLCV frame needs a 'timestamp' column or a DatetimeIndex")
        return out

    @staticmethod
    def resample_htf(df5: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Derive 1h and 4h OHLCV from the 5m frame for the live path (which only
        keeps a 5m buffer). Offline uses Binance's native 1h/4h bars; resampling a
        complete 5m series reproduces them up to bar-boundary rounding. Returns
        ``(df1h, df4h)`` with the same columns the pipeline expects."""
        d = FeatureStore._ensure_dt(df5).set_index("timestamp")
        agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
        out = []
        for rule in ("1h", "4h"):
            r = d.resample(rule, label="left", closed="left").agg(agg).dropna(subset=["close"])
            out.append(r.reset_index())
        return out[0], out[1]

    def build_matrix(self, df5m: pd.DataFrame, df1h: pd.DataFrame, df4h: pd.DataFrame,
                     news_mat: Optional[np.ndarray] = None,
                     earnings_mat: Optional[np.ndarray] = None) -> np.ndarray:
        """Full (N, fs.INPUT) canonical matrix over the whole input window."""
        df5m = self._ensure_dt(df5m)
        df1h = self._ensure_dt(df1h)
        df4h = self._ensure_dt(df4h)
        return assemble_matrix(df5m, df1h, df4h, news_mat=news_mat, earnings_mat=earnings_mat)

    def build_sequence(self, df5m: pd.DataFrame, df1h: pd.DataFrame, df4h: pd.DataFrame,
                       news_mat: Optional[np.ndarray] = None,
                       earnings_mat: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """The model input: the LAST ``seq_len`` rows of the canonical matrix, or
        ``None`` if there aren't enough bars. Identical to what the trainer feeds the
        model for the same window."""
        mat = self.build_matrix(df5m, df1h, df4h, news_mat=news_mat, earnings_mat=earnings_mat)
        if mat.shape[0] < self.seq_len:
            return None
        return mat[-self.seq_len:]
