"""Canonical, vectorised feature engineering — the SINGLE source of truth.

This module holds the exact feature math that was previously embedded in
``scripts/pretrain.py``. It is intentionally dependency-light (numpy + pandas +
``feature_spec`` only) so the LIVE agent and the backtest can import it without
pulling torch / requests / the training script's side effects.

``pretrain.py`` imports these symbols rather than defining its own copies, so
the offline trainer, the backtest, and the live builder can never drift again.
The behaviour here is byte-for-byte the offline behaviour at extraction time —
this was a pure refactor (move, not rewrite).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

try:  # works whether imported as backend.features.* or with backend/ on path
    from backend.signals import feature_spec as fs
except ImportError:  # pragma: no cover
    from signals import feature_spec as fs

# --- shared layout constants (derive from feature_spec → no drift) ---
BASE_FEATURES = fs.BASE      # 62
HTF_FEATURES = fs.HTF        # 8
ZSCORE_WIN = 1000            # rolling normalisation window


# pandas_ta is effectively abandoned: the classic 0.3.14b0 this code was written
# for was pulled from PyPI, and the only remaining 0.4.x demands
# numpy>=2.2.6 / pandas>=2.3.2 — which fights Colab's pinned pandas==2.2.2 and
# kept making training un-installable. We only used 8 standard indicators, so we
# vendor a tiny pure-pandas drop-in named `ta` with the SAME function names AND
# output column order (the code reads them positionally), so behaviour is
# unchanged but there is ZERO external indicator dependency.
class _TA:
    @staticmethod
    def _rma(s, length):
        # Wilder's smoothing (RMA), as used by pandas_ta's rsi/atr/adx.
        return s.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()

    @staticmethod
    def ema(close, length):
        return close.ewm(span=length, adjust=False).mean()

    @staticmethod
    def rsi(close, length=14):
        d = close.diff()
        ag = _TA._rma(d.clip(lower=0.0), length)
        al = _TA._rma((-d).clip(lower=0.0), length)
        rs = ag / al.replace(0.0, np.nan)
        return (100.0 - 100.0 / (1.0 + rs)).fillna(100.0)

    @staticmethod
    def macd(close, fast=12, slow=26, signal=9):
        macd = close.ewm(span=fast, adjust=False).mean() - close.ewm(span=slow, adjust=False).mean()
        sig = macd.ewm(span=signal, adjust=False).mean()
        # pandas_ta order: MACD, MACDh (histogram), MACDs (signal)
        return pd.DataFrame({"MACD": macd, "MACDh": macd - sig, "MACDs": sig})

    @staticmethod
    def stochrsi(close, length=14, rsi_length=14, k=3, d=3):
        r = _TA.rsi(close, rsi_length)
        lo = r.rolling(length).min()
        hi = r.rolling(length).max()
        st = 100.0 * (r - lo) / (hi - lo).replace(0.0, np.nan)
        kl = st.rolling(k).mean()
        # pandas_ta order: STOCHRSIk, STOCHRSId
        return pd.DataFrame({"STOCHRSIk": kl, "STOCHRSId": kl.rolling(d).mean()})

    @staticmethod
    def _tr(high, low, close):
        pc = close.shift(1)
        return pd.concat([(high - low), (high - pc).abs(), (low - pc).abs()], axis=1).max(axis=1)

    @staticmethod
    def atr(high, low, close, length=14):
        return _TA._rma(_TA._tr(high, low, close), length)

    @staticmethod
    def adx(high, low, close, length=14):
        up = high.diff()
        dn = -low.diff()
        plus_dm = ((up > dn) & (up > 0)).astype(float) * up
        minus_dm = ((dn > up) & (dn > 0)).astype(float) * dn
        atr = _TA._rma(_TA._tr(high, low, close), length).replace(0.0, np.nan)
        pdi = 100.0 * _TA._rma(plus_dm, length) / atr
        mdi = 100.0 * _TA._rma(minus_dm, length) / atr
        dx = 100.0 * (pdi - mdi).abs() / (pdi + mdi).replace(0.0, np.nan)
        # pandas_ta order: ADX, DMP (+DI), DMN (-DI)
        return pd.DataFrame({"ADX": _TA._rma(dx, length), "DMP": pdi, "DMN": mdi})

    @staticmethod
    def bbands(close, length=20, std=2.0):
        mid = close.rolling(length).mean()
        sd = close.rolling(length).std(ddof=0)
        lower, upper = mid - std * sd, mid + std * sd
        # pandas_ta order: BBL (lower), BBM (mid), BBU (upper), BBB, BBP
        return pd.DataFrame({
            "BBL": lower, "BBM": mid, "BBU": upper,
            "BBB": 100.0 * (upper - lower) / mid.replace(0.0, np.nan),
            "BBP": (close - lower) / (upper - lower).replace(0.0, np.nan),
        })

    @staticmethod
    def obv(close, volume):
        return (np.sign(close.diff().fillna(0.0)) * volume).cumsum()


ta = _TA
HAS_PANDAS_TA = True


def _safe(series, fill=0.0):
    return series.fillna(fill).values.astype(np.float32)


def build_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    """
    Builds (N, BASE_FEATURES=62) feature matrix for one symbol using
    fully vectorised pandas_ta operations — no row-by-row loops.

    Feature layout (matches live system):
      [0-2]   candle body, upper wick, lower wick (scaled)
      [3]     volume ratio vs 20-bar MA
      [4]     spread placeholder (0.5)
      [5-8]   EMA distances (9, 21, 50, 200)
      [9]     golden cross (ema50 > ema200)
      [10]    VWAP distance
      [11]    RSI normalised
      [12-13] MACD line, MACD hist (normalised)
      [14]    StochRSI %K
      [15]    ADX normalised
      [16]    RSI divergence (binary)
      [17]    ATR normalised
      [18-19] BB width, BB %b
      [20]    Volume momentum (5-bar vs 20-bar MA — was duplicate, now distinct)
      [21]    OBV slope sign
      [22-24] Fibonacci position, distance from 50%, range strength
      [25-34] 10 candlestick pattern flags
      [35-44] Orderbook slots (0.0 — filled live)
      [45-48] Regime one-hot (ranging, bull_trend, bear_trend, volatile)
      [49-52] News slots (0.0 — filled live)
      [53-56] Macro slots (0.5 neutral — filled live)
      [57-60] Time cyclical (sin/cos hour, sin/cos weekday)
      [61]    Regime confidence
    """
    assert HAS_PANDAS_TA, "pandas_ta required for vectorised features"

    n   = len(df)
    out = np.zeros((n, BASE_FEATURES), dtype=np.float32)

    op  = df["open"].values.astype(np.float32)
    hi  = df["high"].values.astype(np.float32)
    lo  = df["low"].values.astype(np.float32)
    cl  = df["close"].values.astype(np.float32)
    vol = df["volume"].values.astype(np.float32)

    eps = 1e-8

    # [0-2] Candle body / wicks
    out[:, 0] = np.clip((cl - op) / (op + eps), -0.05, 0.05) * 20.0
    out[:, 1] = np.clip((hi - op) / (op + eps),  0.0,  0.05) * 20.0
    out[:, 2] = np.clip((lo - op) / (op + eps), -0.05, 0.0 ) * 20.0

    # [3] Volume ratio (current / 20-bar MA)
    vol_ma20  = pd.Series(vol).rolling(20, min_periods=1).mean().values
    out[:, 3] = np.clip(vol / (vol_ma20 + eps), 0, 5) / 5.0

    # [4] Spread placeholder
    out[:, 4] = 0.5

    # [5-8] EMA distances
    c_series = df["close"]
    ema9   = _safe(ta.ema(c_series, length=9))
    ema21  = _safe(ta.ema(c_series, length=21))
    ema50  = _safe(ta.ema(c_series, length=50))
    ema200 = _safe(ta.ema(c_series, length=200))

    out[:, 5] = np.clip((cl - ema9)   / (cl + eps), -0.1, 0.1)
    out[:, 6] = np.clip((cl - ema21)  / (cl + eps), -0.1, 0.1)
    out[:, 7] = np.clip((cl - ema50)  / (cl + eps), -0.1, 0.1)
    out[:, 8] = np.clip((cl - ema200) / (cl + eps), -0.1, 0.1)

    # [9] Golden cross
    out[:, 9] = (ema50 > ema200).astype(np.float32)

    # [10] VWAP distance — ROLLING-window anchored VWAP via the shared helper (was a
    # cumulative-from-epoch average that decayed to a flat constant → dead feature + skew).
    out[:, 10] = np.clip(fs.rolling_vwap_distance(hi, lo, cl, vol), -0.1, 0.1)

    # [11] RSI
    rsi = _safe(ta.rsi(c_series, length=14), fill=50.0)
    out[:, 11] = (rsi - 50.0) / 50.0

    # [12-13] MACD
    macd_df = ta.macd(c_series)
    if macd_df is not None and not macd_df.empty:
        macd_line = _safe(macd_df.iloc[:, 0])
        macd_hist = _safe(macd_df.iloc[:, 1] if macd_df.shape[1] > 1 else macd_df.iloc[:, 0])
    else:
        macd_line = macd_hist = np.zeros(n, dtype=np.float32)
    out[:, 12] = np.clip(macd_line / (cl + eps) * 100, -1, 1)
    out[:, 13] = np.clip(macd_hist / (cl + eps) * 100, -1, 1)

    # [14] StochRSI
    stoch = ta.stochrsi(c_series)
    if stoch is not None and not stoch.empty:
        out[:, 14] = _safe(stoch.iloc[:, 0], fill=50.0) / 100.0
    else:
        out[:, 14] = 0.5

    # [15] ADX
    adx_df = ta.adx(df["high"], df["low"], df["close"])
    if adx_df is not None and not adx_df.empty:
        out[:, 15] = np.clip(_safe(adx_df.iloc[:, 0]) / 100.0, 0, 1)

    # [16] RSI divergence (sign disagreement: RSI vs price 1-bar BACKWARD slope).
    # NOTE: np.gradient uses a CENTERED difference → element i reads i+1 (look-ahead +
    # train/serve skew: live can't see the next bar). Backward diff (x[i]-x[i-1]) is
    # causal and reproducible live.
    rsi_slope   = np.diff(rsi, prepend=rsi[:1])
    price_slope = np.diff(cl,  prepend=cl[:1])
    out[:, 16] = (np.sign(rsi_slope) != np.sign(price_slope)).astype(np.float32)

    # [17] ATR normalised
    atr = _safe(ta.atr(df["high"], df["low"], df["close"], length=14))
    out[:, 17] = np.clip(atr / (cl + eps), 0, 0.1) * 10.0

    # [18-19] Bollinger Bands
    bb = ta.bbands(c_series, length=20)
    if bb is not None and not bb.empty:
        # pandas_ta AND the _TA shim both return columns [BBL, BBM, BBU] = lower, mid, upper.
        # This was read as [upper, mid, lower] → bb_rng went negative → feat 18 (BB width)
        # clipped to 0 (dead) and feat 19 (%b) was inverted. Correct order below.
        bb_lower = _safe(bb.iloc[:, 0])
        bb_mid   = _safe(bb.iloc[:, 1])
        bb_upper = _safe(bb.iloc[:, 2])
        bb_rng   = bb_upper - bb_lower + eps
        out[:, 18] = np.clip((bb_upper - bb_lower) / (bb_mid + eps), 0, 0.2) * 5.0
        out[:, 19] = np.clip((cl - bb_lower) / bb_rng, -0.5, 1.5)

    # [20] Volume momentum (5-bar MA vs 20-bar MA — no longer a duplicate)
    vol_ma5   = pd.Series(vol).rolling(5,  min_periods=1).mean().values
    out[:, 20] = np.clip(vol_ma5 / (vol_ma20 + eps), 0, 5) / 5.0

    # [21] OBV slope sign (1-bar BACKWARD difference — causal; see feat 16 note on np.gradient).
    obv = _safe(ta.obv(c_series, df["volume"]))
    out[:, 21] = np.sign(np.diff(obv, prepend=obv[:1])).astype(np.float32)

    # [22-24] Fibonacci
    win = 50
    roll_hi = pd.Series(hi).rolling(win, min_periods=win).max().values
    roll_lo = pd.Series(lo).rolling(win, min_periods=win).min().values
    fib_rng  = roll_hi - roll_lo + eps
    fib_50   = roll_lo + fib_rng * 0.5
    out[:, 22] = np.clip((cl - roll_lo) / fib_rng, 0, 1)
    out[:, 23] = np.clip(np.abs(cl - fib_50) / fib_rng, 0, 0.5)
    out[:, 24] = np.clip(fib_rng / (cl + eps), 0, 0.2) * 5.0

    # [25-34] Candlestick patterns (10 binary flags)
    body       = np.abs(cl - op)
    rng        = (hi - lo) + eps
    lower_wick = np.where(cl > op, op - lo, cl - lo)
    upper_wick = np.where(cl > op, hi - cl, hi - op)
    cl_s  = pd.Series(cl)
    op_s  = pd.Series(op)
    prev_cl = cl_s.shift(1).values
    prev_op = op_s.shift(1).values

    out[:, 25] = (body / rng < 0.1).astype(np.float32)                                     # doji
    out[:, 26] = ((cl > op) & (body / rng > 0.6)).astype(np.float32)                       # bull marubozu
    out[:, 27] = ((cl < op) & (body / rng > 0.6)).astype(np.float32)                       # bear marubozu
    out[:, 28] = ((lower_wick > 2*body) & (upper_wick < body)).astype(np.float32)          # hammer
    out[:, 29] = ((upper_wick > 2*body) & (lower_wick < body)).astype(np.float32)          # shooting star
    out[:, 30] = ((cl_s > cl_s.shift(1)) & (cl_s.shift(1) > cl_s.shift(2))).astype(np.float32).values  # 3 up
    out[:, 31] = ((cl_s < cl_s.shift(1)) & (cl_s.shift(1) < cl_s.shift(2))).astype(np.float32).values  # 3 down
    out[:, 32] = ((cl > prev_op) & (op < prev_cl) & (prev_cl < prev_op)).astype(np.float32)  # bull engulf
    out[:, 33] = ((cl < prev_op) & (op > prev_cl) & (prev_cl > prev_op)).astype(np.float32)  # bear engulf
    out[:, 34] = 0.0   # spare

    # [35-44] Orderbook — zeros in historical (filled live)
    out[:, fs.ORDERBOOK] = 0.0   # 8 slots (35:43) — canonical FeatureSpec layout

    # [45-48] Regime — filled by detect_regime()
    # [49-52] News — zeros (filled live)
    out[:, fs.NEWS] = 0.0

    # [53-56] Macro — neutral
    out[:, 53] = 0.5
    out[:, 54] = 0.5
    out[:, 55] = 0.0
    out[:, 56] = 0.0

    # [57-60] Time cyclical
    ts = df["timestamp"]
    out[:, 57] = np.sin(2 * np.pi * ts.dt.hour.values / 24.0).astype(np.float32)
    out[:, 58] = np.cos(2 * np.pi * ts.dt.hour.values / 24.0).astype(np.float32)
    out[:, 59] = np.sin(2 * np.pi * ts.dt.dayofweek.values / 7.0).astype(np.float32)
    out[:, 60] = np.cos(2 * np.pi * ts.dt.dayofweek.values / 7.0).astype(np.float32)

    # [61] Regime confidence — filled by detect_regime()
    out[:, 61] = 0.5

    return out


def detect_regime(df: pd.DataFrame, features: np.ndarray) -> np.ndarray:
    """
    Classifies each bar into one of 4 regimes using ADX + EMA slope:
      ranging (0), bull_trend (1), bear_trend (2), volatile (3)

    Writes regime one-hot into features[:, 45:49] and
    confidence into features[:, 61].  Modifies in-place, returns array.
    """
    cl     = df["close"].values.astype(np.float64)
    ema21  = pd.Series(cl).ewm(span=21, adjust=False).mean().values
    ema50  = pd.Series(cl).ewm(span=50, adjust=False).mean().values
    atr14  = _safe(ta.atr(df["high"], df["low"], df["close"], length=14)).astype(np.float64)

    adx_df = ta.adx(df["high"], df["low"], df["close"])
    adx    = _safe(adx_df.iloc[:, 0]).astype(np.float64) if adx_df is not None else np.zeros(len(df))

    ema_slope    = ema21 - np.roll(ema21, 5)
    vol_norm     = atr14 / (cl + 1e-8)
    vol_ma       = pd.Series(vol_norm).rolling(50, min_periods=1).mean().values
    high_vol_flag = vol_norm > (1.5 * vol_ma)

    trending      = adx > 25
    bull          = trending & (ema_slope > 0) & (ema21 > ema50)
    bear          = trending & (ema_slope < 0) & (ema21 < ema50)
    volatile      = high_vol_flag & ~trending
    ranging       = ~trending & ~high_vol_flag

    # Clear and write the canonical 6-class regime one-hot (FeatureSpec 43:49).
    # bull->uptrend, bear->downtrend, ranging->ranging, volatile->high_volatility.
    # news_driven / low_liquidity are not detectable from OHLCV alone (filled live).
    features[:, fs.REGIME] = 0.0
    features[ranging,  fs.regime_index("ranging")] = 1.0
    features[bull,     fs.regime_index("uptrend")] = 1.0
    features[bear,     fs.regime_index("downtrend")] = 1.0
    features[volatile, fs.regime_index("high_volatility")] = 1.0

    # Confidence = normalised ADX (0–1)
    features[:, fs.REGIME_CONFIDENCE] = np.clip(adx / 50.0, 0, 1).astype(np.float32)

    return features


def build_htf_features(df_5m: pd.DataFrame, df_1h: pd.DataFrame, df_4h: pd.DataFrame) -> np.ndarray:
    """
    Builds (N_5m, HTF_FEATURES=8) matrix.
    For each 5m bar, looks up the most recent completed 1h / 4h bar.
    Strictly backward-looking — no lookahead.

    1h features (4): rsi_norm, ema21_dist, macd_hist_norm, atr_norm
    4h features (4): rsi_norm, ema21_dist, trend_dir, atr_norm
    """
    n = len(df_5m)
    htf = np.zeros((n, HTF_FEATURES), dtype=np.float32)
    eps = 1e-8

    def _make_htf_signals(df_htf: pd.DataFrame) -> pd.DataFrame:
        cl = df_htf["close"]
        rsi_n    = (_safe(ta.rsi(cl, length=14), 50.0) - 50.0) / 50.0
        ema21    = _safe(ta.ema(cl, length=21))
        ema50    = _safe(ta.ema(cl, length=50))
        cl_v     = cl.values.astype(np.float32)
        ema21_d  = np.clip((cl_v - ema21) / (cl_v + eps), -0.1, 0.1)
        macd_df  = ta.macd(cl)
        if macd_df is not None and not macd_df.empty:
            mh = _safe(macd_df.iloc[:, 1] if macd_df.shape[1] > 1 else macd_df.iloc[:, 0])
        else:
            mh = np.zeros(len(df_htf), np.float32)
        macd_n   = np.clip(mh / (cl_v + eps) * 100, -1, 1)
        atr_n    = np.clip(_safe(ta.atr(df_htf["high"], df_htf["low"], cl, length=14)) / (cl_v + eps), 0, 0.1) * 10.0
        trend    = (ema21 > ema50).astype(np.float32) * 2 - 1  # +1 bull, -1 bear

        return pd.DataFrame({
            "timestamp": df_htf["timestamp"].values,
            "rsi_n": rsi_n, "ema21_d": ema21_d,
            "macd_n": macd_n, "atr_n": atr_n, "trend": trend
        })

    # CAUSAL mapping: an HTF candle's timestamp is its OPEN time, but its indicators (RSI/EMA/
    # MACD) use the candle's CLOSE — known only when the candle ENDS. Indexing by open-time made
    # the searchsorted-1 below map each 5m bar to the candle CONTAINING it, leaking up to one
    # full HTF period (48 bars for 4h) of the FUTURE — the signal-audit caught this as a
    # 0.73-0.82 AUC at H+12/H+48 that decayed exactly as the 4h candle's reach ran out. Re-index
    # by CLOSE time (open + period) so the lookup picks the last candle that has FULLY closed.
    def _by_close_time(sig_df: pd.DataFrame, src_df: pd.DataFrame) -> pd.DataFrame:
        period = pd.to_datetime(pd.Series(src_df["timestamp"])).diff().median()
        sig_df = sig_df.copy()
        sig_df["timestamp"] = pd.to_datetime(sig_df["timestamp"]) + period
        return sig_df.set_index("timestamp")

    sig1h = _by_close_time(_make_htf_signals(df_1h), df_1h)
    sig4h = _by_close_time(_make_htf_signals(df_4h), df_4h)

    ts5m = df_5m["timestamp"].values

    # For each 5m bar, find the last HTF bar that CLOSED strictly before it (causal)
    for i, ts in enumerate(ts5m):
        # 1h
        idx1h = sig1h.index.searchsorted(ts, side="left") - 1
        if idx1h >= 0:
            row = sig1h.iloc[idx1h]
            htf[i, 0] = row["rsi_n"]
            htf[i, 1] = row["ema21_d"]
            htf[i, 2] = row["macd_n"]
            htf[i, 3] = row["atr_n"]
        # 4h
        idx4h = sig4h.index.searchsorted(ts, side="left") - 1
        if idx4h >= 0:
            row = sig4h.iloc[idx4h]
            htf[i, 4] = row["rsi_n"]
            htf[i, 5] = row["ema21_d"]
            htf[i, 6] = row["trend"]
            htf[i, 7] = row["atr_n"]

    return htf


def apply_rolling_zscore(features: np.ndarray, window: int = ZSCORE_WIN, min_periods: int = 50) -> np.ndarray:
    """
    Per-feature rolling z-score normalisation.
    Computes mean/std over a backward window only — zero future leakage.
    Columns that are clearly binary/one-hot (low variance) are skipped.
    """
    df = pd.DataFrame(features.astype(np.float64))
    stds = df.std()
    skip_cols = stds[stds < 0.05].index.tolist()  # skip binary/constant columns

    roll_mean = df.rolling(window=window, min_periods=min_periods).mean()
    roll_std  = df.rolling(window=window, min_periods=min_periods).std()

    normed = (df - roll_mean) / (roll_std + 1e-8)
    normed[skip_cols] = df[skip_cols]  # restore binary columns as-is
    normed.fillna(0.0, inplace=True)

    return normed.values.astype(np.float32)


def assemble_matrix(df5m, df1h, df4h, news_mat=None, earnings_mat=None) -> np.ndarray:
    """The CANONICAL ``(N, fs.INPUT)`` feature matrix — the SINGLE assembly shared by
    the offline trainer, the backtest, AND the live agent so the three can never drift.

    Order MUST match ``backend/signals/feature_spec.py`` exactly:
        base(62) + htf(8)  →  rolling z-score  →  + news_embed(16)  →  + earnings(4)

    ``news_mat`` (N, NEWS_EMBED_DIM) and ``earnings_mat`` (N, EARNINGS_DIM) are appended
    AFTER z-scoring (raw — they are not z-scored). Pass ``None`` to fill that block with
    zeros, which is exactly what "no aligned news / not a stock" produces both offline
    (PRETRAIN_*_ALIGN off) and live (no fresh news). The WIDTH is always ``fs.INPUT``
    regardless, so a checkpoint's input_size can be hard-validated against it.
    """
    base = build_feature_matrix(df5m)            # (N, 62)
    base = detect_regime(df5m, base)             # fills regime one-hot slots (OHLCV-derived)
    htf = build_htf_features(df5m, df1h, df4h)   # (N, 8)
    combined = apply_rolling_zscore(np.concatenate([base, htf], axis=1))   # (N, 70)
    n = combined.shape[0]
    if news_mat is None:
        news_mat = np.zeros((n, fs.NEWS_EMBED_DIM), dtype=np.float32)
    if earnings_mat is None:
        earnings_mat = np.zeros((n, fs.EARNINGS_DIM), dtype=np.float32)
    combined = np.concatenate(
        [combined, np.asarray(news_mat, dtype=np.float32),
         np.asarray(earnings_mat, dtype=np.float32)], axis=1)             # (N, 90)
    if combined.shape[1] != fs.INPUT:
        raise AssertionError(
            f"assembled feature width {combined.shape[1]} != fs.INPUT {fs.INPUT} — "
            f"feature_spec.py and the assembly have drifted.")
    return combined


# Multi-scale momentum / trend / position-in-range features. The signal audit found these
# carry the HIGHEST rank-IC of anything tested (mom_864 ≈ 0.056 > the best existing feature),
# because they capture 1–5 day momentum the short EMAs/RSI in the base block miss. Formulas
# match scripts/signal_audit.candidate_features exactly (the audited columns), so the measured
# IC carries over. All causal (trailing rolling / backward returns).
MOMENTUM_NAMES = ["mom_48", "mom_288", "mom_864", "mom_2016",
                  "trend_288", "trend_1440", "range_288", "range_1440", "vol_regime_z"]


def momentum_features(df: pd.DataFrame) -> np.ndarray:
    """(N, 9) causal multi-scale momentum/trend/range features — see MOMENTUM_NAMES."""
    close = df["close"].to_numpy(np.float64)
    high = (df["high"] if "high" in df else df["close"]).to_numpy(np.float64)
    low = (df["low"] if "low" in df else df["close"]).to_numpy(np.float64)
    n = close.shape[0]
    idx = np.arange(n)
    cols = []
    for w in (48, 288, 864, 2016):            # log return over w bars (4h, 1d, 3d, 1w)
        cols.append(np.nan_to_num(np.log(close / np.roll(close, w)), posinf=0.0, neginf=0.0)
                    * (idx >= w))
    for w in (288, 1440):                      # trend: (close - SMA_w) / close (1d, 5d)
        sma = pd.Series(close).rolling(w, min_periods=w // 2).mean().to_numpy()
        cols.append(np.nan_to_num((close - sma) / (close + 1e-9)))
    for w in (288, 1440):                      # position in rolling high-low range (1d, 5d)
        hi = pd.Series(high).rolling(w, min_periods=w // 2).max().to_numpy()
        lo = pd.Series(low).rolling(w, min_periods=w // 2).min().to_numpy()
        cols.append(np.nan_to_num((close - lo) / (hi - lo + 1e-9)))
    tr = np.abs(np.diff(close, prepend=close[:1])) / (close + 1e-9)   # vol regime z-score
    atr = pd.Series(tr).rolling(96, min_periods=20).mean()
    z = (atr - atr.rolling(1000, min_periods=100).mean()) / (atr.rolling(1000, min_periods=100).std() + 1e-9)
    cols.append(np.nan_to_num(z.to_numpy()))
    return np.column_stack(cols).astype(np.float32)


# The Phase-2 hybrid feature contract: the 49 ACTIVE base features (the ~41 stubbed/dead ones
# pruned) + the 9 momentum features, all consistently rolling-z-scored. Distinct from the
# legacy 90-wide vector the LSTM uses (kept for A/B), so this can evolve without touching it.
HYBRID_FEATURE_NAMES = [fs.feature_name(i) for i in fs.ACTIVE_FEATURE_INDICES] + MOMENTUM_NAMES


def build_hybrid_matrix(df5m, df1h, df4h) -> np.ndarray:
    """(N, len(HYBRID_FEATURE_NAMES)=58) lean hybrid feature matrix: pruned active base
    features + z-scored multi-scale momentum. The single feature contract the Phase-2
    QuantileGBM / QuantileTCN train and serve on (offline == live, same as assemble_matrix)."""
    full = assemble_matrix(df5m, df1h, df4h)                 # (N, fs.INPUT) canonical
    active = full[:, fs.ACTIVE_FEATURE_INDICES]              # (N, 49) drop stubbed/dead
    mom = apply_rolling_zscore(momentum_features(df5m))      # (N, 9) z-scored to match base scale
    out = np.concatenate([active, mom], axis=1).astype(np.float32)
    if out.shape[1] != len(HYBRID_FEATURE_NAMES):
        raise AssertionError(
            f"hybrid width {out.shape[1]} != len(HYBRID_FEATURE_NAMES) {len(HYBRID_FEATURE_NAMES)}")
    return out
