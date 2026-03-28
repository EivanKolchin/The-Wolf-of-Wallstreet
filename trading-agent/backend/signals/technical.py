import numpy as np
import pandas as pd
import talib
from structlog import get_logger

log = get_logger("signals.technical")

def _clip_scale(val: float, min_val: float, max_val: float, target_min: float, target_max: float) -> float:
    # Handle NaNs
    if pd.isna(val):
        return (target_min + target_max) / 2.0
    val_clipped = max(min_val, min(val, max_val))
    # normalise to [0, 1] first
    norm = (val_clipped - min_val) / (max_val - min_val)
    # then scale to [target_min, target_max]
    return target_min + norm * (target_max - target_min)


def calculate_moving_averages(df: pd.DataFrame) -> dict:
    close = df['close']
    ema_9 = pd.Series(talib.EMA(close.values, timeperiod=9), index=close.index)
    ema_21 = pd.Series(talib.EMA(close.values, timeperiod=21), index=close.index)
    ema_50 = pd.Series(talib.EMA(close.values, timeperiod=50), index=close.index)
    ema_200 = pd.Series(talib.EMA(close.values, timeperiod=200), index=close.index)
    vwap = (df['volume'] * (df['high'] + df['low'] + close) / 3).cumsum() / df['volume'].cumsum()
    
    current_close = close.iloc[-1]
    
    def calc_dist(ema_series):
        if ema_series is None or pd.isna(ema_series.iloc[-1]):
            return 0.0
        dist = (current_close - ema_series.iloc[-1]) / ema_series.iloc[-1]
        return np.clip(dist, -0.2, 0.2)

    try:
        if ema_50 is not None and ema_200 is not None and not pd.isna(ema_50.iloc[-1]) and not pd.isna(ema_200.iloc[-1]):
            golden_cross = 1.0 if ema_50.iloc[-1] > ema_200.iloc[-1] else 0.0
        else:
            golden_cross = 0.5
    except Exception:
        golden_cross = 0.5

    try:
        if vwap is not None and not pd.isna(vwap.iloc[-1]):
            vwap_dist = (current_close - vwap.iloc[-1]) / vwap.iloc[-1]
            vwap_dist_scaled = _clip_scale(vwap_dist, -0.05, 0.05, -1.0, 1.0)
        else:
            vwap_dist_scaled = 0.0
    except Exception:
        vwap_dist_scaled = 0.0
        
    return {
        "ema_9_dist": calc_dist(ema_9),
        "ema_21_dist": calc_dist(ema_21),
        "ema_50_dist": calc_dist(ema_50),
        "ema_200_dist": calc_dist(ema_200),
        "golden_cross": golden_cross,
        "vwap_dist": vwap_dist_scaled
    }

def calculate_momentum(df: pd.DataFrame) -> dict:
    close = df['close']
    high = df['high']
    low = df['low']
    
    rsi = pd.Series(talib.RSI(close.values, timeperiod=14), index=close.index)
    m, s, h = talib.MACD(close.values); macd = pd.DataFrame({'MACD_12_26_9': m, 'MACDh_12_26_9': h, 'MACDs_12_26_9': s}, index=close.index)
    f, s = talib.STOCHRSI(close.values, timeperiod=14); stoch_rsi_res = pd.DataFrame({'STOCHRSIk_14_14_3_3': f*100, 'STOCHRSId_14_14_3_3': s*100}, index=close.index)
    adx_res = pd.DataFrame({'ADX_14': talib.ADX(high.values, low.values, close.values, timeperiod=14)}, index=close.index)
    atr = pd.Series(talib.ATR(high.values, low.values, close.values, timeperiod=14), index=close.index)

    # Latest vals
    v_rsi = rsi.iloc[-1] if (rsi is not None and not pd.isna(rsi.iloc[-1])) else 50.0
    v_atr = atr.iloc[-1] if (atr is not None and not pd.isna(atr.iloc[-1]) and atr.iloc[-1] != 0) else 1.0
    
    # MACD lines
    try:
        macd_val = macd.iloc[-1, 0] # MACD line
        macd_hist = macd.iloc[-1, 1] # MACD histogram
        macd_norm = _clip_scale(macd_val / v_atr, -2.0, 2.0, -1.0, 1.0)
        macd_hist_norm = _clip_scale(macd_hist / v_atr, -2.0, 2.0, -1.0, 1.0)
    except Exception:
        macd_norm = 0.0
        macd_hist_norm = 0.0

    # Stoch RSI (often returns string-formatted column names)
    try:
        v_stoch_rsi = stoch_rsi_res.iloc[-1, 0] if stoch_rsi_res is not None else 50.0
    except Exception:
        v_stoch_rsi = 50.0

    # ADX
    try:
        v_adx = adx_res.iloc[-1, 0] if adx_res is not None else 0.0
    except Exception:
        v_adx = 0.0

    # RSI Divergence logic (Last 10 bars)
    try:
        if len(close) >= 10 and rsi is not None:
            last_10_close = close.iloc[-10:]
            last_10_rsi = rsi.iloc[-10:]
            # Price new high but RSI didn't
            if last_10_close.iloc[-1] >= last_10_close.max() and last_10_rsi.iloc[-1] < last_10_rsi.max():
                rsi_div = 1.0
            # Price new low but RSI didn't
            elif last_10_close.iloc[-1] <= last_10_close.min() and last_10_rsi.iloc[-1] > last_10_rsi.min():
                rsi_div = -1.0
            else:
                rsi_div = 0.0
        else:
            rsi_div = 0.0
    except Exception:
        rsi_div = 0.0

    return {
        "rsi": np.clip(v_rsi / 100.0, 0.0, 1.0),
        "macd_norm": macd_norm,
        "macd_hist_norm": macd_hist_norm,
        "stoch_rsi": np.clip(v_stoch_rsi / 100.0, 0.0, 1.0),
        "adx_norm": np.clip(v_adx / 100.0, 0.0, 1.0),
        "rsi_divergence": rsi_div
    }

def calculate_volatility(df: pd.DataFrame) -> dict:
    if len(df) < 100:
        return {"atr_norm": 0.5, "bb_width_norm": 0.5, "bb_pct_b": 0.5}

    close = df['close']
    high = df['high']
    low = df['low']

    atr = pd.Series(talib.ATR(high.values, low.values, close.values, timeperiod=14), index=close.index)
    u, m, l = talib.BBANDS(close.values, timeperiod=20); bbands = pd.DataFrame({'BBL_20_2.0': l, 'BBM_20_2.0': m, 'BBU_20_2.0': u}, index=close.index)

    try:
        atr_pct = atr / close
        rolling_atr_pct = atr_pct.rolling(window=100)
        atr_pct_rank = rolling_atr_pct.apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1]).iloc[-1]
        v_atr_norm = np.clip(atr_pct_rank, 0.0, 1.0)
        if pd.isna(v_atr_norm):
            v_atr_norm = 0.5
    except Exception:
        v_atr_norm = 0.5

    try:
        bb_lower = bbands.iloc[:, 0] # BBL
        bb_mid = bbands.iloc[:, 1]   # BBM
        bb_upper = bbands.iloc[:, 2] # BBU
        
        bb_width = (bb_upper - bb_lower) / bb_mid
        rolling_bb_width = bb_width.rolling(window=100)
        bb_width_rank = rolling_bb_width.apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1]).iloc[-1]
        v_bb_width_norm = np.clip(bb_width_rank, 0.0, 1.0)
        if pd.isna(v_bb_width_norm):
            v_bb_width_norm = 0.5

        v_bb_pct_b = (close.iloc[-1] - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1])
        v_bb_pct_b = np.clip(v_bb_pct_b, 0.0, 1.0)
    except Exception:
        v_bb_width_norm = 0.5
        v_bb_pct_b = 0.5

    return {
        "atr_norm": v_atr_norm,
        "bb_width_norm": v_bb_width_norm,
        "bb_pct_b": v_bb_pct_b
    }

def calculate_volume(df: pd.DataFrame) -> dict:
    volume = df['volume']
    close = df['close']

    sma_vol_20 = pd.Series(talib.SMA(volume.values, timeperiod=20), index=volume.index)
    
    try:
        curr_vol = volume.iloc[-1]
        sma_vol = sma_vol_20.iloc[-1]
        if pd.isna(sma_vol) or sma_vol == 0:
            vol_ratio = 1.0
        else:
            vol_ratio = curr_vol / sma_vol
        
        volume_ratio_norm = _clip_scale(vol_ratio, 0.0, 5.0, 0.0, 1.0)
    except Exception:
        volume_ratio_norm = 0.2 # default roughly neutral-low vol

    try:
        obv = pd.Series(talib.OBV(close.values, volume.values), index=close.index)
        if len(obv) >= 10:
            last_10_obv = obv.iloc[-10:].values
            # linear regression slope
            x = np.arange(10)
            slope, _ = np.polyfit(x, last_10_obv, 1)
            
            std_obv = np.std(last_10_obv)
            if std_obv > 0:
                obv_slope_norm = slope / std_obv
            else:
                obv_slope_norm = 0.0
            
            obv_slope_norm = np.clip(obv_slope_norm, -1.0, 1.0)
        else:
            obv_slope_norm = 0.0
    except Exception:
        obv_slope_norm = 0.0

    return {
        "volume_ratio": volume_ratio_norm,
        "obv_slope": obv_slope_norm
    }

def calculate_fibonacci(df: pd.DataFrame, lookback: int = 50) -> dict:
    if len(df) < lookback:
        return {"fib_nearest_level_pct": 0.5, "fib_distance": 0.0, "fib_strength": 0.6}
        
    try:
        recent_df = df.iloc[-lookback:]
        high_price = recent_df['high'].max()
        low_price = recent_df['low'].min()
        close = df['close'].iloc[-1]

        diff = high_price - low_price
        if diff == 0:
             return {"fib_nearest_level_pct": 0.5, "fib_distance": 0.0, "fib_strength": 0.6}

        fib_levels = {
            23.6: high_price - 0.236 * diff,
            38.2: high_price - 0.382 * diff,
            50.0: high_price - 0.500 * diff,
            61.8: high_price - 0.618 * diff,
            78.6: high_price - 0.786 * diff
        }

        # Find nearest
        nearest_level = None
        min_dist_abs = float('inf')
        nearest_price = 0.0

        for level, price in fib_levels.items():
            dist = abs(close - price)
            if dist < min_dist_abs:
                min_dist_abs = dist
                nearest_level = level
                nearest_price = price

        fib_nearest_level_pct = nearest_level / 100.0
        
        raw_dist = (close - nearest_price) / close
        fib_distance = _clip_scale(raw_dist, -0.02, 0.02, -1.0, 1.0)
        
        fib_strength = 1.0 if nearest_level in [61.8, 38.2] else 0.6

        return {
            "fib_nearest_level_pct": fib_nearest_level_pct,
            "fib_distance": fib_distance,
            "fib_strength": fib_strength
        }
    except Exception:
        return {"fib_nearest_level_pct": 0.5, "fib_distance": 0.0, "fib_strength": 0.6}


def calculate_patterns(df: pd.DataFrame) -> dict:
    try:
        o = df['open'].values
        h = df['high'].values
        l = df['low'].values
        c = df['close'].values

        # Proxy engulfing with general engulf since TA-lib doesn't have strict separate bullish/bearish engulfing functions
        # CDLENGULFING returns > 0 for bullish, < 0 for bearish.
        engulfing = talib.CDLENGULFING(o, h, l, c)[-1]
        bullish_engulf = 1.0 if engulfing > 0 else 0.0
        bearish_engulf = 1.0 if engulfing < 0 else 0.0

        hammer = 1.0 if talib.CDLHAMMER(o, h, l, c)[-1] > 0 else 0.0
        inv_hammer = 1.0 if talib.CDLINVERTEDHAMMER(o, h, l, c)[-1] > 0 else 0.0
        morning_star = 1.0 if talib.CDLMORNINGSTAR(o, h, l, c)[-1] > 0 else 0.0
        evening_star = 1.0 if talib.CDLEVENINGSTAR(o, h, l, c)[-1] < 0 else 0.0 # Evening stars are bearish
        doji = 1.0 if talib.CDLDOJI(o, h, l, c)[-1] > 0 else 0.0
        spinning_top = 1.0 if talib.CDLSPINNINGTOP(o, h, l, c)[-1] > 0 else 0.0
        marubozu = 1.0 if talib.CDLMARUBOZU(o, h, l, c)[-1] > 0 else 0.0
        three_white_soldiers = 1.0 if talib.CDL3WHITESOLDIERS(o, h, l, c)[-1] > 0 else 0.0

        patterns = [
            bullish_engulf, bearish_engulf,
            hammer, inv_hammer,
            morning_star, evening_star,
            doji, spinning_top,
            marubozu, three_white_soldiers
        ]
        return {"pattern_flags": patterns}
    except Exception:
        # 10 zeroes if pattern matching fails
        return {"pattern_flags": [0.0] * 10}


def build_technical_feature_dict(df: pd.DataFrame) -> dict:
    features = {}

    def safe_calc(func, default_keys, default_val=0.5):
        try:
            res = func(df)
            return res
        except Exception as e:
            log.error(f"Failed to calculate {func.__name__}", error=str(e))
            return {k: default_val for k in default_keys}

    features.update(safe_calc(
        calculate_moving_averages,
        ["ema_9_dist", "ema_21_dist", "ema_50_dist", "ema_200_dist", "golden_cross", "vwap_dist"]
    ))

    features.update(safe_calc(
        calculate_momentum,
        ["rsi", "macd_norm", "macd_hist_norm", "stoch_rsi", "adx_norm", "rsi_divergence"]
    ))

    features.update(safe_calc(
        calculate_volatility,
        ["atr_norm", "bb_width_norm", "bb_pct_b"]
    ))

    features.update(safe_calc(
        calculate_volume,
        ["volume_ratio", "obv_slope"]
    ))

    features.update(safe_calc(
        calculate_fibonacci,
        ["fib_nearest_level_pct", "fib_distance", "fib_strength"]
    ))

    # Pattern default needs to be the list type
    try:
        p_res = calculate_patterns(df)
        features.update(p_res)
    except Exception as e:
        log.error("Failed to calculate calculate_patterns", error=str(e))
        features["pattern_flags"] = [0.0] * 10

    return features