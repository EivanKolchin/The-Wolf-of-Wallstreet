import numpy as np
from typing import List, Dict
from structlog import get_logger

log = get_logger("signals.orderbook")

def calculate_book_imbalance(bids: list, asks: list, depth: int = 10) -> float:
    try:
        if not bids or not asks:
            return 0.0

        top_bids = bids[:depth]
        top_asks = asks[:depth]

        bid_vol = sum(float(qty) for _, qty in top_bids)
        ask_vol = sum(float(qty) for _, qty in top_asks)

        total_vol = bid_vol + ask_vol
        if total_vol == 0:
            return 0.0

        imbalance = (bid_vol - ask_vol) / total_vol
        return float(np.clip(imbalance, -1.0, 1.0))
    except Exception:
        return 0.0

def calculate_depth_ratios(bids: list, asks: list) -> dict:
    try:
        def get_ratio(levels: list) -> float:
            if not levels:
                return 0.0
            top_5_vol = sum(float(qty) for _, qty in levels[:5])
            top_20_vol = sum(float(qty) for _, qty in levels[:20])
            if top_20_vol == 0:
                return 0.0
            return float(np.clip(top_5_vol / top_20_vol, 0.0, 1.0))

        return {
            "bid_depth_ratio": get_ratio(bids),
            "ask_depth_ratio": get_ratio(asks)
        }
    except Exception:
        return {
            "bid_depth_ratio": 0.5,
            "ask_depth_ratio": 0.5
        }

def calculate_cvd(trades: List[Dict], window: int = 100) -> dict:
    if len(trades) < 2:
        return {"cvd_slope": 0.0, "cvd_divergence": 0.0}

    try:
        recent_trades = trades[-window:]
        
        cvd_series = []
        current_cvd = 0.0
        
        for t in recent_trades:
            qty = float(t.get("qty", 0.0))
            is_buyer_maker = t.get("is_buyer_maker", False)
            
            # buyer_maker == True means aggressive sell hit the bid
            # buyer_maker == False means aggressive buy hit the ask
            delta = -qty if is_buyer_maker else qty
            current_cvd += delta
            cvd_series.append(current_cvd)

        cvd_array = np.array(cvd_series)
        
        n_points = len(cvd_array)
        if n_points < 2:
            return {"cvd_slope": 0.0, "cvd_divergence": 0.0}

        x = np.arange(n_points)
        slope, _ = np.polyfit(x, cvd_array, 1)
        
        cvd_std = np.std(cvd_array)
        if cvd_std > 0:
            cvd_slope = slope / cvd_std
        else:
            cvd_slope = 0.0
            
        cvd_slope = float(np.clip(cvd_slope, -1.0, 1.0))

        # Divergence
        last_price = float(recent_trades[-1].get("price", 0.0))
        first_price = float(recent_trades[0].get("price", 0.0))
        price_change = last_price - first_price

        # bearish divergence (cvd going up, price down) = 1.0
        if cvd_slope > 0 and price_change < 0:
            cvd_divergence = 1.0
        # bullish divergence (cvd going down, price up) = -1.0
        elif cvd_slope < 0 and price_change > 0:
            cvd_divergence = -1.0
        else:
            cvd_divergence = 0.0

        return {
            "cvd_slope": cvd_slope,
            "cvd_divergence": cvd_divergence
        }
    except Exception:
        return {"cvd_slope": 0.0, "cvd_divergence": 0.0}

def detect_whale_activity(trades: List[Dict], window: int = 200) -> float:
    if not trades:
        return 0.0

    try:
        recent_window = trades[-window:]
        if not recent_window:
            return 0.0
            
        qtys = [float(t.get("qty", 0.0)) for t in recent_window]
        avg_qty = float(np.mean(qtys))
        
        if avg_qty == 0:
            return 0.0

        last_50 = qtys[-50:]
        whale_count = sum(1 for q in last_50 if q > (5 * avg_qty))
        
        return float(min(whale_count / 5.0, 1.0))
    except Exception:
        return 0.0

def detect_liquidity_sweep(klines: List[Dict], sr_levels: List[float]) -> dict:
    if not klines or not sr_levels:
        return {
            "bullish_sweep_strength": 0.0,
            "bearish_sweep_strength": 0.0,
            "sweep_detected": 0.0
        }

    try:
        last_3 = klines[-3:]
        
        max_bull_strength = 0.0
        max_bear_strength = 0.0

        for k in last_3:
            low = float(k.get("low", 0.0))
            high = float(k.get("high", 0.0))
            close = float(k.get("close", 0.0))
            
            for level in sr_levels:
                # Bullish sweep: Low went below level, but closed above it
                if low < level and close > level:
                    strength = (close - level) / close
                    max_bull_strength = max(max_bull_strength, strength)
                
                # Bearish sweep: High went above level, but closed below it
                if high > level and close < level:
                    # Positive strength magnitude representing distance below
                    strength = (level - close) / close
                    max_bear_strength = max(max_bear_strength, strength)

        # Normalize strengths arbitrarily to [0, 1]. A 1% move from a sweep is huge in forex/crypto.
        # Let's say a 0.5% (0.005) divergence equates to max strength 1.0. 
        # (The spec doesn't give a specific normalization max, we just bound to 1.0)
        norm_bull_str = float(np.clip(max_bull_strength * 200, 0.0, 1.0)) 
        norm_bear_str = float(np.clip(max_bear_strength * 200, 0.0, 1.0))

        return {
            "bullish_sweep_strength": norm_bull_str,
            "bearish_sweep_strength": norm_bear_str,
            "sweep_detected": 1.0 if (norm_bull_str > 0 or norm_bear_str > 0) else 0.0
        }
    except Exception:
        return {
            "bullish_sweep_strength": 0.0,
            "bearish_sweep_strength": 0.0,
            "sweep_detected": 0.0
        }

def build_orderbook_feature_dict(bids: list, asks: list, trades: List[Dict], klines: List[Dict], sr_levels: List[float]) -> dict:
    features = {}

    def safe_exec(func, kwargs, defaults):
        try:
            return func(**kwargs)
        except Exception as e:
            log.error(f"Failed to calculate {func.__name__}", error=str(e))
            return defaults

    f_imb = safe_exec(
        calculate_book_imbalance, 
        {"bids": bids, "asks": asks}, 
        0.0  # single float fallback
    )
    features["book_imbalance"] = f_imb if isinstance(f_imb, float) else 0.0

    f_depth = safe_exec(
        calculate_depth_ratios, 
        {"bids": bids, "asks": asks}, 
        {"bid_depth_ratio": 0.5, "ask_depth_ratio": 0.5}
    )
    features.update(f_depth)

    f_cvd = safe_exec(
        calculate_cvd, 
        {"trades": trades}, 
        {"cvd_slope": 0.0, "cvd_divergence": 0.0}
    )
    features.update(f_cvd)

    f_whale = safe_exec(
        detect_whale_activity, 
        {"trades": trades}, 
        0.0
    )
    features["whale_activity"] = f_whale if isinstance(f_whale, float) else 0.0

    f_sweep = safe_exec(
        detect_liquidity_sweep, 
        {"klines": klines, "sr_levels": sr_levels}, 
        {"bullish_sweep_strength": 0.0, "bearish_sweep_strength": 0.0, "sweep_detected": 0.0}
    )
    features.update(f_sweep)

    return features
