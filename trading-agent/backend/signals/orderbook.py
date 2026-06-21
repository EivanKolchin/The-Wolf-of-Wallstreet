"""Trade-microstructure features — replaces the old L2-orderbook block [35:42].

All 8 microstructure features are now computed from the PUBLIC trade stream 
(aggressive buy/sell flags) instead of L2 orderbook depth, which:
  • requires no historical L2 data for training (trade data is available from Binance);
  • works identically offline (pretrain) and live (inference);
  • produces the same features at train time and inference time — no more 
    NN_ZERO_ORDERBOOK hack that zeroed live features the model never trained on.

Feature layout in the 8 slots [35:42]:
  35  aggressive_buy_ratio   — taker buy volume / total volume (rolling 100 trades)
  36  buy_concentration      — std of last 20 buy sizes / mean (concentration signal)
  37  sell_concentration     — std of last 20 sell sizes / mean
  38  cvd_slope              — cumulative volume delta slope (KEEP from original)
  39  cvd_divergence         — CVD vs price divergence (KEEP from original)
  40  whale_activity         — trade size / rolling_median(200 trades) — % whale trades
  41  bull_flow_ratio        — aggressive buy ratio over last 3 trades (sweep signal)
  42  bear_flow_ratio        — aggressive sell ratio over last 3 trades (sweep signal)
"""
import numpy as np
from typing import List, Dict
from structlog import get_logger

log = get_logger("signals.orderbook")


def _safe_trades(trades: list):
    """Return trades list or empty list if None/invalid."""
    return trades if isinstance(trades, list) and len(trades) > 0 else []


def _taker_volumes(trades: list):
    """Split last 100 trades into buy/sell volumes from is_buyer_maker flag.
    is_buyer_maker = True  → aggressive sell (taker sold into bid)
    is_buyer_maker = False → aggressive buy  (taker bought from ask)
    Returns (buy_vol, sell_vol, total_vol, n_trades)."""
    trades = _safe_trades(trades)
    if not trades:
        return 0.0, 0.0, 0.0, 0
    recent = trades[-100:]
    buy_vol = 0.0
    sell_vol = 0.0
    for t in recent:
        qty = float(t.get("qty", 0.0) or 0.0)
        if t.get("is_buyer_maker", False):
            sell_vol += qty
        else:
            buy_vol += qty
    total = buy_vol + sell_vol
    return buy_vol, sell_vol, total, len(recent)


def aggressive_buy_ratio(trades: list) -> float:
    """Slot 35: fraction of total volume that is taker-buys (aggressive).
    Ranges [0, 1]. Above 0.65 = buying pressure, below 0.35 = selling pressure."""
    buy, sell, total, _ = _taker_volumes(trades)
    if total < 1e-12:
        return 0.5  # neutral when no data
    return float(np.clip(buy / total, 0.0, 1.0))


def _concentration(trades: list, is_buy: bool) -> float:
    """Slot 36/37: std/mean of last 20 trade sizes for buys or sells.
    Low (<0.5) = uniform flow, High (>1.5) = lumpy/whale flow."""
    recent = _safe_trades(trades)
    if not recent:
        return 0.5
    sizes = []
    for t in recent[-100:]:
        if is_buy == (not t.get("is_buyer_maker", True)):
            sizes.append(float(t.get("qty", 0.0) or 0.0))
    if len(sizes) < 3:
        return 0.5
    sizes = np.array(sizes[-20:])
    mean = float(sizes.mean())
    if mean < 1e-12:
        return 0.5
    std = float(sizes.std(ddof=1))
    return float(np.clip(std / mean, 0.0, 3.0) / 3.0)  # normalize to [0, 1]


def buy_concentration(trades: list) -> float:
    """Slot 36: size concentration among buys — low = uniform, high = chunky."""
    return _concentration(trades, is_buy=True)


def sell_concentration(trades: list) -> float:
    """Slot 37: size concentration among sells."""
    return _concentration(trades, is_buy=False)


def calculate_cvd(trades: List[Dict], window: int = 100) -> dict:
    """Slots 38-39: KEPT from original implementation. Cumulative Volume Delta
    slope + divergence. Pure trade-stream, no L2 dependence."""
    trades = _safe_trades(trades)
    if len(trades) < 2:
        return {"cvd_slope": 0.0, "cvd_divergence": 0.0}
    try:
        recent_trades = trades[-window:]
        cvd_series = []
        current_cvd = 0.0
        for t in recent_trades:
            qty = float(t.get("qty", 0.0))
            is_buyer_maker = t.get("is_buyer_maker", False)
            delta = -qty if is_buyer_maker else qty
            current_cvd += delta
            cvd_series.append(current_cvd)
        cvd_array = np.array(cvd_series)
        n_points = len(cvd_array)
        if n_points < 2:
            return {"cvd_slope": 0.0, "cvd_divergence": 0.0}
        x = np.arange(n_points)
        slope, _ = np.polyfit(x, cvd_array, 1)
        cvd_std = float(np.std(cvd_array))
        if cvd_std > 0:
            cvd_slope_val = float(np.clip(slope / cvd_std, -1.0, 1.0))
        else:
            cvd_slope_val = 0.0
        last_price = float(recent_trades[-1].get("price", 0.0))
        first_price = float(recent_trades[0].get("price", 0.0))
        price_change = last_price - first_price
        if cvd_slope_val > 0 and price_change < 0:
            cvd_divergence_val = 1.0
        elif cvd_slope_val < 0 and price_change > 0:
            cvd_divergence_val = -1.0
        else:
            cvd_divergence_val = 0.0
        return {"cvd_slope": cvd_slope_val, "cvd_divergence": cvd_divergence_val}
    except Exception as e:
        log.warning("cvd calculation failed", error=str(e))
        return {"cvd_slope": 0.0, "cvd_divergence": 0.0}


def detect_whale_activity(trades: List[Dict], window: int = 200) -> float:
    """Slot 40: fraction of recent trades that exceed 3x the rolling median size.
    More robust than the old 5x-mean approach (median resists outlier pull)."""
    trades = _safe_trades(trades)
    if len(trades) < 20:
        return 0.0
    try:
        recent = trades[-window:]
        qtys = np.array([float(t.get("qty", 0.0) or 0.0) for t in recent])
        med = float(np.median(qtys))
        if med < 1e-12:
            return 0.0
        last_50 = qtys[-50:]
        whale_count = float(np.sum(last_50 > 3.0 * med))
        return float(min(whale_count / 10.0, 1.0))  # cap at 10 whales out of 50
    except Exception as e:
        log.warning("whale detection failed", error=str(e))
        return 0.0


def _flow_ratio(trades: list, lookback: int = 3) -> float:
    """Slots 41-42 helper: aggressive buy ratio over the last N trades.
    >0.8 = strong buying pressure, <0.2 = strong selling pressure."""
    recent = _safe_trades(trades)
    if len(recent) < lookback:
        return 0.5
    window = recent[-lookback:]
    buys = sum(1 for t in window if not t.get("is_buyer_maker", True))
    return float(np.clip(buys / max(len(window), 1), 0.0, 1.0))


def bull_flow_ratio(trades: list) -> float:
    """Slot 41: aggressive buy ratio over last 3 trades — fast sweep signal."""
    return _flow_ratio(trades, lookback=3)


def bear_flow_ratio(trades: list) -> float:
    """Slot 42: aggressive sell ratio = 1 - bull_flow_ratio."""
    return 1.0 - _flow_ratio(trades, lookback=3)


def build_orderbook_feature_dict(bids: list, asks: list, trades: List[Dict],
                                  klines: list = None, sr_levels: list = None) -> dict:
    """Build all 8 trade-microstructure features.

    Args:
        bids, asks: KEPT for API compatibility (no longer used for computation)
        trades: list of trade dicts with 'qty', 'is_buyer_maker', 'price'
        klines, sr_levels: KEPT for API compatibility (no longer used)

    Returns:
        dict with keys matching the 8 feature slots [35:42]
    """
    features = {
        "book_imbalance": aggressive_buy_ratio(trades),
        "bid_depth_ratio": buy_concentration(trades),
        "ask_depth_ratio": sell_concentration(trades),
    }
    try:
        cvd = calculate_cvd(trades)
        features.update(cvd)
    except Exception as e:
        log.warning("cvd calculation failed", error=str(e))
        features.update({"cvd_slope": 0.0, "cvd_divergence": 0.0})

    features["whale_activity"] = detect_whale_activity(trades)
    features["bullish_sweep_strength"] = bull_flow_ratio(trades)
    features["bearish_sweep_strength"] = bear_flow_ratio(trades)

    return features
