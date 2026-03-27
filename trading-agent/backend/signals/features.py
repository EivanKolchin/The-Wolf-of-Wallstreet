import hashlib
import json
import logging
import math
from datetime import datetime
from typing import Any, List, Optional

import numpy as np
import pandas as pd
from structlog import get_logger

from memory.redis_client import FeatureCache, NewsImpact

log = get_logger("signals.features")


class FeatureVectorBuilder:
    def __init__(self, redis_client: Any, technical_calculator: Any, orderbook_calculator: Any):
        self.redis_client = redis_client
        self.feature_cache = FeatureCache(redis_client)
        self.tech_calc = technical_calculator
        self.ob_calc = orderbook_calculator

    def _clip_scale(self, val: float, min_val: float, max_val: float, target_min: float, target_max: float) -> float:
        if pd.isna(val):
            return 0.0
        val_clipped = max(min_val, min(val, max_val))
        norm = (val_clipped - min_val) / (max_val - min_val)
        return target_min + norm * (target_max - target_min)

    async def build(
        self,
        symbol: str,
        df: pd.DataFrame,
        bids: list,
        asks: list,
        trades: list,
        sr_levels: List[float],
        regime: str,
        regime_confidence: float,
        news_impact: Optional[NewsImpact]
    ) -> np.ndarray:
        
        vec = np.zeros(62, dtype=np.float32)

        try:
            # Helper to safely assign values and log errors if they fail
            def safe_assign(idx: int, val: float, name: str):
                try:
                    if pd.isna(val):
                        vec[idx] = 0.0
                    else:
                        vec[idx] = float(val)
                except Exception as e:
                    log.warning(f"Feature calculation failed for {name}", error=str(e))
                    vec[idx] = 0.0

            # ---------------------------------------------------------
            # 0-2: Price Action
            # ---------------------------------------------------------
            try:
                open_p = df['open'].iloc[-1]
                high_p = df['high'].iloc[-1]
                low_p = df['low'].iloc[-1]
                close_p = df['close'].iloc[-1]

                if open_p > 0:
                    price_pct = (close_p - open_p) / open_p
                    high_pct = (high_p - open_p) / open_p
                    low_pct = (low_p - open_p) / open_p
                else:
                    price_pct = high_pct = low_pct = 0.0

                safe_assign(0, self._clip_scale(price_pct, -0.05, 0.05, -1.0, 1.0), "price_pct_change")
                safe_assign(1, self._clip_scale(high_pct, 0.0, 0.05, 0.0, 1.0), "high_pct")
                safe_assign(2, self._clip_scale(low_pct, -0.05, 0.0, -1.0, 0.0), "low_pct")
            except Exception as e:
                log.warning("Failed to compute basic price action", error=str(e))

            # ---------------------------------------------------------
            # Technical Indicators (Dict returns)
            # ---------------------------------------------------------
            tech_features = {}
            try:
                tech_features = self.tech_calc.build_technical_feature_dict(df)
            except Exception as e:
                log.warning("Technical calculation engine failed", error=str(e))

            safe_assign(3, tech_features.get("volume_ratio", 0.0), "volume_norm")

            # 4: Spread
            try:
                if bids and asks and len(bids) > 0 and len(asks) > 0 and close_p > 0:
                    best_bid = float(bids[0][0])
                    best_ask = float(asks[0][0])
                    spread_pct = (best_ask - best_bid) / close_p
                    safe_assign(4, self._clip_scale(spread_pct, 0.0, 0.01, 0.0, 1.0), "spread_pct")
                else:
                    vec[4] = 0.0
            except Exception as e:
                log.warning("Failed to compute spread", error=str(e))

            # 5-10: MAs
            safe_assign(5, tech_features.get("ema_9_dist", 0.0), "ema_9_dist")
            safe_assign(6, tech_features.get("ema_21_dist", 0.0), "ema_21_dist")
            safe_assign(7, tech_features.get("ema_50_dist", 0.0), "ema_50_dist")
            safe_assign(8, tech_features.get("ema_200_dist", 0.0), "ema_200_dist")
            safe_assign(9, tech_features.get("golden_cross", 0.0), "golden_cross")
            safe_assign(10, tech_features.get("vwap_dist", 0.0), "vwap_dist")

            # 11-16: Momentum
            safe_assign(11, tech_features.get("rsi", 0.0), "rsi")
            safe_assign(12, tech_features.get("macd_norm", 0.0), "macd_norm")
            safe_assign(13, tech_features.get("macd_hist_norm", 0.0), "macd_hist_norm")
            safe_assign(14, tech_features.get("stoch_rsi", 0.0), "stoch_rsi")
            safe_assign(15, tech_features.get("adx_norm", 0.0), "adx_norm")
            safe_assign(16, tech_features.get("rsi_divergence", 0.0), "rsi_divergence")

            # 17-19: Volatility
            safe_assign(17, tech_features.get("atr_norm", 0.0), "atr_norm")
            safe_assign(18, tech_features.get("bb_width_norm", 0.0), "bb_width_norm")
            safe_assign(19, tech_features.get("bb_pct_b", 0.0), "bb_pct_b")

            # 20-21: Volume (Duplicated volume_ratio as per spec)
            safe_assign(20, tech_features.get("volume_ratio", 0.0), "volume_ratio_dup")
            safe_assign(21, tech_features.get("obv_slope", 0.0), "obv_slope")

            # 22-24: Fibonacci
            safe_assign(22, tech_features.get("fib_nearest_level_pct", 0.0), "fib_nearest_level_pct")
            safe_assign(23, tech_features.get("fib_distance", 0.0), "fib_distance")
            safe_assign(24, tech_features.get("fib_strength", 0.0), "fib_strength")

            # 25-34: Patterns
            patterns = tech_features.get("pattern_flags", [0.0]*10)
            for i in range(10):
                safe_assign(25 + i, patterns[i] if i < len(patterns) else 0.0, f"pattern_{i}")

            # ---------------------------------------------------------
            # Orderbook & Microstructure (35-42)
            # ---------------------------------------------------------
            ob_features = {}
            try:
                 # We recreate the klines slice as a list of dicts for detect_liquidity_sweep
                klines_list = df.reset_index().to_dict('records') 
                ob_features = self.ob_calc.build_orderbook_feature_dict(
                    bids, asks, trades, klines_list, sr_levels
                )
            except Exception as e:
                log.warning("Orderbook calculation engine failed", error=str(e))

            safe_assign(35, ob_features.get("book_imbalance", 0.0), "book_imbalance")
            safe_assign(36, ob_features.get("bid_depth_ratio", 0.0), "bid_depth_ratio")
            safe_assign(37, ob_features.get("ask_depth_ratio", 0.0), "ask_depth_ratio")
            safe_assign(38, ob_features.get("cvd_slope", 0.0), "cvd_slope")
            safe_assign(39, ob_features.get("cvd_divergence", 0.0), "cvd_divergence")
            safe_assign(40, ob_features.get("whale_activity", 0.0), "whale_activity")
            safe_assign(41, ob_features.get("bullish_sweep_strength", 0.0), "bullish_sweep")
            safe_assign(42, ob_features.get("bearish_sweep_strength", 0.0), "bearish_sweep")

            # ---------------------------------------------------------
            # Regime One-Hot & Confidence (43-48, 61)
            # ---------------------------------------------------------
            regimes_map = ["uptrend", "downtrend", "ranging", "high_volatility", "news_driven", "low_liquidity"]
            try:
                for i, r in enumerate(regimes_map):
                    vec[43 + i] = 1.0 if r == regime else 0.0
            except Exception as e:
                log.warning("Failed to encode regime", error=str(e))

            safe_assign(61, regime_confidence, "regime_confidence")

            # ---------------------------------------------------------
            # News Impact (49-52)
            # ---------------------------------------------------------
            try:
                if news_impact:
                    # Direction
                    nd = 0.0
                    if news_impact.direction.lower() == "up":
                        nd = 1.0
                    elif news_impact.direction.lower() == "down":
                        nd = -1.0
                    
                    vec[49] = nd
                    
                    # Magnitude
                    mag_avg = (news_impact.magnitude_pct_low + news_impact.magnitude_pct_high) / 2.0
                    vec[50] = self._clip_scale(mag_avg / 10.0, 0.0, 1.0, 0.0, 1.0)
                    
                    # Confidence
                    vec[51] = float(news_impact.confidence)
                    
                    # Age Norm
                    try:
                        # Convert ISO back to timestamp 
                        # Use simple replace 'Z' hack if present for Python 3.10- compatibility
                        iso_str = news_impact.created_at.replace("Z", "+00:00")
                        news_time = datetime.fromisoformat(iso_str)
                        mins_since = (datetime.utcnow().replace(tzinfo=news_time.tzinfo) - news_time).total_seconds() / 60.0
                        
                        max_mins = max(float(news_impact.t_max_minutes), 1.0) # Prevent div by 0
                        age_norm = 1.0 - min(mins_since / max_mins, 1.0)
                        vec[52] = max(age_norm, 0.0)
                    except Exception as age_e:
                        log.warning("Failed to parse news age", error=str(age_e))
                        vec[52] = 0.0

                else:
                    vec[49] = vec[50] = vec[51] = vec[52] = 0.0
            except Exception as e:
                log.warning("Failed to encode news impact", error=str(e))

            # ---------------------------------------------------------
            # Macro Cache (53-56)
            # ---------------------------------------------------------
            try:
                macro = await self.feature_cache.get_macro() or {}
            except Exception as e:
                log.warning("Failed to fetch macro cache", error=str(e))
                macro = {}

            safe_assign(53, macro.get("fear_greed_norm", 0.5), "fear_greed_norm")
            safe_assign(54, macro.get("btc_dominance_norm", 0.5), "btc_dominance_norm")
            safe_assign(55, macro.get("funding_rate_norm", 0.0), "funding_rate_norm")
            safe_assign(56, macro.get("oi_change_norm", 0.0), "oi_change_norm")

            # ---------------------------------------------------------
            # Time Encodings (57-60)
            # ---------------------------------------------------------
            try:
                now = datetime.utcnow()
                vec[57] = math.sin(2 * math.pi * now.hour / 24.0)
                vec[58] = math.cos(2 * math.pi * now.hour / 24.0)
                vec[59] = math.sin(2 * math.pi * now.weekday() / 7.0)
                vec[60] = math.cos(2 * math.pi * now.weekday() / 7.0)
            except Exception as e:
                log.warning("Failed to encode time", error=str(e))

        except Exception as e:
            log.error("Fatal error in feature builder", error=str(e))
            # vec is already completely zeroed out at init, which acts as safe default vector.

        # Verify exact shape requirements
        assert len(vec) == 62, f"Feature vector length {len(vec)} != 62"

        if log.isEnabledFor(logging.DEBUG):
            vector_hash = hashlib.sha256(vec.tobytes()).hexdigest()
            log.debug("Feature vector built", symbol=symbol, hash=vector_hash)

        return vec