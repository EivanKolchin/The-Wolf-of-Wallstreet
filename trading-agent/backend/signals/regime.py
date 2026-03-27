from datetime import datetime
from typing import Optional, Tuple, List
import pandas as pd
import pandas_ta as ta
import numpy as np
from structlog import get_logger

from memory.redis_client import NewsImpact

log = get_logger("signals.regime")

class RegimeDetector:
    
    def detect(self, df: pd.DataFrame, active_news: Optional[NewsImpact]) -> Tuple[str, float]:
        df_len = len(df)
        
        # We need enough data realistically, else fallback to ranging
        if df_len < 200:
            log.debug("Regime detected: fallback ranging (insufficient data)", regime="ranging", confidence=0.0)
            return "ranging", 0.0

        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']

        # 1. News driven
        if active_news is not None and getattr(active_news, "confidence", 0.0) > 0.65:
            conf = float(active_news.confidence)
            log.debug("Regime detected", regime="news_driven", confidence=conf)
            return "news_driven", conf
            
        try:
            current_close = close.iloc[-1]
            current_vol = volume.iloc[-1]
            
            # inline indicators
            atr = ta.atr(high, low, close, length=14)
            adx_res = ta.adx(high, low, close, length=14)
            ema_50 = ta.ema(close, length=50)
            ema_200 = ta.ema(close, length=200)
            
            recent_atr = atr.iloc[-200:]
            recent_close = close.iloc[-200:]
            recent_vol = volume.iloc[-200:]
            
            atr_pct = recent_atr / recent_close
            current_atr_pct = atr_pct.iloc[-1]
            
            # 2. High Volatility
            if current_atr_pct > atr_pct.quantile(0.95):
                log.debug("Regime detected", regime="high_volatility", confidence=0.85)
                return "high_volatility", 0.85
                
            # 3. Low Liquidity
            current_hour = datetime.utcnow().hour
            if current_vol < recent_vol.quantile(0.30) and current_hour in [0, 1, 2, 3, 4, 5]:
                log.debug("Regime detected", regime="low_liquidity", confidence=0.70)
                return "low_liquidity", 0.70
                
            # ADX values
            if adx_res is not None and not adx_res.empty:
                current_adx = float(adx_res.iloc[-1, 0])
            else:
                current_adx = 0.0
                
            val_ema_50 = float(ema_50.iloc[-1]) if ema_50 is not None else 0.0
            val_ema_200 = float(ema_200.iloc[-1]) if ema_200 is not None else 0.0

            # 4. Uptrend
            if current_adx > 25.0 and current_close > val_ema_200 and val_ema_50 > val_ema_200:
                conf = min(current_adx / 50.0, 1.0)
                log.debug("Regime detected", regime="uptrend", confidence=conf)
                return "uptrend", conf

            # 5. Downtrend
            if current_adx > 25.0 and current_close < val_ema_200 and val_ema_50 < val_ema_200:
                conf = min(current_adx / 50.0, 1.0)
                log.debug("Regime detected", regime="downtrend", confidence=conf)
                return "downtrend", conf

            # 6. Ranging (Default)
            conf = max(0.0, 1.0 - (current_adx / 20.0))
            log.debug("Regime detected", regime="ranging", confidence=conf)
            return "ranging", conf

        except Exception as e:
            log.warning("Regime detection failed, defaulting to ranging", error=str(e))
            return "ranging", 0.0

def get_regime_onehot(regime: str) -> List[float]:
    mapping = ["uptrend", "downtrend", "ranging", "high_volatility", "news_driven", "low_liquidity"]
    return [1.0 if r == regime else 0.0 for r in mapping]