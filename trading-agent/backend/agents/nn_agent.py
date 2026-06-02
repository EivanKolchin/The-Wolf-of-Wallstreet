import asyncio
import ctypes
import multiprocessing
import json
from collections import deque
from datetime import datetime, timedelta, timezone
import time
from dataclasses import dataclass
from dataclasses import asdict

import numpy as np
import structlog

from backend.data.market_feed import BinanceMarketFeed
from backend.signals.features import FeatureVectorBuilder
from backend.signals.regime import RegimeDetector
from backend.agents.nn_model import PersistentTradingModel, TradeExperience
from backend.memory.redis_client import PriorityNewsQueue, NewsImpact, HeartbeatClient
from backend.memory.database import Trade, TradeStatus, get_session
from sqlalchemy import select, func
from backend.core.config import settings
from backend.training.backbone import TrainingBackbone, map_asset_to_symbol
from backend.agents.improved_model import SYMBOL_TO_ID, HORIZONS
from backend.signals import feature_spec as fs
from backend.signals.news_embedding import NewsEmbedder, get_embedder
from backend.execution.position_monitor import PositionMonitor
from backend.agents.attention_controller import AttentionController, Attention
from backend.core import universe as _universe
from backend.signals.htf import HTFFeatureProvider

logger = structlog.get_logger(__name__)


def passes_uncertainty_gate(edge_mean: float, edge_std: float, gate: float) -> bool:
    """Workstream C: True if the MC-dropout edge dominates its own spread enough
    to justify acting. ``gate`` is the minimum |edge_mean|/(edge_std+eps) ratio;
    ``gate <= 0`` disables the gate (always True)."""
    if gate <= 0:
        return True
    return (abs(float(edge_mean)) / (abs(float(edge_std)) + 1e-6)) >= gate


@dataclass
class TradeDecision:
    symbol: str
    direction: str        # "long" / "short" / "hold"
    size_pct: float       # fraction of available capital [0, 0.20]
    nn_confidence: float  # max(probs.values())
    nn_probs: dict
    regime: str
    active_news: NewsImpact | None
    timestamp: datetime
    sl: float = 0.0
    tp: float = 0.0
    trail: float = 0.0
    edge_mean: float = 0.0
    edge_std: float = 0.0
    target_price: float = 0.0
    expected_execution_ts: float = 0.0
    rationale: dict | None = None

class NNTradingAgent:
    def __init__(
        self,
        market_feed: BinanceMarketFeed,
        feature_builder: FeatureVectorBuilder,
        regime_detector: RegimeDetector,
        model: PersistentTradingModel,
        risk_manager,
        execution_engine,
        news_queue: PriorityNewsQueue,
        severe_flag: multiprocessing.Value,
        symbols: list[str],
        cycle_interval_seconds: float = 5.0
    ):
        self.market_feed = market_feed
        self.feature_builder = feature_builder
        self.regime_detector = regime_detector
        self.model = model
        self.risk_manager = risk_manager
        self.execution_engine = execution_engine
        self.news_queue = news_queue
        self.severe_flag = severe_flag
        self.cycle_interval_seconds = cycle_interval_seconds
        self.symbols = symbols
        
        self.heartbeat_client = HeartbeatClient(news_queue.redis)

        # Phase 3: semantic news-text embedder feeds the NEWS_EMBED slots [70:86].
        # Redis-backed so repeated news isn't re-embedded each cycle.
        try:
            self.news_embedder: NewsEmbedder = get_embedder(news_queue.redis)
        except Exception as e:
            logger.warning("news_embedder_init_failed", error=str(e))
            self.news_embedder = get_embedder()

        # Cycle 7: earnings-calendar features feed the EARNINGS slots [86:90] for
        # stocks (Finnhub). Cached per symbol; zeros when no key/data — matching
        # exactly what scripts/pretrain.py writes offline.
        try:
            from backend.signals.earnings import EarningsProvider
            self.earnings_provider = EarningsProvider(getattr(settings, "FINNHUB_API_KEY", "") or "")
        except Exception as e:
            logger.warning("earnings_provider_init_failed", error=str(e))
            self.earnings_provider = None

        # Phase 15: real 1h/4h HTF features populate the 8 slots [62:70].
        # Provider runs background refresh tasks per (symbol, tf); get_features
        # is synchronous and returns the cached 8-vec (or zeros during warm-up).
        try:
            self.htf_provider = HTFFeatureProvider(news_queue.redis, symbols=self.symbols)
        except Exception as e:
            logger.warning("htf_provider_init_failed", error=str(e))
            self.htf_provider = None

        self.feature_sequences: dict[str, deque] = {
            sym: deque(maxlen=self.model.SEQUENCE_LENGTH) for sym in self.symbols
        }
        self.open_trades: dict[str, Trade] = {}
        
        self.current_news_impact: NewsImpact | None = None
        self.news_impact_expires_at: datetime | None = None
        # Phase 3 online self-labeling accumulator: latest price per symbol +
        # pending (news_embedding, price_then) records awaiting their forward
        # window so we can label them by realized forward return.
        self._last_price: dict[str, float] = {}
        self._pending_news_labels: deque = deque(maxlen=5000)
        # A2: cache the per-symbol MC predictive distribution computed during the
        # decision pass so the visualization loop can reuse it instead of running
        # its own (now-batched) MC forward. {symbol: {"horizons","edge_samples","ts"}}
        self._last_distribution: dict[str, dict] = {}
        # G7: US stocks get prediction bands on their charts too (viz-only — the
        # trading path stays crypto). Stock bars come from Alpaca; sequences are
        # rebuilt only when a new 5m bar arrives (cached) to bound the cost.
        self.stock_symbols: list[str] = list(_universe.STOCK_UNDERLYINGS)
        self._stock_df_cache: dict[str, tuple[float, object]] = {}
        self._stock_seq_cache: dict[str, tuple[str, np.ndarray]] = {}
        self.db_session_factory = None
        self.training_backbone = TrainingBackbone()
        self.open_trade_context: dict[str, np.ndarray] = {}
        self.open_trade_actions: dict[str, dict] = {}
        self.cycles_since_trade: int = 0
        self.attention_controller = AttentionController()
        self._attention: dict[str, str] = {}
        # Cycle 20: lock used by _process_symbol to serialise the risk-state
        # check + execution submit so two concurrently-evaluated symbols can't
        # both pass the same capacity guard before either records its trade.
        self._risk_lock: asyncio.Lock = asyncio.Lock()
        if hasattr(self.execution_engine, "set_trade_closed_callback"):
            try:
                self.execution_engine.set_trade_closed_callback(self._on_trade_closed)
            except Exception as e:
                logger.warning("failed_register_trade_closed_callback", error=str(e))

    def _symbol_id(self, symbol: str) -> int:
        return SYMBOL_TO_ID.get(str(symbol).upper(), 0)

    def _extend_with_htf(self, base_vec: np.ndarray, symbol: str | None = None) -> np.ndarray:
        """Extend a 62-feature BASE vector to the model's full INPUT-wide input.

        Phase 15: HTF slots [62:70] are populated from `HTFFeatureProvider`
        (real 1h+4h RSI/EMA-dist/MACD/ATR/trend), refreshed every ~30 min by a
        background task. Falls back to zeros during the brief warm-up before
        the first refresh completes, or if the provider failed to initialise.

        Phase 3: NEWS_EMBED slots [70:86] are populated from a semantic embedding
        of the symbol's currently-active news (or zeros when there's no fresh,
        relevant news). Mirrors exactly what scripts/pretrain.py writes offline,
        so offline-trained weights stay valid against live vectors.
        """
        base_vec = np.asarray(base_vec, dtype=np.float32)
        if base_vec.shape[0] == fs.INPUT:
            return base_vec
        out = np.zeros(fs.INPUT, dtype=np.float32)
        n = min(base_vec.shape[0], fs.BASE)
        out[:n] = base_vec[:n]
        if symbol is not None and self.htf_provider is not None:
            try:
                htf_vec = self.htf_provider.get_features(symbol)
                if htf_vec is not None and htf_vec.shape[0] == fs.HTF:
                    out[fs.HTF_START:fs.HTF_END] = htf_vec
            except Exception as e:
                logger.debug("htf_extend_failed", symbol=symbol, error=str(e))
        # NEWS_EMBED [70:86]
        if symbol is not None:
            try:
                news = self._get_news_for_symbol(symbol)
                if news is not None:
                    emb = self.news_embedder.embed_news_impact(news)
                    if emb is not None and emb.shape[0] == fs.NEWS_EMBED_DIM:
                        out[fs.NEWS_EMBED] = emb
            except Exception as e:
                logger.debug("news_embed_extend_failed", symbol=symbol, error=str(e))
        # EARNINGS [86:90] — stocks only (Finnhub calendar); zeros otherwise. Mirrors
        # scripts/pretrain.build_earnings_matrix so offline-trained weights stay valid.
        if symbol is not None and getattr(self, "earnings_provider", None) is not None \
                and _universe.asset_class_of(symbol) == "us_stock":
            try:
                import pandas as pd
                from backend.signals.earnings import earnings_features_at
                now = pd.Timestamp.utcnow().tz_localize(None)
                ev = self.earnings_provider.events(
                    symbol,
                    (now - pd.Timedelta(days=400)).strftime("%Y-%m-%d"),
                    (now + pd.Timedelta(days=120)).strftime("%Y-%m-%d"))
                if ev:
                    out[fs.EARNINGS] = earnings_features_at(ev, now)
            except Exception as e:
                logger.debug("earnings_extend_failed", symbol=symbol, error=str(e))
        return out

    def _get_news_for_symbol(self, symbol: str) -> NewsImpact | None:
        if not self.current_news_impact:
            return None

        impact = self.current_news_impact
        mapped_symbol = map_asset_to_symbol(getattr(impact, "asset", None))
        relevance = (impact.symbol_relevance or {}).get(symbol, 0.0) if hasattr(impact, "symbol_relevance") else 0.0
        if mapped_symbol and mapped_symbol != symbol and relevance <= 0:
            return None
        if relevance <= 0 and not mapped_symbol:
            return None

        # Clone and adjust confidence by symbol-level keyword relevance
        adjusted = NewsImpact(**asdict(impact))
        if relevance > 0:
            adjusted.confidence = float(max(0.0, min(1.0, adjusted.confidence * (0.5 + relevance))))
        return adjusted

    # ------------------------------------------------------------------ Phase 3
    # Online self-labeling news accumulator. When a news item is applied we snap
    # the news embedding + the price for each relevant symbol; once the forward
    # window elapses we realize the label (forward return) and append a training
    # sample to NN_NEWS_LABEL_LOG. This lets the news→price mapping keep learning
    # from whatever the LLM has actually seen, with no historical corpus needed.
    def _log_news_label(self, impact: NewsImpact) -> None:
        try:
            horizon_min = float(getattr(settings, "NN_NEWS_LABEL_HORIZON_MIN", 60))
            now = datetime.now(timezone.utc)
            due = now + timedelta(minutes=horizon_min)
            for sym in self.symbols:
                relevant = self._get_news_for_symbol(sym) if self.current_news_impact is impact else None
                # _get_news_for_symbol uses self.current_news_impact; impact has
                # just been assigned to it, so this returns the per-symbol view.
                if relevant is None:
                    continue
                price_then = self._last_price.get(sym)
                if not price_then or price_then <= 0:
                    continue
                emb = self.news_embedder.embed_news_impact(relevant)
                self._pending_news_labels.append({
                    "symbol": sym,
                    "embedding": [float(x) for x in emb.tolist()],
                    "price_then": float(price_then),
                    "logged_at": now.isoformat(),
                    "due_at": due.timestamp(),
                })
        except Exception as e:
            logger.debug("log_news_label_failed", error=str(e))

    def _drain_matured_news_labels(self) -> None:
        """Realize forward-return labels for pending records whose window has
        elapsed, appending them to the JSONL accumulator. Cheap, synchronous,
        and uses the cached last price (no extra market calls)."""
        if not self._pending_news_labels:
            return
        try:
            now_ts = datetime.now(timezone.utc).timestamp()
            log_path = str(getattr(settings, "NN_NEWS_LABEL_LOG", "training_data/news_labels.jsonl"))
            still_pending: deque = deque(maxlen=self._pending_news_labels.maxlen)
            matured: list[dict] = []
            for rec in self._pending_news_labels:
                if rec.get("due_at", 0) > now_ts:
                    still_pending.append(rec)
                    continue
                price_now = self._last_price.get(rec["symbol"])
                if not price_now or price_now <= 0:
                    continue  # drop — no price to label against
                fwd_ret = (price_now - rec["price_then"]) / rec["price_then"]
                matured.append({
                    "symbol": rec["symbol"],
                    "embedding": rec["embedding"],
                    "forward_return": float(fwd_ret),
                    "label": 0 if fwd_ret > 0.0 else (1 if fwd_ret < 0.0 else 2),  # long/short/flat
                    "logged_at": rec.get("logged_at"),
                    "horizon_min": float(getattr(settings, "NN_NEWS_LABEL_HORIZON_MIN", 60)),
                })
            self._pending_news_labels = still_pending
            if matured:
                import os as _os
                _os.makedirs(_os.path.dirname(log_path) or ".", exist_ok=True)
                with open(log_path, "a", encoding="utf-8") as fh:
                    for m in matured:
                        fh.write(json.dumps(m) + "\n")
                logger.info("news_labels_realized", count=len(matured))
        except Exception as e:
            logger.debug("drain_news_labels_failed", error=str(e))

    # ------------------------------------------------------------------ G7 stocks
    async def _acquire_viz_data(self, symbol: str):
        """Return (df, current_price, sequence) for the prediction viz, or None.
        Crypto uses the live rolling deque + market feed; stocks use Alpaca bars
        and a sequence rebuilt from the bar history (cached per 5m bar)."""
        if _universe.asset_class_of(symbol) == "us_stock":
            df = await self._get_stock_df_cached(symbol)
            if df is None or df.empty:
                return None
            sequence = await self._build_stock_sequence(symbol, df)
            if sequence is None:
                return None
            return df, float(df.iloc[-1]['close']), sequence
        # crypto
        if len(self.feature_sequences[symbol]) < self.model.SEQUENCE_LENGTH:
            return None
        df, recent_trades = await asyncio.gather(
            self.market_feed.get_dataframe(symbol),
            self.market_feed.get_recent_trades(symbol, n=1),
        )
        if df is None or df.empty:
            return None
        if recent_trades and 'price' in recent_trades[-1]:
            current_price = recent_trades[-1]['price']
        else:
            current_price = df.iloc[-1]['close']
        return df, current_price, np.stack(self.feature_sequences[symbol])

    async def _fetch_stock_dataframe(self, symbol: str):
        """Alpaca 5m bars → OHLCV DataFrame (datetime index). None on failure."""
        import datetime as _dt
        key = getattr(settings, "ALPACA_API_KEY", "") or ""
        secret = getattr(settings, "ALPACA_SECRET_KEY", "") or getattr(settings, "ALPACA_SECRET", "") or ""
        if not key or not secret or "your_" in key.lower():
            return None
        end = _dt.datetime.utcnow() - _dt.timedelta(minutes=16)   # IEX 15m delay slack
        start = end - _dt.timedelta(days=10)
        url = f"https://data.alpaca.markets/v2/stocks/{symbol.upper()}/bars"
        params = {
            "timeframe": "5Min", "start": start.replace(microsecond=0).isoformat() + "Z",
            "end": end.replace(microsecond=0).isoformat() + "Z",
            "limit": "10000", "adjustment": "raw", "feed": "iex", "sort": "asc",
        }
        headers = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}
        try:
            import aiohttp
            import pandas as pd
            rows = []
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as s:
                async with s.get(url, params=params, headers=headers) as r:
                    if r.status != 200:
                        return None
                    data = await r.json(content_type=None)
                    for b in (data.get("bars") or []):
                        rows.append({
                            "timestamp": pd.to_datetime(b["t"]),
                            "open": float(b["o"]), "high": float(b["h"]), "low": float(b["l"]),
                            "close": float(b["c"]), "volume": float(b["v"]),
                        })
            if not rows:
                return None
            return pd.DataFrame(rows).set_index("timestamp")
        except Exception as e:
            logger.debug("stock_df_fetch_failed", symbol=symbol, error=str(e))
            return None

    async def _get_stock_df_cached(self, symbol: str):
        """Cache stock bars ~60s to bound Alpaca calls; serve stale on failure."""
        now = time.time()
        cached = self._stock_df_cache.get(symbol)
        if cached and (now - cached[0]) < 60.0:
            return cached[1]
        df = await self._fetch_stock_dataframe(symbol)
        if df is not None:
            self._stock_df_cache[symbol] = (now, df)
            return df
        return cached[1] if cached else None

    async def _build_stock_sequence(self, symbol: str, df) -> np.ndarray | None:
        """Build a SEQUENCE_LENGTH feature sequence from the stock bar history.
        Rebuilt only when the latest bar changes (cached) since 5m bars are slow."""
        req = self.model.SEQUENCE_LENGTH
        if df is None or len(df) < req + 5:
            return None
        last_ts = str(df.index[-1])
        cached = self._stock_seq_cache.get(symbol)
        if cached and cached[0] == last_ts:
            return cached[1]
        news = self._get_news_for_symbol(symbol)
        n = len(df)
        seq = []
        for i in range(n - req, n):
            sub = df.iloc[: i + 1]
            try:
                regime_name, regime_conf = self.regime_detector.detect(sub, news)
                vec = await self.feature_builder.build(
                    symbol=symbol, df=sub, bids=[], asks=[], trades=[], sr_levels=[],
                    regime=regime_name, regime_confidence=regime_conf, news_impact=news,
                )
                seq.append(self._extend_with_htf(vec, symbol=symbol))
            except Exception as e:
                logger.debug("stock_seq_build_step_failed", symbol=symbol, error=str(e))
                return None
        arr = np.stack(seq)
        self._stock_seq_cache[symbol] = (last_ts, arr)
        return arr

    async def _background_predictions_loop(self) -> None:
        """Phase 18 v1 — honest MC-dropout uncertainty bands for the UI.

        Replaces the old deterministic ramp (`velocity * 0.85^i`). For each chart
        step we project price from K MC-dropout samples of the closest model
        horizon's `(p_long - p_short)` edge, scaled by realized step-vol with
        sqrt-time growth (Wiener-like). The median is the purple line; p25/p75
        becomes the shaded band (frontend reads `median_close/p25_close/p75_close`).
        """
        CHART_STEPS = 12
        while True:
            try:
                # Cycle 20: parallelise across symbols so charts don't serialise
                # into one slow tick. G7: include stocks so their charts also get
                # prediction bands (viz-only).
                await asyncio.gather(
                    *[self._render_prediction_chart(s, CHART_STEPS)
                      for s in (list(self.symbols) + self.stock_symbols)],
                    return_exceptions=True,
                )
            except Exception as e:
                logger.error("background_predictions_loop_error", error=str(e))

            await asyncio.sleep(self.cycle_interval_seconds)

    async def _render_prediction_chart(self, symbol: str, CHART_STEPS: int = 12) -> None:
        """Cycle 20: per-symbol body of the predictions loop, extracted so
        the outer loop can gather() them. Returns silently on warm-up / no-data
        so an exception inside one symbol can't poison the others."""
        try:
            # G7: route data acquisition by asset class (crypto deque vs Alpaca
            # stock bars). Returns (df, current_price, sequence) or None.
            acquired = await self._acquire_viz_data(symbol)
            if acquired is None:
                return
            df, current_price, sequence = acquired

            # Phase 3: cache the latest price per symbol so the online news-label
            # accumulator can compute forward returns without extra market calls.
            try:
                self._last_price[symbol] = float(current_price)
            except Exception:
                pass

            # Single point estimate for the thought text + size/decision context
            _res = self.model.infer(sequence, symbol_id=self._symbol_id(symbol))
            decision_str, size_pct, probs = _res.direction, _res.size, _res.probs

            # Phase 18 v1: MC-dropout per-horizon edge distribution.
            # A2: reuse the distribution the decision pass already computed this
            # cycle (if fresh) instead of running another MC forward; only fall
            # back to a fresh MC pass when the cache is stale/absent.
            K = int(getattr(settings, "NN_MC_SAMPLES", 16))
            cached = self._last_distribution.get(symbol)
            if cached and (time.time() - cached.get("ts", 0.0)) < self.cycle_interval_seconds * 2.0:
                horizons = cached["horizons"]
                edge_samples = cached["edge_samples"]
            else:
                dist = self.model.infer_predictive_distribution(
                    sequence, symbol_id=self._symbol_id(symbol), mc_samples=K
                )
                horizons = dist["horizons"]             # e.g. [3, 12, 48]
                edge_samples = dist["edge_samples"]     # (K, H)

            # Realized per-bar volatility — magnitude proxy that adapts to regime.
            try:
                rets = df['close'].pct_change().tail(60).dropna()
                realized_step_vol = float(rets.std())
                if not np.isfinite(realized_step_vol) or realized_step_vol < 1e-5:
                    realized_step_vol = 0.005
            except Exception:
                realized_step_vol = 0.005

            predictions = []
            prev_close = float(current_price)
            for i in range(1, CHART_STEPS + 1):
                closest_h_idx = int(np.argmin([abs(h - i) for h in horizons]))
                edges_at_step = edge_samples[:, closest_h_idx]
                mag_i = realized_step_vol * float(np.sqrt(i))
                sample_prices = current_price * (1.0 + edges_at_step * mag_i)
                median_close = float(np.median(sample_prices))
                p25_close = float(np.percentile(sample_prices, 25))
                p75_close = float(np.percentile(sample_prices, 75))
                predictions.append({
                    "step": i, "open": prev_close, "close": median_close,
                    "high": p75_close, "low": p25_close,
                    "median_close": median_close,
                    "p25_close": p25_close, "p75_close": p75_close,
                })
                prev_close = median_close

            target_price = predictions[-1]['close']
            confidence_pct = max(probs.values()) * 100
            time_range = "10 to 15 minutes"
            target_buy_price = None
            target_sell_price = None
            if target_price > current_price:
                target_sell_price = target_price
            else:
                target_buy_price = target_price

            if decision_str == "hold":
                if target_price > current_price:
                    price_diff = ((target_price - current_price) / current_price) * 100
                    if price_diff > 0.05:
                        thought = f"Detected significant upside potential (+{price_diff:.2f}%). Price is currently low (~${current_price:,.2f}). Considering entering LONG position to target ~${target_price:,.2f} within {time_range}."
                    else:
                        thought = f"Projecting mild upside to ~${target_price:,.2f} within {time_range}, but current price (~${current_price:,.2f}) isn't low enough for a strong entry. Waiting for better risk/reward."
                else:
                    price_diff = ((current_price - target_price) / current_price) * 100
                    thought = f"Projecting a dip of ~{price_diff:.2f}% down to ~${target_price:,.2f} over the next {time_range}. Holding off buys until price drops to that support level."
            elif decision_str == "long":
                thought = f"Optimal buying conditions met! Current price (~${current_price:,.2f}) is favorable. Executing LONG order to capture projected run-up to ~${target_price:,.2f} ({confidence_pct:.1f}% confidence)."
            elif decision_str == "short":
                thought = f"Price is overextended at ~${current_price:,.2f}. Executing SHORT order to capture projected drop to ~${target_price:,.2f} ({confidence_pct:.1f}% confidence)."
            else:
                thought = "Analyzing momentum across recent history and news impact..."

            payload = {
                "symbol": symbol,
                "predictions": predictions,
                "current_price": current_price,
                "target_buy_price": target_buy_price,
                "target_sell_price": target_sell_price,
                "thought": thought,
            }
            await self.heartbeat_client.redis.set(f"agent_visual_predictions:{symbol}", json.dumps(payload))
            await self.heartbeat_client.redis.set("agent_visual_predictions", json.dumps(payload))
        except Exception as e:
            logger.error("render_prediction_chart_error", symbol=symbol, error=str(e))

    async def _read_portfolio_state(self) -> dict:
        """Cycle 20: hoisted out of the per-symbol loop. All symbols share the
        same cash pool, so reading it once per cycle eliminates N redundant DB
        round-trips and avoids parallel-write contention."""
        initial_usdc = settings.INITIAL_USDC_AMOUNT
        try:
            async with get_session() as session:
                result = await session.execute(
                    select(func.sum(Trade.pnl_usd)).where(Trade.status == TradeStatus.closed)
                )
                realized_pnl = result.scalar() or 0.0
                result_open = await session.execute(
                    select(func.sum(Trade.size_usd)).where(Trade.status == TradeStatus.open)
                )
                locked_cash = result_open.scalar() or 0.0
        except Exception as e:
            logger.warning("portfolio_state_read_failed", error=str(e))
            realized_pnl, locked_cash = 0.0, 0.0
        available_cash = max(0.0, initial_usdc + realized_pnl - locked_cash)
        return {
            "available_cash": available_cash,
            "available_usdc": available_cash,
            # Private breadcrumbs the run-loop uses to update RiskManager once.
            "_initial_usdc": initial_usdc,
            "_realized_pnl": realized_pnl,
        }

    async def _build_symbol_features(self, symbol: str, portfolio_state: dict):
        """A3 build phase: fetch market data, update the rolling feature buffer,
        and return the context needed for a decision — or None if the symbol is
        warming up / has no data. Pure feature work; no inference, no execution,
        so all symbols' builds can be gathered, then inferred in ONE batch."""
        # Layer 1 parallelism: 3 independent network reads in one gather.
        try:
            df, orderbook, trades = await asyncio.gather(
                self.market_feed.get_dataframe(symbol),
                self.market_feed.get_orderbook(symbol),
                self.market_feed.get_recent_trades(symbol, n=100),
            )
        except AttributeError as e:
            logger.error("market_feed_missing_attribute", symbol=symbol, error=str(e))
            return None
        except Exception as e:
            logger.error("market_feed_fetch_failed", symbol=symbol, error=str(e))
            return None
        if df is None or df.empty:
            return None

        # Variable attention: set this symbol's compute cadence (Phase 11)
        try:
            closes = df['close'].tail(40).astype(float).tolist()
            vol_tail = df['volume'].tail(20).astype(float)
            vr = float(df['volume'].iloc[-1] / max(float(vol_tail.mean()), 1e-9))
            att = self.attention_controller.evaluate(
                symbol, _universe.asset_class_of(symbol), closes, volume_ratio=vr
            )
            self._attention[symbol] = att.value
        except Exception:
            self._attention.setdefault(symbol, "low")

        bids = orderbook.get("bids", []) if orderbook else []
        asks = orderbook.get("asks", []) if orderbook else []
        sr_levels: list = []

        symbol_news_impact = self._get_news_for_symbol(symbol)
        regime_name, regime_conf = self.regime_detector.detect(df, symbol_news_impact)

        vector = await self.feature_builder.build(
            symbol=symbol, df=df, bids=bids, asks=asks, trades=trades,
            sr_levels=sr_levels, regime=regime_name,
            regime_confidence=regime_conf, news_impact=symbol_news_impact,
        )
        if vector.shape[0] != fs.BASE:
            logger.error("feature_vector_invalid_shape", symbol=symbol, shape=vector.shape)
            return None
        self.feature_sequences[symbol].append(self._extend_with_htf(vector, symbol=symbol))

        if len(self.feature_sequences[symbol]) < self.model.SEQUENCE_LENGTH:
            return None
        if time.time() - self.started_at < 300:
            logger.debug("agent_warming_up", symbol=symbol, remaining_seconds=300 - (time.time() - self.started_at))
            return None

        return {
            "sequence": np.stack(self.feature_sequences[symbol]),
            "df": df,
            "regime_name": regime_name,
            "news": symbol_news_impact,
        }

    async def _process_symbol(self, symbol: str, portfolio_state: dict) -> None:
        """Single-symbol path (fallback / tests): build → infer → decide. The main
        cycle uses the batched path (_build_symbol_features + infer_batch +
        _decide_and_execute) instead."""
        ctx = await self._build_symbol_features(symbol, portfolio_state)
        if ctx is None:
            return
        _res, _dist = self.model.infer_with_distribution(
            ctx["sequence"], symbol_id=self._symbol_id(symbol),
            mc_samples=int(settings.NN_MC_SAMPLES),
        )
        await self._decide_and_execute(symbol, ctx, _res, _dist, portfolio_state)

    async def _decide_and_execute(self, symbol: str, ctx: dict, _res, _dist, portfolio_state: dict) -> None:
        """A3 decide phase: apply the gates, sizing, risk approval and execution
        using a pre-computed inference result (so the model forward can be shared
        across symbols in one batch). ``ctx`` is from _build_symbol_features."""
        sequence = ctx["sequence"]
        df = ctx["df"]
        regime_name = ctx["regime_name"]
        symbol_news_impact = ctx["news"]
        self._last_distribution[symbol] = {**_dist, "ts": time.time()}
        decision_str, size_pct, probs = _res.direction, _res.size, _res.probs
        nn_confidence = max(probs.values())
        if decision_str == "long" and probs.get("long", 0.0) < float(settings.NN_LONG_CONFIDENCE_THRESHOLD):
            decision_str = "hold"
        if decision_str == "short" and probs.get("short", 0.0) < float(settings.NN_SHORT_CONFIDENCE_THRESHOLD):
            decision_str = "hold"
        min_edge = float(settings.NN_MIN_EDGE_OVER_FEE) * (2.0 * 0.001)
        if decision_str in ("long", "short") and abs(_res.edge_mean) < min_edge:
            decision_str = "hold"
        # Workstream C: uncertainty gate. Only act when the MC-dropout edge
        # dominates its own spread (|edge_mean| / edge_std). Skips low-conviction
        # signals where the network disagrees with itself across dropout samples,
        # improving profit-per-loss without overfitting. Set 0 to disable.
        _gate = float(getattr(settings, "NN_MIN_EDGE_TO_UNCERTAINTY", 1.0))
        if decision_str in ("long", "short") and not passes_uncertainty_gate(
            _res.edge_mean, _res.edge_std, _gate
        ):
            logger.debug("uncertainty_gate_hold", symbol=symbol,
                         edge_mean=round(_res.edge_mean, 5),
                         edge_std=round(_res.edge_std, 5), gate=_gate)
            decision_str = "hold"

        # Phase 9: statistical R:R floors + target_price + ETA
        try:
            recent_returns = df['close'].pct_change().tail(20).dropna()
            recent_vol = float(recent_returns.std()) if len(recent_returns) else 0.0
        except Exception:
            recent_vol = 0.0
        sl_eff, tp_eff = (_res.sl, _res.tp)
        if hasattr(self.risk_manager, "enforce_exit_floors"):
            sl_eff, tp_eff = self.risk_manager.enforce_exit_floors(_res.sl, _res.tp, recent_vol)
        try:
            last_close = float(df['close'].iloc[-1])
        except Exception:
            last_close = 0.0
        horizon_min = (HORIZONS[0] if HORIZONS else 3) * 5
        if decision_str == "long":
            target_price = last_close * (1.0 + tp_eff)
        elif decision_str == "short":
            target_price = last_close * (1.0 - tp_eff)
        else:
            target_price = last_close
        expected_execution_ts = time.time() + horizon_min * 60.0

        decision = TradeDecision(
            symbol=symbol, direction=decision_str, size_pct=size_pct,
            nn_confidence=nn_confidence, nn_probs=probs, regime=regime_name,
            active_news=symbol_news_impact, timestamp=datetime.utcnow(),
            sl=sl_eff, tp=tp_eff, trail=_res.trail,
            edge_mean=_res.edge_mean, edge_std=_res.edge_std,
            target_price=target_price, expected_execution_ts=expected_execution_ts,
        )

        # Phase 13: structured rationale
        try:
            from backend.agents.xai import build_rationale as _xai_build
            decision.rationale = _xai_build(decision, extras={
                "recent_vol": recent_vol, "last_close": last_close,
            })
        except Exception:
            pass

        # Kelly sizing — Cycle 19.6 ATR-scaled
        _atr_pct = 0.0
        try:
            _atr_norm = float(sequence[-1, fs.VOLATILITY.start])
            _atr_pct = max(0.0, _atr_norm / 10.0)
        except Exception:
            pass
        _ksize = self.risk_manager.kelly_size(
            _res.edge_mean, _res.edge_std, atr_pct=_atr_pct
        ) if hasattr(self.risk_manager, "kelly_size") else None
        if _ksize is not None:
            decision.size_pct = _ksize
            size_pct = _ksize

        # Duplicate-direction guard
        if symbol in self.open_trades and hasattr(self.open_trades[symbol], 'direction'):
            if self.open_trades[symbol].direction.value == decision_str:
                decision_str = "hold"
                decision.direction = "hold"

        # Cycle 20: serialise risk approval + execute submit so two concurrent
        # symbols can't both pass the same capacity check and race to open.
        async with self._risk_lock:
            approved, reason = self.risk_manager.approve(decision, portfolio_state)
            self.training_backbone.record_decision(
                symbol=symbol, sequence=sequence, decision=decision_str,
                size_pct=size_pct, probs=probs, regime=regime_name,
                approved=approved, reason=reason, news_impact=symbol_news_impact,
            )
            if approved:
                trade = await self.execution_engine.execute(decision, portfolio_state)
                if trade:
                    self.open_trades[symbol] = trade
                    self.open_trade_context[symbol] = sequence.copy()
                    self.open_trade_actions[symbol] = {"size": size_pct, "sl": _res.sl, "tp": _res.tp}
                    self.cycles_since_trade = 0
                    logger.info("trade_executed", symbol=symbol, direction=decision_str, size=size_pct)
            else:
                if decision_str != "hold":
                    logger.info("trade_rejected", symbol=symbol, reason=reason, direction=decision_str)
                else:
                    logger.info("trade_hold_decision", symbol=symbol, probs=probs,
                                  seq_mean=float(sequence.mean()), seq_max=float(sequence.max()))

    async def _rebuild_open_trades(self) -> int:
        """Cycle 19.2: rebuild in-memory ``open_trades`` from the DB on startup.

        Without this, a backend restart with a paper trade in flight silently
        bypasses SL/TP/trailing (the PositionMonitor only sees what's in this
        dict) and can open a duplicate position because the ``symbol in
        self.open_trades`` guard returns False. Reads every ``Trade`` row whose
        status is still ``open`` and re-attaches it both to ``self.open_trades``
        and to the broker's in-memory tracking dict where present.
        """
        try:
            async with get_session() as session:
                rows = (await session.execute(
                    select(Trade).where(Trade.status == TradeStatus.open)
                )).scalars().all()
            for t in rows:
                sym = getattr(t, "asset", None) or getattr(t, "symbol", None)
                if not sym:
                    continue
                self.open_trades[sym] = t
                # Brokers that keep their own dict (DefiBroker, AlpacaBroker)
                # also need re-population so close_position can find the trade.
                broker_dict = getattr(self.execution_engine, "_open", None)
                if isinstance(broker_dict, dict):
                    broker_dict[sym] = t
            logger.info("open_positions_recovered", count=len(rows))
            return len(rows)
        except Exception as e:
            logger.warning("open_positions_recovery_failed", error=str(e))
            return 0

    async def run(self) -> None:
        logger.info("nn_trading_agent_started", symbols=self.symbols)
        self.started_at = time.time()

        # Cycle 19.2: rehydrate any still-open trades from the DB BEFORE the
        # position monitor starts, otherwise it briefly polls an empty dict.
        await self._rebuild_open_trades()

        # Cycle 23: re-baseline the drawdown breaker to the ACTUAL starting book.
        # peak_portfolio_value defaults to RiskManager's init value; if that ever
        # drifts from the real cash (or a prior session left realized losses in
        # the DB), the very first approve() reads a huge phantom drawdown and
        # latches the agent into HALTED forever. Seeding both value and peak from
        # the live portfolio measures drawdown from session start, not a guess.
        try:
            _ps0 = await self._read_portfolio_state()
            _v0 = float(_ps0["_initial_usdc"] + _ps0["_realized_pnl"])
            if _v0 > 0 and hasattr(self.risk_manager, "portfolio_value_usd"):
                self.risk_manager.portfolio_value_usd = _v0
                self.risk_manager.peak_portfolio_value = _v0
                self.risk_manager.is_halted = False
                logger.info("risk_baseline_seeded", portfolio_value_usd=_v0)
        except Exception as e:
            logger.warning("risk_baseline_seed_failed", error=str(e))

        # Pre-fill historical sequences
        for symbol in self.symbols:
            logger.info("prefilling_historical_features", symbol=symbol)
            try:
                df = await self.market_feed.get_dataframe(symbol)
                if df is not None and not df.empty:
                    req = self.model.SEQUENCE_LENGTH
                    available = len(df); logger.info("df_length_check", length=available)
                    if available > req:
                        start_idx = max(0, available - req)
                        for i in range(start_idx, available):
                            sub_df = df.iloc[:i+1] 
                            regime_name, regime_conf = self.regime_detector.detect(sub_df, None)
                            vector = await self.feature_builder.build(
                                symbol=symbol,
                                df=sub_df,
                                bids=[],
                                asks=[],
                                trades=[],
                                sr_levels=[],
                                regime=regime_name,
                                regime_confidence=regime_conf,
                                news_impact=None
                            )
                            self.feature_sequences[symbol].append(self._extend_with_htf(vector, symbol=symbol))
                        logger.info("buffer_filled_successfully", buf_len=len(self.feature_sequences[symbol]))
            except Exception as e:
                logger.error("failed_to_prefill_buffer", symbol=symbol, error=str(e))
                
        # Start background predictions loop
        asyncio.create_task(self._background_predictions_loop())

        # Phase 15: kick off per-symbol HTF refresh tasks (1h+4h klines → Redis cache).
        if self.htf_provider is not None:
            try:
                self.htf_provider.start()
            except Exception as e:
                logger.warning("htf_provider_start_failed", error=str(e))

        # Start the position monitor (applies model-emitted SL/TP/trailing + closes,
        # plus Cycle 19.3 talib momentum-reversal exit using the market feed).
        if hasattr(self.execution_engine, "get_price"):
            self.position_monitor = PositionMonitor(
                self.open_trades, self.execution_engine, poll_interval=3.0,
                market_feed=self.market_feed,
                reversal_macd_drop=float(getattr(settings, "NN_REVERSAL_MACD_DROP", 0.5)),
                reversal_vol_multiple=float(getattr(settings, "NN_REVERSAL_VOL_MULTIPLE", 2.0)),
            )
            asyncio.create_task(self.position_monitor.run())

        while True:
            try:
                # Expose state to Redis for the frontend timer and check for manual halt
                try:
                    longest_seq = max([len(seq) for seq in self.feature_sequences.values()]) if self.feature_sequences else 0
                    is_forced_stop = await self.heartbeat_client.redis.get("agent_force_stopped")
                    
                    status_payload = {
                        "is_halted": is_forced_stop == b"true",
                        "buffer_current": longest_seq,
                        "buffer_required": self.model.SEQUENCE_LENGTH,
                        "cycle_interval": self.cycle_interval_seconds,
                        "started_at": self.started_at,
                        "has_market_data": False
                    }
                    
                    # Update has_market_data if any symbol has dataframe
                    for sym in self.symbols:
                        try:
                            df = await self.market_feed.get_dataframe(sym)
                            if df is not None and not df.empty:
                                status_payload["has_market_data"] = True
                        except:
                            pass

                    await self.heartbeat_client.redis.set("agent_frontend_status", json.dumps(status_payload))
                    risk_status = self.risk_manager.get_status() if hasattr(self.risk_manager, "get_status") else {}
                    risk_status["updated_at_ts"] = time.time()
                    risk_status["source"] = "nn_agent_process"
                    await self.heartbeat_client.redis.set("risk:status", json.dumps(risk_status))
                    reset_requested = await self.heartbeat_client.redis.get("risk:reset_requested")
                    # redis-py returns bytes here (cf. is_forced_stop == b"true" above),
                    # so the old `== "true"` str compare never matched and the manual
                    # reset valve was dead. Accept both forms.
                    if reset_requested in (b"true", "true") and hasattr(self.risk_manager, "reset_halt"):
                        self.risk_manager.reset_halt()
                        await self.heartbeat_client.redis.delete("risk:reset_requested")
                        logger.info("risk_halt_reset_via_redis_flag")
                    
                    if is_forced_stop == b"true":
                        await asyncio.sleep(self.cycle_interval_seconds)
                        continue
                except Exception as e:
                    logger.error("redis_status_sync_error", error=str(e))

                # 1. Check severe flag
                if self.severe_flag.value:
                    await self._emergency_protocol()
                    await asyncio.sleep(60)
                    continue

                # 2. Drain news queue
                while True:
                    impact = await self.news_queue.get_nowait()
                    if not impact:
                        break
                    
                    if impact.severity == "SEVERE":
                        self.severe_flag.value = True
                        logger.error("severe_news_received_triggering_emergency", impact=impact.to_json() if hasattr(impact, 'to_json') else str(impact))
                        await self._emergency_protocol()
                        break
                    elif impact.severity == "SIGNIFICANT":
                        self.current_news_impact = impact
                        self.news_impact_expires_at = datetime.utcnow() + timedelta(minutes=impact.t_max_minutes)
                        # Phase 3: snapshot (embedding, price) per relevant symbol
                        # for the online self-labeling accumulator.
                        self._log_news_label(impact)

                if self.severe_flag.value:
                    continue

                # Phase 3: realize any matured news labels (forward-return).
                self._drain_matured_news_labels()

                # 3. Expire news impact
                if self.current_news_impact and self.news_impact_expires_at:
                    if datetime.utcnow() > self.news_impact_expires_at:
                        logger.info("news_impact_expired")
                        self.current_news_impact = None
                        self.news_impact_expires_at = None

                # Read live attention overrides from Redis (Phase 12 UI controls)
                try:
                    raw_overrides = await self.heartbeat_client.redis.get("attention:overrides")
                    if raw_overrides:
                        ov = raw_overrides if isinstance(raw_overrides, str) else raw_overrides.decode()
                        self.attention_controller.replace_overrides(json.loads(ov))
                    else:
                        self.attention_controller.replace_overrides({})
                except Exception:
                    pass

                # 4. Anti-inaction: ramp idle pressure the longer we go without trading
                idle_pressure = min(self.cycles_since_trade / max(int(settings.NN_IDLE_PATIENCE), 1), 1.0)
                if hasattr(self.model, "set_idle_pressure"):
                    self.model.set_idle_pressure(idle_pressure)

                # 5. Process all symbols in parallel (Cycle 20).
                # Portfolio state is computed once per cycle (it's identical
                # across symbols — they all share the same cash pool).
                portfolio_state = await self._read_portfolio_state()
                try:
                    self.risk_manager.update_portfolio({
                        "total_value_usd": float(portfolio_state["_initial_usdc"] + portfolio_state["_realized_pnl"])
                    })
                except Exception:
                    pass

                # A3: build all symbols' features in parallel, run ONE batched
                # inference for the ready ones, then decide/execute per symbol.
                # Collapses N model forwards into 1 (×2 with the MC pass).
                ctxs = await asyncio.gather(
                    *[self._build_symbol_features(s, portfolio_state) for s in self.symbols],
                    return_exceptions=True,
                )
                ready = [
                    (s, c) for s, c in zip(self.symbols, ctxs)
                    if isinstance(c, dict) and c.get("sequence") is not None
                ]
                if ready:
                    try:
                        batch = self.model.infer_batch(
                            [c["sequence"] for _, c in ready],
                            [self._symbol_id(s) for s, _ in ready],
                            mc_samples=int(settings.NN_MC_SAMPLES),
                        )
                        await asyncio.gather(
                            *[self._decide_and_execute(s, c, batch[i][0], batch[i][1], portfolio_state)
                              for i, (s, c) in enumerate(ready)],
                            return_exceptions=True,
                        )
                    except Exception as e:
                        logger.error("batched_inference_failed_fallback", error=str(e))
                        # Defensive fallback to the per-symbol path so a batching
                        # bug can never halt trading.
                        await asyncio.gather(
                            *[self._process_symbol(s, portfolio_state) for s in self.symbols],
                            return_exceptions=True,
                        )

                await self.heartbeat_client.ping("nn_trading_agent")
                self.cycles_since_trade += 1

            except Exception as e:
                logger.error("nn_agent_loop_error", error=str(e))

            # Variable Attention Engine (Phase 11): next cadence = the most urgent asset (high -> 1s)
            next_interval = self.cycle_interval_seconds
            try:
                intervals = [self.attention_controller.interval_for(Attention(a)) for a in self._attention.values()]
                if intervals:
                    next_interval = min(intervals)
                await self.heartbeat_client.redis.set("attention:state", json.dumps(self._attention))
            except Exception:
                pass
            await asyncio.sleep(next_interval)

    async def _emergency_protocol(self) -> None:
        logger.error("emergency_protocol_activated", open_trades=len(self.open_trades))

        for symbol, trade in list(self.open_trades.items()):
            logger.info("attempting_emergency_close", symbol=symbol, trade_id=str(getattr(trade, "id", "")))
            try:
                if hasattr(self.execution_engine, "close_position"):
                    await self.execution_engine.close_position(symbol, reason="emergency")
                else:
                    self.open_trades.pop(symbol, None)
            except Exception as e:
                logger.error("emergency_close_failed", symbol=symbol, error=str(e))

        # close_position pops via the trade-closed callback; clear any stragglers.
        self.open_trades.clear()
        
    async def _on_trade_closed(self, trade: Trade, pnl_pct: float) -> None:
        logger.info("trade_closed", trade_id=str(trade.id), pnl_pct=pnl_pct)
        trade_symbol = getattr(trade, "symbol", None) or getattr(trade, "asset", None)
        if not trade_symbol:
            logger.warning("trade_closed_missing_symbol", trade_id=str(getattr(trade, "id", "")))
            return
        seq = self.open_trade_context.get(trade_symbol)
        if seq is None:
            seq = np.zeros((self.model.SEQUENCE_LENGTH, fs.INPUT), dtype=np.float32)
        
        direction_map = {"long": 0, "short": 1, "hold": 2}
        direction_value = "hold"
        if getattr(trade, "direction", None) is not None:
            direction_value = trade.direction.value if hasattr(trade.direction, "value") else str(trade.direction)
        dir_taken = direction_map.get(direction_value, 2)

        action = self.open_trade_actions.get(trade_symbol, {})
        bars_held = 0.0
        try:
            opened = getattr(trade, "opened_at", None)
            closed = getattr(trade, "closed_at", None)
            if opened and closed:
                bars_held = max(0.0, (closed - opened).total_seconds() / 300.0)
        except Exception:
            pass

        experience = TradeExperience(
            features_sequence=seq,
            direction_taken=dir_taken,
            actual_pnl_pct=pnl_pct,
            symbol_id=self._symbol_id(trade_symbol),
            size_taken=float(action.get("size", 0.1)),
            sl_taken=float(action.get("sl", 0.0)),
            tp_taken=float(action.get("tp", 0.0)),
            bars_held=bars_held,
        )

        self.model.online_update(experience)
        self.training_backbone.record_outcome(symbol=trade_symbol, pnl_pct=pnl_pct, trade_id=str(getattr(trade, "id", "")))
        if hasattr(self.risk_manager, "record_return"):
            self.risk_manager.record_return(pnl_pct)
        
        if self.model.check_and_rollback(recent_pnl_pct=pnl_pct):
            logger.warning("rollback_triggered_by_recent_pnl")
        
        if trade_symbol in self.open_trades:
            del self.open_trades[trade_symbol]
        if trade_symbol in self.open_trade_context:
            del self.open_trade_context[trade_symbol]
        if trade_symbol in self.open_trade_actions:
            del self.open_trade_actions[trade_symbol]