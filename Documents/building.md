# Build Prompts — AI Trading Agent
### Prompts for a coding agent (vibe-safe sections only)
### Read the manual review summary at the end before starting any prompt

> **Before running any prompt:** set up your repo structure first.
> Every prompt assumes the following exists:
> ```
> /trading-agent
>   /backend
>   /frontend
>   /models
>   /scripts
>   /tests
>   docker-compose.yml
>   .env.example
> ```
> Never commit `.env`. Never hardcode API keys. Never hardcode private keys.

---

## PROMPT 1 — Project scaffold & environment

```
Create the full project scaffold for a Python trading agent with a Next.js frontend.

Backend structure (Python 3.11):
/backend
  /core
    __init__.py
    config.py          # loads from .env using pydantic-settings
    logger.py          # structlog JSON logger setup
  /data
    __init__.py
    market_feed.py     # placeholder
    news_feed.py       # placeholder
  /signals
    __init__.py
    technical.py       # placeholder
    orderbook.py       # placeholder
    regime.py          # placeholder
    features.py        # placeholder - builds the feature vector
  /agents
    __init__.py
    nn_agent.py        # placeholder
    news_agent.py      # placeholder
  /risk
    __init__.py
    manager.py         # placeholder
  /execution
    __init__.py
    engine.py          # placeholder
    kite_chain.py      # placeholder
  /memory
    __init__.py
    database.py        # placeholder
    redis_client.py    # placeholder
  main.py              # process launcher

Frontend structure (Next.js 14 + TypeScript + Tailwind + shadcn/ui):
/frontend
  /app
    /dashboard         # main trading dashboard page
    /positions         # open positions page
    /signals           # live signal feed page
    /audit             # on-chain trade log page
    layout.tsx
    page.tsx
  /components
    /ui                # shadcn components
    /charts            # TradingView lightweight charts
    /widgets           # PnL card, confidence gauge, signal feed, news ticker
  /lib
    api.ts             # fetch wrapper for backend API
    web3.ts            # wagmi/viem Kite chain connection
    types.ts           # shared TypeScript types

Root files:
  docker-compose.yml   # postgres, redis, backend, frontend services
  .env.example         # all required env vars listed with descriptions, no values
  requirements.txt     # all Python dependencies
  README.md            # setup instructions

Fill in config.py with all environment variables needed:
- DATABASE_URL, REDIS_URL
- ANTHROPIC_API_KEY
- BINANCE_API_KEY, BINANCE_SECRET
- ALPACA_API_KEY, ALPACA_SECRET
- KITE_CHAIN_RPC_URL, KITE_CHAIN_PRIVATE_KEY, KITE_AGENT_ADDRESS
- X_API_KEY (Twitter)
- TELEGRAM_API_ID, TELEGRAM_API_HASH

Fill in logger.py with structlog configured for JSON output with timestamps, 
log level from env, and a get_logger(name) helper function.

Fill in docker-compose.yml with:
- postgres:16 service with volume
- redis:7 service
- backend service (Python, mounts /backend, depends on postgres + redis)
- frontend service (Node 20, mounts /frontend, port 3000)
All services on a shared network. Backend on port 8000.
```

---

## PROMPT 2 — PostgreSQL schema & migrations

```
Create the full PostgreSQL database schema for the trading agent using SQLAlchemy 2.0 
async ORM. File: /backend/memory/database.py

Tables required:

1. source_trust_scores
   - id (UUID, primary key)
   - source_domain (str, unique, indexed)
   - base_score (float, default from config)
   - current_score (float)
   - total_predictions (int, default 0)
   - correct_predictions (int, default 0)
   - last_updated (timestamp)

2. news_predictions
   - id (UUID, primary key)
   - source_domain (str)
   - headline (text)
   - article_hash (str, unique — SHA256 of content, for deduplication)
   - severity (enum: NEUTRAL, SIGNIFICANT, SEVERE)
   - asset (str)
   - direction (enum: up, down, neutral)
   - magnitude_pct_low (float)
   - magnitude_pct_high (float)
   - confidence (float)
   - t_min_minutes (int)
   - t_max_minutes (int)
   - rationale (text)
   - trust_score_at_time (float)
   - created_at (timestamp)
   - outcome_checked (bool, default False)
   - actual_move_pct (float, nullable)
   - prediction_score (float, nullable)

3. trades
   - id (UUID, primary key)
   - asset (str, indexed)
   - direction (enum: long, short)
   - size_usd (float)
   - entry_price (float)
   - exit_price (float, nullable)
   - status (enum: open, closed, cancelled)
   - order_type (enum: market, limit, twap)
   - nn_confidence (float)
   - nn_direction_probs (JSON — {long, short, hold})
   - active_news_impact (JSON, nullable — snapshot of news features at entry)
   - regime_at_entry (str)
   - stop_loss (float)
   - take_profit (float)
   - opened_at (timestamp)
   - closed_at (timestamp, nullable)
   - pnl_usd (float, nullable)
   - pnl_pct (float, nullable)
   - kite_tx_hash (str, nullable)

4. model_checkpoints
   - id (UUID, primary key)
   - trade_count (int)
   - file_path (str)
   - cumulative_pnl (float)
   - win_rate_7d (float)
   - sharpe_24h (float, nullable)
   - created_at (timestamp)
   - is_current (bool)

5. agent_events
   - id (UUID, primary key)
   - event_type (str) -- e.g. STARTED, STOPPED, SEVERE_INTERRUPT, ROLLBACK
   - details (JSON)
   - created_at (timestamp)

Include:
- AsyncEngine and AsyncSession factory
- Base declarative model
- async init_db() function that creates all tables if not exist
- async get_session() context manager
- Enums defined as Python Enum classes matching the column types
```

---

## PROMPT 3 — Redis client & inter-process queue

```
Create /backend/memory/redis_client.py

Requirements:
- Async Redis client using aioredis (redis-py async)
- Connection pool, URL from config
- NewsImpact dataclass that can be serialised to/from JSON:
    severity: str  (NEUTRAL / SIGNIFICANT / SEVERE)
    asset: str
    direction: str
    magnitude_pct_low: float
    magnitude_pct_high: float
    confidence: float
    t_min_minutes: int
    t_max_minutes: int
    rationale: str
    source_domain: str
    trust_score: float
    created_at: str (ISO timestamp)

- PriorityNewsQueue class:
    - Uses Redis sorted set (ZADD/ZPOPMIN) with score: SEVERE=0, SIGNIFICANT=1
    - async put(impact: NewsImpact) -> None
    - async get_nowait() -> NewsImpact | None  (non-blocking, returns None if empty)
    - async size() -> int

- FeatureCache class:
    - async set_features(asset: str, features: dict) -> None  (TTL 60 seconds)
    - async get_features(asset: str) -> dict | None
    - async set_macro(data: dict) -> None  (TTL 900 seconds — 15 minutes)
    - async get_macro() -> dict | None

- HeartbeatClient class:
    - async ping(process_name: str) -> None  (writes timestamp to key with 10s TTL)
    - async check_alive(process_name: str) -> bool  (returns False if key expired)

- async get_redis() -> Redis  (returns connection from pool)
```

---

## PROMPT 4 — Binance WebSocket market data feed

```
Create /backend/data/market_feed.py

A production-grade async market data feed using Binance WebSocket streams.

Class: BinanceMarketFeed

Config (from constructor):
- symbols: list[str]  e.g. ["BTCUSDT", "ETHUSDT"]
- on_kline: async callback(symbol: str, kline: dict) -> None
- on_orderbook: async callback(symbol: str, book: dict) -> None  
- on_trade: async callback(symbol: str, trade: dict) -> None

Methods:
- async start() -> None  — opens WebSocket connections
- async stop() -> None   — graceful shutdown

Implementation requirements:
- Use websockets library directly (not python-binance)
- Subscribe to: {symbol}@kline_5m, {symbol}@depth20@100ms, {symbol}@aggTrade
- Parse kline to: {open, high, low, close, volume, is_closed, timestamp}
- Parse depth to: {bids: [[price, qty]...], asks: [[price, qty]...], timestamp}
- Parse aggTrade to: {price, qty, is_buyer_maker, timestamp}
- Auto-reconnect with exponential backoff (max 60s) on any disconnect or exception
- Log reconnect attempts with structlog
- Track last_message_time per stream; if no message in 30s, force reconnect
- Store last 300 closed klines per symbol in a deque (for indicator calculation)
- async get_klines(symbol: str) -> list[dict]  — returns stored klines
- async get_orderbook(symbol: str) -> dict | None  — returns latest book snapshot
- async get_recent_trades(symbol: str, n: int = 100) -> list[dict]

Also create:
- OHLCVBuffer class: stores last 500 klines as a pandas DataFrame, 
  updates on each closed kline, thread-safe access via asyncio.Lock
- Method: async get_dataframe(symbol: str) -> pd.DataFrame
```

---

## PROMPT 5 — Technical indicator engine

```
Create /backend/signals/technical.py

A stateless technical indicator calculator. All functions take a pandas DataFrame 
with columns [open, high, low, close, volume] and return a dict of indicator values.
All output values must be normalised to [0,1] or [-1,1] as specified.
Use pandas_ta for all indicators. Return the most recent bar's values only (iloc[-1]).

Functions to implement:

1. calculate_moving_averages(df: pd.DataFrame) -> dict
   Returns (all as % distance from close price, clipped to [-0.2, 0.2]):
   - ema_9_dist, ema_21_dist, ema_50_dist, ema_200_dist
   - golden_cross: 1.0 if EMA50 > EMA200 else 0.0
   - vwap_dist: (close - vwap) / vwap, clipped to [-0.05, 0.05], scaled to [-1,1]

2. calculate_momentum(df: pd.DataFrame) -> dict
   Returns:
   - rsi: rsi_14 / 100  → [0, 1]
   - macd_norm: macd_line / atr_14, clipped to [-2, 2], scaled to [-1, 1]
   - macd_hist_norm: macd_histogram / atr_14, clipped to [-2, 2], scaled to [-1, 1]
   - stoch_rsi: stochrsi_14 / 100  → [0, 1]
   - adx_norm: adx_14 / 100  → [0, 1]
   - rsi_divergence: 1.0 if price made new high but RSI didn't (last 10 bars), 
                    -1.0 if price made new low but RSI didn't, else 0.0

3. calculate_volatility(df: pd.DataFrame) -> dict
   Returns:
   - atr_norm: atr_14 / close  → normalise by rolling 100-bar percentile → [0, 1]
   - bb_width_norm: (bb_upper - bb_lower) / bb_mid, normalised by 100-bar percentile → [0, 1]
   - bb_pct_b: (close - bb_lower) / (bb_upper - bb_lower), clipped to [0, 1]

4. calculate_volume(df: pd.DataFrame) -> dict
   Returns:
   - volume_ratio: current_volume / sma_volume_20, clipped to [0, 5], scaled to [0, 1]
   - obv_slope: linear regression slope of OBV over last 10 bars, 
                normalised by std of OBV → [-1, 1]

5. calculate_fibonacci(df: pd.DataFrame, lookback: int = 50) -> dict
   Returns:
   - fib_nearest_level_pct: which fib level (23.6, 38.2, 50, 61.8, 78.6) 
                             normalised: level_pct / 100 → [0, 1]
   - fib_distance: (close - nearest_level_price) / close, clipped [-0.02, 0.02],
                   scaled to [-1, 1]. Positive = above level, negative = below
   - fib_strength: 1.0 if nearest level is 61.8 or 38.2, 0.6 otherwise

6. calculate_patterns(df: pd.DataFrame) -> dict
   Use TA-Lib (import talib). Return a 10-element list mapped to [0, 1]:
   patterns = [
     CDL_BULLISH_ENGULF, CDL_BEARISH_ENGULF,  # engulfing candles as proxies
     CDLHAMMER, CDLINVERTEDHAMMER,
     CDLMORNINGSTAR, CDLEVENINGSTAR,
     CDLDOJI, CDLSPINNINGTOP,
     CDLMARUBOZU, CDL3WHITESOLDIERS
   ]
   Each: 1.0 if talib returns > 0, 0.0 otherwise. Return as pattern_flags: list[float]

7. build_technical_feature_dict(df: pd.DataFrame) -> dict
   Calls all 6 functions above and merges results into a single flat dict.
   Add error handling: if any single function fails, fill its outputs with 0.5 (neutral).
   Log the failure with structlog but do not raise.
```

---

## PROMPT 6 — Order book & microstructure signals

```
Create /backend/signals/orderbook.py

All functions take raw order book and trade data from the BinanceMarketFeed.
All outputs normalised to [-1, 1] or [0, 1].

Functions:

1. calculate_book_imbalance(bids: list, asks: list, depth: int = 10) -> float
   Returns (bid_vol - ask_vol) / (bid_vol + ask_vol) → [-1, 1]
   Where bid_vol = sum of qty for top `depth` bid levels
   Returns 0.0 if book is empty or malformed.

2. calculate_depth_ratios(bids: list, asks: list) -> dict
   Returns:
   - bid_depth_ratio: sum(top 5 bids qty) / sum(top 20 bids qty) → [0, 1]
   - ask_depth_ratio: sum(top 5 asks qty) / sum(top 20 asks qty) → [0, 1]
   Indicates whether liquidity is concentrated near the market (high) or spread out (low)

3. calculate_cvd(trades: list[dict], window: int = 100) -> dict
   CVD = cumulative volume delta = sum of (buy_volume - sell_volume) per trade
   trade has fields: qty, is_buyer_maker (True = sell, False = buy)
   Returns:
   - cvd_slope: linear regression slope of CVD over last `window` trades,
                normalised by std of CVD → [-1, 1]
   - cvd_divergence: 1.0 if cvd_slope > 0 and last price change < 0 (bearish div)
                    -1.0 if cvd_slope < 0 and last price change > 0 (bullish div)
                     0.0 otherwise

4. detect_whale_activity(trades: list[dict], window: int = 200) -> float
   avg_qty = mean of trade sizes in last `window` trades
   whale_count = count of trades > 5 × avg_qty in last 50 trades
   Returns: min(whale_count / 5, 1.0) → [0, 1]
   (0 = no whales, 1 = many whale trades)

5. detect_liquidity_sweep(klines: list[dict], sr_levels: list[float]) -> dict
   For each S/R level in sr_levels:
     bullish_sweep: kline low < level AND kline close > level
     bearish_sweep: kline high > level AND kline close < level
   Returns:
   - bullish_sweep_strength: max sweep strength across all levels → [0, 1]
     strength = (close - level) / close if bullish_sweep else 0
   - bearish_sweep_strength: same logic for bearish → [0, 1]
   - sweep_detected: 1.0 if any sweep, 0.0 otherwise
   Use only the last 3 klines.

6. build_orderbook_feature_dict(bids, asks, trades, klines, sr_levels) -> dict
   Calls all 5 functions and merges. Error handling same as technical.py.
```

---

## PROMPT 7 — Feature vector builder

```
Create /backend/signals/features.py

This module assembles the complete 62-element feature vector for the LSTM.
It calls the technical and orderbook signal modules, fetches macro data from Redis cache,
and injects the current news impact (from the priority queue).

Class: FeatureVectorBuilder

Constructor args:
- redis_client: RedisClient
- technical_calculator: module reference (backend.signals.technical)
- orderbook_calculator: module reference (backend.signals.orderbook)

Method: async build(
    symbol: str,
    df: pd.DataFrame,           # OHLCV dataframe from OHLCVBuffer
    bids: list,
    asks: list,
    trades: list,
    sr_levels: list[float],
    regime: str,                # current regime string
    news_impact: NewsImpact | None
) -> np.ndarray

Feature vector layout (62 elements, all float32):
 [0]    price_pct_change      close vs open, clipped [-0.05, 0.05], scaled [-1,1]
 [1]    high_pct              (high - open) / open, clipped, scaled
 [2]    low_pct               (low - open) / open, clipped, scaled
 [3]    volume_norm           from calculate_volume()['volume_ratio'], scaled [0,1]
 [4]    spread_pct            (ask[0] - bid[0]) / close, clipped [0, 0.01], scaled [0,1]
 [5-8]  ema_dists             ema_9_dist, ema_21_dist, ema_50_dist, ema_200_dist
 [9]    golden_cross          from MAs
 [10]   vwap_dist             from MAs
 [11]   rsi                   from momentum
 [12]   macd_norm             from momentum
 [13]   macd_hist_norm        from momentum
 [14]   stoch_rsi             from momentum
 [15]   adx_norm              from momentum
 [16]   rsi_divergence        from momentum
 [17]   atr_norm              from volatility
 [18]   bb_width_norm         from volatility
 [19]   bb_pct_b              from volatility
 [20]   volume_ratio          from volume (already in [3], keep for sequence context)
 [21]   obv_slope             from volume
 [22]   fib_level_pct         from fibonacci
 [23]   fib_distance          from fibonacci
 [24]   fib_strength          from fibonacci
 [25-34] pattern_flags        10 elements from calculate_patterns()
 [35]   book_imbalance        from orderbook
 [36]   bid_depth_ratio       from orderbook
 [37]   ask_depth_ratio       from orderbook
 [38]   cvd_slope             from orderbook
 [39]   cvd_divergence        from orderbook
 [40]   whale_activity        from orderbook
 [41]   bullish_sweep         from orderbook
 [42]   bearish_sweep         from orderbook
 [43-48] regime_onehot        6 elements: [uptrend, downtrend, ranging, high_vol, news_driven, low_liq]
 [49]   news_direction        -1.0 (down), 0.0 (neutral/no news), +1.0 (up)
 [50]   news_magnitude        (magnitude_pct_low + magnitude_pct_high) / 2 / 10, clipped [0,1]
 [51]   news_confidence       0.0 if no news
 [52]   news_age_norm         1.0 - min(minutes_since_news / t_max_minutes, 1.0). 0 if no news.
 [53]   fear_greed_norm       from macro cache, default 0.5
 [54]   btc_dominance_norm    from macro cache, default 0.5
 [55]   funding_rate_norm     from macro cache, default 0.0
 [56]   oi_change_norm        from macro cache, default 0.0
 [57]   hour_sin              sin(2π × hour / 24)
 [58]   hour_cos              cos(2π × hour / 24)
 [59]   dow_sin               sin(2π × day_of_week / 7)
 [60]   dow_cos               cos(2π × day_of_week / 7)
 [61]   regime_confidence     float from regime detector [0,1]

Rules:
- If any individual feature calculation raises, substitute 0.0 and log warning
- Final vector must have exactly 62 elements — assert this before returning
- Return as np.ndarray, dtype=float32
- Log the full vector hash (sha256 of bytes) at DEBUG level for reproducibility
```

---

## PROMPT 8 — Market regime detector

```
Create /backend/signals/regime.py

Class: RegimeDetector

Method: detect(df: pd.DataFrame, active_news: NewsImpact | None) -> tuple[str, float]
Returns (regime_name, confidence) where regime_name is one of:
  "uptrend", "downtrend", "ranging", "high_volatility", "news_driven", "low_liquidity"

Logic (evaluate in order — first match wins):

1. "news_driven" — if active_news is not None and active_news.confidence > 0.65
   confidence = active_news.confidence

2. "high_volatility" — if ATR/close > 95th percentile of last 200 bars
   confidence = 0.85

3. "low_liquidity" — if current volume < 30th percentile of last 200 bars
   AND current UTC hour in [0,1,2,3,4,5] (weekend proxy — imperfect but fast)
   confidence = 0.70

4. "uptrend" — if ADX > 25 AND close > EMA_200 AND EMA_50 > EMA_200
   confidence = min(adx / 50, 1.0)

5. "downtrend" — if ADX > 25 AND close < EMA_200 AND EMA_50 < EMA_200
   confidence = min(adx / 50, 1.0)

6. "ranging" — default (ADX < 20)
   confidence = max(0.0, 1.0 - adx / 20)

All indicators computed from df using pandas_ta inline (do not import technical.py here).
Log the detected regime at DEBUG level each cycle.

Also implement:
- get_regime_onehot(regime: str) -> list[float]
  Returns 6-element one-hot list in order: [uptrend, downtrend, ranging, high_vol, news_driven, low_liq]
```

---

## PROMPT 9 — News ingestion pipeline

```
Create /backend/data/news_feed.py

An async news ingestion pipeline that reads from multiple sources 
and yields deduplicated NewsArticle objects.

Dataclass NewsArticle:
  headline: str
  body: str
  source_domain: str
  url: str
  published_at: datetime
  article_hash: str  — SHA256 of (headline + source_domain)

Class: NewsIngestionPipeline

Constructor args:
  - rss_urls: list[str]  — list of RSS feed URLs to poll
  - poll_interval_seconds: int = 60
  - on_article: async callback(article: NewsArticle) -> None

Methods:
- async start() -> None  — starts all ingestion loops concurrently
- async stop() -> None   — graceful shutdown

RSS sources to configure by default (as constants at top of file):
DEFAULT_RSS_FEEDS = [
  "https://feeds.reuters.com/reuters/businessNews",
  "https://rss.nytimes.com/services/xml/rss/nyt/Business.xml",
  "https://finance.yahoo.com/rss/",
  "https://www.coindesk.com/arc/outboundfeeds/rss/",
  "https://cointelegraph.com/rss",
  "https://feeds.bbci.co.uk/news/business/rss.xml",
]

Implementation:
- Use feedparser for RSS parsing
- Deduplication via in-memory set of article_hash (last 10,000 hashes, using collections.deque)
- Each RSS feed polled independently with its own asyncio task and its own error handling
- Any single feed failing must not stop others
- Exponential backoff on feed failure (start 30s, max 600s)
- Heartbeat: log "feed alive" at INFO level every 10 polls
- filter_relevant(article: NewsArticle) -> bool:
    Returns True if headline contains any of these keywords (case-insensitive):
    ["bitcoin", "btc", "ethereum", "eth", "crypto", "fed", "rate", "inflation",
     "sec", "cftc", "exchange", "hack", "regulation", "stablecoin", "defi",
     "market", "stock", "nasdaq", "sp500", "recession", "tariff", "sanctions"]
    Returns False otherwise (skip irrelevant news entirely)
```

---

## PROMPT 10 — Credibility engine

```
Create /backend/agents/credibility.py

Class: CredibilityEngine

Constructor args:
  - db_session_factory: async session factory from database.py
  - anthropic_client: anthropic.AsyncAnthropic

BASE_TRUST_SCORES (module-level constant):
{
  "reuters.com": 0.95, "bbc.co.uk": 0.93, "bloomberg.com": 0.92,
  "nytimes.com": 0.88, "ft.com": 0.87, "wsj.com": 0.86,
  "coindesk.com": 0.80, "yahoo.com": 0.78, "cointelegraph.com": 0.65,
  "cryptopanic.com": 0.55, "unknown": 0.25
}

FAST_LANE_THRESHOLD = 0.85  # skip LLM plausibility check if above this

Methods:

1. async get_trust_score(source_domain: str) -> float
   - Look up in DB. If not found, insert with base score from BASE_TRUST_SCORES (or 0.25 if unknown)
   - Return current_score

2. async check_plausibility(article: NewsArticle) -> float
   - Only called for sources below FAST_LANE_THRESHOLD
   - Calls claude-haiku with this prompt (temperature=0):
     "Rate the plausibility of this financial news headline from 0.0 to 1.0.
      Respond with a single float only. No explanation.
      Headline: {article.headline}
      Source: {article.source_domain}"
   - Parse response as float, clamp to [0.0, 1.0]
   - On parse error, return 0.5

3. async score_article(article: NewsArticle) -> tuple[float, bool]
   Returns (final_trust_score, is_fast_lane)
   - base = await get_trust_score(article.source_domain)
   - if base >= FAST_LANE_THRESHOLD: return (base, True)
   - plausibility = await check_plausibility(article)
   - final = 0.7 * base + 0.3 * plausibility
   - return (final, False)

4. async update_trust_score(source_domain: str, prediction_score: float) -> None
   - EMA update: new_score = 0.92 * current + 0.08 * prediction_score
   - Clamp result to [0.10, 0.97]
   - Update in DB
```

---

## PROMPT 11 — LLM news severity agent

```
Create /backend/agents/news_agent.py

This is Process 2 — the LLM news processing loop. It runs as a standalone process.

Class: LLMNewsAgent

Constructor args:
  - news_pipeline: NewsIngestionPipeline
  - credibility_engine: CredibilityEngine
  - news_queue: PriorityNewsQueue  (Redis-backed)
  - anthropic_client: anthropic.AsyncAnthropic
  - db_session_factory
  - min_trust_to_analyse: float = 0.40

SEVERITY_CLASSIFICATION_PROMPT (module-level constant):
"""You are a financial severity classifier for a live trading system.
Classify the market impact of this news article.

Article headline: {headline}
Article body (first 500 chars): {body}
Source trust score: {trust_score}

Severity rules:
- NEUTRAL: routine news, confidence < 0.5, magnitude < 1%, or unrelated to markets
- SIGNIFICANT: clear directional catalyst, confidence > 0.6, magnitude 1-10%, 
               no systemic risk, does NOT require immediate position closure
- SEVERE: systemic risk event — exchange collapse, regulatory ban, protocol exploit, 
          stablecoin depeg, magnitude likely > 10%

Respond in raw JSON only. No preamble, no markdown, no explanation:
{
  "severity": "SIGNIFICANT",
  "asset": "BTC-USD",
  "direction": "down",
  "magnitude_pct_low": 3.0,
  "magnitude_pct_high": 8.0,
  "confidence": 0.78,
  "t_min_minutes": 5,
  "t_max_minutes": 30,
  "rationale": "one sentence"
}
If NEUTRAL, respond: {"severity": "NEUTRAL"}"""

Methods:

1. async analyse_article(article: NewsArticle, trust_score: float) -> NewsImpact | None
   - Call claude-sonnet-4 with SEVERITY_CLASSIFICATION_PROMPT, temperature=0, max_tokens=300
   - Parse JSON response using extract_json() helper (strip fences, find first { to last })
   - If severity is NEUTRAL, return None
   - Build NewsImpact from parsed JSON + article metadata + trust_score
   - Save to news_predictions table in DB
   - Return NewsImpact

2. async run() -> None  (the main loop — called by multiprocessing.Process)
   - Start news_pipeline
   - Loop: for each article from pipeline:
       trust_score, is_fast = await credibility_engine.score_article(article)
       if trust_score < min_trust_to_analyse: continue
       impact = await analyse_article(article, trust_score)
       if impact is not None:
           await news_queue.put(impact)
           log impact at INFO level
   - Wrap in try/except — log errors, continue loop (never crash)
   - Write heartbeat to Redis every 30 seconds

3. async check_prediction_outcomes() -> None
   - Runs as a separate asyncio task within run()
   - Every 5 minutes: query news_predictions where outcome_checked=False 
     AND created_at < now() - t_max_minutes
   - For each: fetch actual price move from market feed cache
   - Score with score_news_prediction() formula
   - Update DB, update trust score via credibility engine
```

---

## PROMPT 12 — LSTM neural network model

```
Create /backend/agents/nn_model.py

Implement the TradingLSTM and PersistentTradingModel classes.

TradingLSTM (nn.Module):
- Input: tensor of shape (batch, sequence_length, 62)
- LSTM: input_size=62, hidden_size=128, num_layers=2, dropout=0.2, batch_first=True
- Attention: nn.MultiheadAttention(128, num_heads=4, batch_first=True)
- fc_direction: Linear(128, 3)  — outputs [long_prob, short_prob, hold_prob]
- fc_size: Linear(128, 1)       — outputs position size [0, 1]
- Forward: lstm → attention over lstm output → take last timestep → dropout 
           → softmax(direction), sigmoid(size)
- Return: tuple(direction_probs: tensor[3], size: tensor[1])

ReplayBuffer:
- max_size: int = 10_000
- Stores TradeExperience dataclasses:
    features_sequence: np.ndarray  (sequence_length, 62)
    direction_taken: int           (0=long, 1=short, 2=hold)
    actual_pnl_pct: float
    reward: float                  (computed from pnl_pct: tanh(pnl_pct * 10))
- add(experience: TradeExperience) -> None  (evict oldest if full)
- sample(n: int) -> list[TradeExperience]   (random sample, n <= len(buffer))
- __len__() -> int

PersistentTradingModel:
Constants:
  MODEL_PATH = Path("models/trading_lstm_latest.pt")
  CHECKPOINT_DIR = Path("models/checkpoints/")
  SEQUENCE_LENGTH = 60  # 60 candles of context (5 hours on 5m chart)

Constructor:
  - Instantiate TradingLSTM
  - Adam optimiser, lr=1e-4, weight_decay=1e-5
  - CosineAnnealingLR scheduler, T_max=500
  - ReplayBuffer(max_size=10_000)
  - Call _load_or_initialise()

_load_or_initialise():
  - If MODEL_PATH exists: load checkpoint, restore model + optimiser state
  - Else: log "first run", call _pretrain_on_synthetic_data() as placeholder,
    then safe_checkpoint()
  - Log trade_count and cumulative_pnl on load

safe_checkpoint(label: str = ""):
  - Write to MODEL_PATH.with_suffix(".tmp") then rename (atomic)
  - Also copy to CHECKPOINT_DIR / f"ckpt_{self.trade_count}.pt"
  - Keep only last 10 checkpoints in CHECKPOINT_DIR (delete oldest)
  - Update model_checkpoints table in DB

infer(feature_sequence: np.ndarray) -> tuple[str, float, dict]:
  - Input: (SEQUENCE_LENGTH, 62) numpy array
  - Convert to tensor, add batch dim, set model to eval, torch.no_grad()
  - Returns: (decision: "long"/"short"/"hold", position_size_pct: float, 
              probs: {"long": float, "short": float, "hold": float})
  - decision = argmax of direction_probs
  - position_size_pct = float(size_output) clamped to [0.02, 0.20]

online_update(experience: TradeExperience) -> None:
  - Add to replay buffer
  - Increment trade_count
  - If trade_count % 10 == 0 and len(buffer) >= 32:
      batch = buffer.sample(32)
      Compute loss: cross_entropy(predicted_direction, actual_better_direction)
        where actual_better_direction = 0 (long) if pnl > 0 and direction was long,
              1 (short) if pnl > 0 and direction was short, 2 (hold) if pnl near 0
      Clip gradients to 1.0, step optimiser, step scheduler
  - If trade_count % 50 == 0: safe_checkpoint()

check_and_rollback(recent_pnl_pct: float, threshold: float = -0.05) -> bool:
  - If recent_pnl_pct < threshold:
      Load second-most-recent checkpoint from CHECKPOINT_DIR
      Restore weights
      Log ROLLBACK event to DB agent_events table
      Return True
  - Return False

_pretrain_on_synthetic_data():
  - Generate 1000 synthetic feature sequences (random noise around 0.5)
  - Labels: hold for all (conservative default)
  - Train for 5 epochs
  - This is just to initialise weights — real learning happens online
```

---

## PROMPT 13 — NN trading agent (Process 1)

```
Create /backend/agents/nn_agent.py

This is Process 1 — the main NN trading loop. Runs as a multiprocessing.Process.

Class: NNTradingAgent

Constructor args:
  - market_feed: BinanceMarketFeed
  - feature_builder: FeatureVectorBuilder
  - regime_detector: RegimeDetector
  - model: PersistentTradingModel
  - risk_manager: RiskManager
  - execution_engine: ExecutionEngine
  - news_queue: PriorityNewsQueue
  - severe_flag: multiprocessing.Value(ctypes.c_bool)
  - cycle_interval_seconds: float = 5.0  (run every 5 seconds)
  - symbols: list[str]

State:
  - feature_sequences: dict[str, deque]  — rolling (SEQUENCE_LENGTH, 62) per symbol
  - open_trades: dict[str, Trade]         — currently open position per symbol
  - current_news_impact: NewsImpact | None
  - news_impact_expires_at: datetime | None

async run() -> None:
  Main loop:
  1. Check severe_flag.value — if True: call _emergency_protocol(), await asyncio.sleep(60), continue
  2. Drain news_queue (non-blocking) — update self.current_news_impact if SIGNIFICANT
     If SEVERE in queue: set severe_flag, call _emergency_protocol(), continue
  3. Expire news_impact if now > news_impact_expires_at
  4. For each symbol:
     a. Get df from market_feed.get_dataframe(symbol)
     b. Get bids, asks from market_feed.get_orderbook(symbol)
     c. Get trades from market_feed.get_recent_trades(symbol)
     d. Detect regime
     e. Build feature vector
     f. Append to feature_sequences[symbol]
     g. If len(feature_sequences[symbol]) < SEQUENCE_LENGTH: continue (not enough data yet)
     h. Build sequence array from deque
     i. decision, size_pct, probs = model.infer(sequence)
     j. Build TradeDecision object
     k. If risk_manager.approve(decision, portfolio_state): 
            await execution_engine.execute(decision)
     l. Write heartbeat to Redis
  5. await asyncio.sleep(cycle_interval_seconds)

_emergency_protocol():
  - Log SEVERE_INTERRUPT to agent_events table
  - For each open trade: attempt to close at market
  - Set all open_trades to cancelled
  - Log outcome

async _on_trade_closed(trade: Trade, pnl_pct: float):
  - Build TradeExperience from trade
  - Call model.online_update(experience)
  - Check rollback threshold

TradeDecision dataclass:
  symbol: str
  direction: str        # "long" / "short" / "hold"
  size_pct: float       # fraction of available capital [0, 0.20]
  nn_confidence: float  # max(probs.values())
  nn_probs: dict
  regime: str
  active_news: NewsImpact | None
  timestamp: datetime
```

---

## PROMPT 14 — Risk manager

```
Create /backend/risk/manager.py

Class: RiskManager

HARD_LIMITS (class constant):
{
  "max_portfolio_drawdown_pct": 15.0,
  "max_daily_loss_pct": 5.0,
  "max_single_position_pct": 20.0,
  "min_nn_confidence": 0.52,
  "max_trades_per_hour": 20,
  "min_position_usd": 12.0,
  "max_position_usd": 5000.0,
  "min_signal_classes_agreeing": 1,   # at least 1 class strong signal (NN handles fusion)
}

State tracked:
  - portfolio_value_usd: float
  - peak_portfolio_value: float
  - daily_pnl_usd: float (resets at midnight UTC)
  - trades_this_hour: int (resets each hour)
  - last_hour_reset: datetime
  - last_day_reset: datetime
  - is_halted: bool (set True on max drawdown breach — requires manual reset)

Methods:

approve(decision: TradeDecision, portfolio_state: dict) -> tuple[bool, str]:
  Returns (approved: bool, reason: str)
  
  Checks in order (fail fast):
  1. if is_halted: return False, "HALTED: max drawdown exceeded — manual reset required"
  2. if decision.direction == "hold": return False, "HOLD decision — no trade"
  3. if decision.nn_confidence < HARD_LIMITS["min_nn_confidence"]: 
       return False, f"Low confidence: {decision.nn_confidence:.3f}"
  4. Reset hourly/daily counters if needed
  5. if trades_this_hour >= max_trades_per_hour: return False, "Trade rate limit"
  6. Compute position_usd = decision.size_pct * portfolio_state["available_cash"]
  7. if position_usd < min_position_usd: return False, "Below min notional"
  8. if position_usd > max_position_usd: return False, "Above max position"
  9. Compute current drawdown = (peak - current) / peak * 100
  10. if current drawdown > max_portfolio_drawdown_pct: 
        is_halted = True; return False, "MAX DRAWDOWN EXCEEDED — HALTED"
  11. if daily_pnl_pct < -max_daily_loss_pct: return False, "Daily loss limit"
  12. trades_this_hour += 1
  13. return True, "APPROVED"

update_portfolio(portfolio_state: dict) -> None:
  - Update portfolio_value_usd and peak_portfolio_value
  - Update daily_pnl_usd

reset_halt() -> None:
  - Set is_halted = False (requires explicit call — never auto-reset on halt)

get_status() -> dict:
  - Returns all current risk state as dict (for dashboard API)
```

---

## PROMPT 15 — Execution engine & Kite chain client

```
Create /backend/execution/engine.py and /backend/execution/kite_chain.py

--- engine.py ---

Class: ExecutionEngine

Constructor args:
  - exchange: ccxt.binance (pre-configured with API keys)
  - kite_chain: KiteChainClient
  - paper_mode: bool = True  (MUST default to True — never default to live)
  - db_session_factory

Methods:

async execute(decision: TradeDecision, available_cash: float) -> Trade | None:
  1. Compute size_usd = decision.size_pct * available_cash
  2. qty, price = normalise_order(decision.symbol, size_usd)
  3. If paper_mode: simulate fill, create Trade with status=open, log "PAPER TRADE"
  4. If not paper_mode:
       order_type = select_order_type(decision)
       place via ccxt
       Confirm order ID in response — if missing, raise ExecutionError
  5. Save Trade to DB
  6. Fire-and-forget: asyncio.create_task(kite_chain.log_trade_decision(trade, decision))
  7. Return Trade

normalise_order(symbol: str, size_usd: float) -> tuple[float, float]:
  market = exchange.market(symbol)
  price = exchange.fetch_ticker(symbol)["last"]
  min_notional = market["limits"]["cost"]["min"] or 10.0
  qty = exchange.amount_to_precision(symbol, size_usd / price)
  if float(qty) * price < min_notional:
      raise ValueError(f"Below min notional")
  return float(qty), price

select_order_type(decision: TradeDecision) -> str:
  if decision.active_news and decision.active_news.confidence > 0.75: return "market"
  if decision.size_pct > 0.10: return "limit"  # large order → limit
  return "limit"  # default

--- kite_chain.py ---

Class: KiteChainClient

Constructor args:
  - rpc_url: str
  - private_key: str  (NEVER log this)
  - agent_address: str

Use web3.py. Connect to rpc_url. Account from private_key.

Methods (all async, all fire-and-forget safe — log errors, never raise to caller):

async log_trade_decision(trade: Trade, decision: TradeDecision) -> str | None:
  - Encode trade summary as UTF-8 JSON string
  - Send a transaction to self (agent_address → agent_address, value=0)
  - Set data field to encoded trade summary (hex)
  - Handle gas: fetch gas_price, multiply by 1.1, set reasonable gas limit (100_000)
  - Sign with private_key, send
  - Update trade.kite_tx_hash in DB with returned tx hash
  - Return tx hash or None on failure

async log_prediction(prediction_id: str, summary: dict) -> str | None:
  - Same pattern as above, encode prediction summary

async get_agent_reputation() -> dict:
  - Fetch from DB (not on-chain for now — placeholder for future on-chain reputation)
  - Return: {trade_count, win_rate, avg_prediction_score, total_pnl_usd}

Note: For the hackathon, on-chain logging via transaction data field is acceptable.
Real smart contract integration can be added in Week 4 if Kite AI provides ABI.
```

---

## PROMPT 16 — FastAPI backend & REST API

```
Create /backend/main.py and /backend/api/ directory.

/backend/main.py:
- Process launcher using multiprocessing
- Spawns NNTradingAgent as Process 1 and LLMNewsAgent as Process 2
- Starts FastAPI app in the main process (uvicorn)
- Handles SIGTERM/SIGINT: graceful shutdown — send stop signals to child processes,
  call model.safe_checkpoint(), then exit
- shared_severe_flag: multiprocessing.Value(ctypes.c_bool, False) — passed to both processes
- shared_news_queue: Redis-backed PriorityNewsQueue (both processes use same Redis key)

/backend/api/routes.py — FastAPI router with these endpoints:

GET /api/health
  Returns: {status: "ok", nn_alive: bool, news_alive: bool, model_trade_count: int}
  Check liveness via Redis heartbeats.

GET /api/portfolio
  Returns: {
    total_value_usd, available_cash, unrealised_pnl, daily_pnl, 
    drawdown_pct, peak_value, is_halted
  }

GET /api/positions
  Returns: list of open trades from DB with current unrealised PnL

GET /api/trades?limit=50&offset=0
  Returns: paginated list of all trades (open + closed) from DB

GET /api/signals/latest
  Returns: {
    symbol: str,
    regime: str, regime_confidence: float,
    nn_decision: str, nn_confidence: float, nn_probs: dict,
    feature_snapshot: dict (key indicator values only, not full 62-vector),
    active_news: NewsImpact | None,
    timestamp: str
  }
  Source: Redis feature cache (latest cycle data)

GET /api/news/recent?limit=20
  Returns: recent news_predictions from DB ordered by created_at desc

GET /api/audit?limit=20
  Returns: trades with kite_tx_hash, prediction outcomes, agent_events

GET /api/risk/status
  Returns: RiskManager.get_status()

POST /api/risk/reset-halt
  Body: {confirm: true}
  Calls risk_manager.reset_halt()
  Protected: require a secret header (X-Admin-Key from env)

WebSocket /ws/live
  Pushes live updates every 5 seconds:
  {type: "cycle_update", data: same shape as GET /api/signals/latest}
  {type: "trade_opened", data: Trade}
  {type: "trade_closed", data: Trade}
  {type: "news_impact", data: NewsImpact}
  {type: "severe_alert", data: {message, timestamp}}

Enable CORS for frontend origin. Use pydantic response models for all endpoints.
```

---

## PROMPT 17 — Frontend: Next.js dashboard

```
Create the full Next.js 14 frontend for the AI trading agent dashboard.
Use TypeScript, Tailwind CSS, shadcn/ui components, and lightweight-charts for price charts.
Connect to the FastAPI backend via REST + WebSocket.
Connect to Kite AI chain via wagmi + viem for wallet connection and on-chain audit viewing.

Design requirements:
- Dark theme by default (trading terminal aesthetic — dark background, green/red PnL)
- Responsive: works on desktop, acceptable on tablet
- Real-time: all data updates via WebSocket without page reload

Pages and components:

1. /dashboard (main page)
   Layout: top navbar + main grid

   Navbar:
   - Agent status indicator (green dot = running, red = halted, yellow = paper mode)
   - "PAPER MODE" badge if paper_mode=true (always visible)
   - Wallet connect button (wagmi ConnectButton for Kite chain)
   - Agent address displayed (truncated, copy on click)

   Main grid (4 columns, responsive):
   - PnLCard: total value, daily PnL (green/red), drawdown %, peak value
   - ConfidenceGauge: semi-circular gauge 0-100%, current NN confidence, coloured by threshold
   - RegimeWidget: current regime badge + confidence bar
   - ActiveNewsWidget: latest news impact (if any) — severity badge, asset, direction, rationale

   Below grid:
   - TradingChart: lightweight-charts candlestick for BTC/USDT, 5m candles (last 200)
     Fetched from Binance public API (no auth needed for historical)
     Overlays: EMA 9 (thin), EMA 21 (medium), EMA 200 (thick, different colour)
   - SignalFeed: live list of NN decisions with confidence, direction, timestamp
     Items animate in from top, max 20 shown, auto-scroll

2. /positions
   - Table of open positions: asset, direction badge, size USD, entry price, 
     current price, unrealised PnL (green/red), stop loss, take profit
   - Auto-refreshes every 10s

3. /news
   - Table of recent news predictions: headline, source, severity badge, 
     direction badge, confidence %, trust score, rationale
   - Filter by severity (SIGNIFICANT / SEVERE)

4. /audit
   - Table of trades with Kite chain tx hash
   - Each row: trade details + tx hash as clickable link to Kite chain explorer
   - Show prediction_score if available (how accurate was the prediction)
   - "Verified on Kite Chain" badge if tx_hash present

5. /risk
   - Current risk limits display (read-only cards)
   - Current state: drawdown, daily PnL, trades this hour
   - Circuit breaker status (red card if halted)
   - Reset Halt button (admin only — prompts for admin key)

WebSocket integration:
- Connect to ws://backend/ws/live on mount
- On cycle_update: update signals display, refresh confidence gauge, refresh regime
- On trade_opened/closed: toast notification, refresh positions
- On news_impact: toast notification with severity colour coding
- On severe_alert: full-screen red alert banner with dismiss button

Web3 integration (wagmi + viem):
- Chain config: Kite AI chain (chainId, RPC, explorer from env)
- Wallet connect: MetaMask / WalletConnect
- On /audit page: "Verify on Chain" button per trade — opens explorer link
- Display connected wallet address in navbar
```

---

## PROMPT 18 — Pre-training data pipeline & initial model setup script

```
Create /scripts/pretrain.py

A standalone script (run once before first launch) that:

1. Downloads 30 days of BTC/USDT 5-minute OHLCV data from Binance public API
   Endpoint: GET https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=5m&limit=8640
   (8640 = 30 days * 288 candles/day)

2. Computes all technical indicator features for each candle using the same
   build_technical_feature_dict() function from signals/technical.py
   (orderbook + news features default to neutral: 0.0 or 0.5)

3. Labels each candle:
   Look 3 candles ahead (15 minutes):
   - If future_close > current_close * 1.005 → label = 0 (long)
   - If future_close < current_close * 0.995 → label = 1 (short)
   - Else → label = 2 (hold)

4. Builds sequence dataset:
   Each sample: (SEQUENCE_LENGTH=60 feature vectors, label)
   Sliding window, step=1

5. Train/validation split: 80/20 chronological (no shuffle — time series)

6. Trains TradingLSTM for 30 epochs:
   - Loss: CrossEntropyLoss for direction (ignore size output during pretraining)
   - Adam, lr=1e-3
   - Print train loss and val accuracy each epoch
   - Save best checkpoint (lowest val loss) to models/pretrain_best.pt

7. Copy pretrain_best.pt to models/trading_lstm_latest.pt

8. Print summary: val accuracy, class distribution, feature stats (mean/std per feature)
   Flag any features with std < 0.01 (likely not normalising correctly)

Run this script with: python scripts/pretrain.py
It should complete in under 10 minutes on a CPU.
```

---

## PROMPT 19 — Docker & deployment configuration

```
Create complete Docker and deployment configuration.

docker-compose.yml (development):
Services:
- postgres:16-alpine
    environment: POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD from env
    volume: postgres_data
    healthcheck: pg_isready
- redis:7-alpine
    volume: redis_data
    command: redis-server --maxmemory 256mb --maxmemory-policy allkeys-lru
- backend:
    build: ./backend
    Dockerfile: Python 3.11-slim, pip install -r requirements.txt
    command: python main.py
    depends_on: postgres, redis
    volumes: ./backend:/app, ./models:/app/models  (models dir persisted on host)
    environment: all env vars from .env
    ports: 8000:8000
    restart: unless-stopped
- frontend:
    build: ./frontend
    Dockerfile: node:20-alpine, npm ci, npm run build
    command: npm start
    depends_on: backend
    ports: 3000:3000
    environment: NEXT_PUBLIC_API_URL, NEXT_PUBLIC_WS_URL, NEXT_PUBLIC_KITE_CHAIN_ID etc

volumes: postgres_data, redis_data

/backend/Dockerfile:
FROM python:3.11-slim
RUN apt-get update && apt-get install -y gcc g++ libta-lib-dev  (for TA-Lib)
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python", "main.py"]

/frontend/Dockerfile:
FROM node:20-alpine AS builder
WORKDIR /app
COPY package*.json .
RUN npm ci
COPY . .
RUN npm run build

FROM node:20-alpine AS runner
WORKDIR /app
COPY --from=builder /app/.next/standalone ./
COPY --from=builder /app/.next/static ./.next/static
CMD ["node", "server.js"]

README.md — complete setup guide:
1. Prerequisites: Docker, Docker Compose, Git
2. Clone repo
3. cp .env.example .env and fill in all values
4. docker-compose up --build
5. In a second terminal: docker-compose exec backend python scripts/pretrain.py
6. Visit http://localhost:3000
7. Connect MetaMask to Kite AI chain (include chain params)
8. Expected state after setup: paper mode active, NN loading pretrained weights,
   news feed connecting, dashboard showing live data within 2 minutes

Vercel deployment instructions (for frontend only):
- Set all NEXT_PUBLIC_ env vars in Vercel dashboard
- Backend must be deployed separately (Railway, Render, or AWS EC2)
- Note: multiprocessing requires a persistent server — Vercel serverless is NOT 
  compatible with the backend. Use a VPS or container service.

AWS EC2 one-liner deploy:
  docker-compose -f docker-compose.prod.yml up -d
  (create docker-compose.prod.yml variant with resource limits and logging config)
```

---

## PROMPT 20 — Unit test scaffolding

```
Create /tests/ directory with test scaffolding.

/tests/conftest.py:
- pytest fixtures: async_session (test DB), mock_redis, sample_feature_vector (62 float32s),
  sample_ohlcv_df (200 rows of realistic BTC OHLCV data — generate with realistic values)
  sample_news_impact (SIGNIFICANT, BTC, down, confidence=0.75)
  sample_trade_decision (long, size_pct=0.05, nn_confidence=0.65)

/tests/test_risk_manager.py:
- test_approve_normal_trade() — should pass all checks
- test_reject_low_confidence() — confidence below threshold
- test_reject_above_max_position() — size_pct * cash > max_position_usd
- test_halt_on_max_drawdown() — portfolio drops 16%, should halt
- test_halt_blocks_all_subsequent_trades() — after halt, all trades blocked
- test_rate_limit() — 21 trades in one hour, 21st should be rejected
- test_daily_loss_limit() — daily PnL below threshold

/tests/test_feature_vector.py:
- test_vector_has_correct_length() — assert len == 62
- test_vector_all_finite() — no NaN or inf
- test_vector_bounds() — all values in [-1.5, 1.5] (allow slight overflow from tanh)
- test_news_features_default_neutral() — when no news, features [49-52] are [0,0,0,0]
- test_news_features_injected() — when SIGNIFICANT news injected, features update

/tests/test_model_persistence.py:
- test_save_and_load() — save checkpoint, load in new instance, weights identical
- test_trade_count_persists() — trade_count increments and survives reload
- test_corrupt_checkpoint_recovery() — corrupt latest.pt, verify fallback to previous
- test_online_update_changes_weights() — after 32 updates, weights should differ from initial

/tests/test_interrupt_protocol.py:
- test_significant_injects_features() — SIGNIFICANT queued, next cycle uses news features
- test_severe_halts_immediately() — severe_flag set, emergency protocol called
- test_severe_flag_blocks_execution() — severe flag true, no order placed

Each test file: use pytest-asyncio for async tests. Mock external calls (ccxt, Claude API).
```

---

---

# Manual Review Summary

## What must be implemented and checked by hand after vibe coding

### ❌ Do not vibe code these — write manually and review line by line

**1. Risk manager checks (manager.py)**
Every individual limit check must be manually written and manually unit tested. The automated tests scaffold is there but you must write the actual test assertions — do not let the agent generate expected values. Run each test, understand why it passes.

**2. LSTM training loop in PersistentTradingModel.online_update()**
The gradient computation, loss function, and gradient clipping must be verified manually. Specifically check: does the loss actually decrease over 10 updates on mock data? Print it. If loss is NaN after one step, the learning rate is too high or features are not normalised.

**3. Atomic SEVERE flag implementation**
`multiprocessing.Value(ctypes.c_bool)` must be used — not a regular Python variable, not a Queue item. Verify this explicitly. A regular Python bool shared between processes will not propagate. Test this in isolation: set the flag in one process, read it in another within 100ms.

**4. Severity classification prompt (news_agent.py)**
Do not let the agent finalise the prompt. Read it carefully. The boundary between SIGNIFICANT and SEVERE is a judgment call — a wrong boundary means the agent either over-reacts to routine news or under-reacts to genuine crises. Test it manually against 20 real headlines from your news feeds before going live.

**5. Kite chain transaction signing (kite_chain.py)**
Every line that touches `PRIVATE_KEY` must be reviewed. Confirm: the private key is never logged (check every logger call in the file), the key is only used in `sign_transaction()` and nowhere else. Run the transaction code against Kite testnet first.

**6. Feature normalisation in features.py**
The `build()` method must produce exactly 62 elements in the correct order every time. After the agent generates it, add a manual assertion test: fill a real OHLCV DataFrame, run build(), print each element and its index, verify it matches the documented layout. One off-by-one error here silently corrupts every NN inference.

**7. Model checkpoint atomic write**
Verify the `.tmp` → rename pattern works on your OS. On Windows, `Path.replace()` may not be atomic. If deploying on Linux (Docker), this is fine. Do not assume.

---

### ⚠️ Vibe-coded but likely to have errors — review these carefully

**news_feed.py — keyword filter**
The agent will likely generate a reasonable list but may miss crypto-specific terms. Review `filter_relevant()` and expand the keyword list. Also check: does the deduplication deque actually prevent the same article being processed twice across different RSS feeds that syndicate the same story? Test with two feeds that carry Reuters content.

**technical.py — RSI divergence logic**
The divergence check ("price new high but RSI didn't") over the last N bars is subtle. The agent will likely implement a simplified version. Check: is it comparing the correct rolling windows? Is it triggering on trivial noise? Log divergence detections for the first 24 hours and review them manually against a chart.

**technical.py — VWAP calculation**
VWAP resets at the start of each trading session. For crypto (24/7 markets), the reset convention must be defined explicitly. The agent will likely use a rolling 24-hour VWAP. Decide if this is what you want and verify the implementation matches.

**orderbook.py — CVD computation**
CVD requires trade-level data (aggTrade stream), not just OHLCV. Verify the agent is correctly identifying buyer vs seller-initiated trades using `is_buyer_maker`. The Binance field: `is_buyer_maker=True` means the buyer was the market maker (i.e. the trade was a sell). This is counterintuitive. Confirm the CVD sign is correct.

**nn_agent.py — feature sequence deque management**
The rolling deque of (SEQUENCE_LENGTH, 62) feature vectors must correctly drop the oldest and append the newest each cycle. The agent may generate code that appends the whole vector each cycle correctly, but verify: after exactly 60 cycles, does `np.array(deque)` produce shape `(60, 62)` with the correct ordering (oldest first)?

**credibility.py — EMA alpha**
The agent will likely use alpha=0.1 or similar. Verify the convergence rate is appropriate: with alpha=0.08 and a base score of 0.95, it takes ~50 predictions before a consistently wrong source drops below 0.70. This is intentionally slow — but confirm this matches your intent.

**pretrain.py — lookahead labels**
The labelling function "look 3 candles ahead" introduces lookahead bias during the offline training phase — this is acceptable for pretraining since we're just initialising weights. But verify the agent hasn't introduced lookahead bias in the live feature pipeline. The live pipeline must only use data available at prediction time.

**frontend WebSocket reconnection**
The agent will likely write a simple `useEffect` with a WebSocket connection. Verify it handles: connection drops (auto-reconnect with backoff), server restarts (reconnect without page reload), and stale state (clears old data on reconnect). Most generated WebSocket code is too naive for a live trading demo.

**docker-compose.yml — models directory volume**
The model checkpoint files must persist on the host machine between Docker restarts. Verify the volume mount for `/models` is on the host (e.g. `./models:/app/models`) not a named Docker volume — named volumes are harder to inspect and back up. If you lose the model weights, you lose the training history.

**execution engine paper mode default**
Manually verify that `paper_mode=True` is the default in every code path that instantiates `ExecutionEngine`. grep the codebase for `ExecutionEngine(` and confirm no call passes `paper_mode=False` without an explicit config flag. This is a safety check, not a debugging check.

---

### 🔍 Things the agent almost certainly got wrong

**PyTorch multiprocessing + CUDA**
If you are running on a machine with a GPU, PyTorch models cannot be shared between processes using standard `multiprocessing`. You must use `multiprocessing.set_start_method("spawn")` and pass the model path (not the model object) to the child process, loading it inside the child. The agent will likely try to pass the model object directly, which will fail or silently corrupt on GPU.

**Redis sorted set priority queue**
The agent may implement the priority queue as a regular Redis list (RPUSH/BLPOP) rather than a sorted set. A list has no priority ordering — SEVERE events would not jump the queue. Verify the implementation uses ZADD with numeric score and ZPOPMIN for dequeueing.

**CCXT exchange initialisation in a subprocess**
CCXT connections and authenticated sessions must be initialised inside the subprocess, not in the parent process and passed to the child. The agent may generate code that creates the CCXT exchange in `main.py` and passes it to the Process — this will not work correctly with multiprocessing. Each process must create its own CCXT instance from config.

**lightweight-charts in Next.js**
The `lightweight-charts` library does not support SSR (server-side rendering). The agent will likely import it without `dynamic()` wrapping, causing a `window is not defined` error in Next.js. Every chart component must use `dynamic(() => import(...), { ssr: false })`.

**wagmi chain configuration for Kite AI**
The agent will use a placeholder chain config. You must replace the chainId, RPC URL, and explorer URL with the actual Kite AI chain parameters from their documentation. Do not assume the agent's placeholders are correct.

**TA-Lib installation in Docker**
TA-Lib requires system-level C libraries (`libta-lib-dev`). The agent's Dockerfile likely installs it correctly, but verify the exact package name for `python:3.11-slim` (Debian-based). The Python wrapper `TA-Lib` pip package will silently install without the C library on some systems and then crash on first import. Test this in the Docker build explicitly.