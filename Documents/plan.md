# AI Trading Agent: Build Plan

## 0. Product Vision, Purpose & Target Audience

### What this product is

A fully autonomous AI trading system operating on the Kite AI blockchain. It runs two parallel intelligence systems simultaneously: a neural network making sub-100ms trade decisions from technical analysis and market signals, and an LLM continuously reading and interpreting news in the background looking for potential market stimulae. Neither waits for the other under normal conditions. When the LLM detects something significant, it interrupts the NN's next cycle. When it detects something severe, it stops the current cycle immediately. The result is a system with the speed and consistency of an ML model and the semantic intelligence of an LLM, without paying the latency cost of combining them serially.

### Who this is for

| Audience | How they use it | What they care about |
|---|---|---|
| **Retail crypto traders** | Deploy with their own API keys and a starting allocation | Autonomous operation, risk limits, no babysitting |
| **DeFi power users** | On-chain agent identity, Kite AI native execution | Verifiability, autonomous payments, trustless operation |
| **Quantitative hobbyists** | Extend signal pipeline with custom indicators or data feeds | Modularity, transparency, backtest capability |

### What makes it different

| Existing tools | Gap | This agent |
|---|---|---|
| TradingView alerts | Manual execution, no sizing logic | Fully autonomous execution |
| 3Commas / Pionex bots | Fixed rules, no learning | NN that learns from every trade |
| ChatGPT trading prompts | No execution, no memory, no learning | On-chain execution, persistent model, feedback loop |
| Single-model quant systems | Either fast OR intelligent | Both — parallel NN + LLM, non-blocking |

### Core principle

> **The NN is the trading brain. The LLM is the news interpreter. They run in parallel and communicate on events, not on every tick.**

---

## Table of Contents

1. [Core Architecture — Parallel NN + LLM](#1-core-architecture--parallel-nn--llm)
2. [Multi-Agent Design](#2-multi-agent-design)
3. [News Severity Tiers & Interrupt Protocol](#3-news-severity-tiers--interrupt-protocol)
4. [Neural Network — Design & Persistent Learning](#4-neural-network--design--persistent-learning)
5. [Signal Pipeline — Inputs to the NN](#5-signal-pipeline--inputs-to-the-nn)
6. [LLM News Agent](#6-llm-news-agent)
7. [Risk Manager](#7-risk-manager)
8. [Execution Engine](#8-execution-engine)
9. [Persistent Memory & State](#9-persistent-memory--state)
10. [Kite AI Chain Integration](#10-kite-ai-chain-integration)
11. [Build Timeline — 4 Weeks](#11-build-timeline--4-weeks)
12. [Common Problems & Debugging](#12-common-problems--debugging)
13. [Tech Stack](#13-tech-stack)
14. [Testing Strategy](#14-testing-strategy)

---

## 1. Core Architecture — Parallel NN + LLM

### The fundamental design

The LLM is not on the critical path for trade decisions. It runs in a completely separate process. The NN trading loop never waits for the LLM to finish. Communication between them happens via a shared priority queue, the LLM writes to it asynchronously, the NN reads from it at the start of each decision cycle.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            PROCESS 1 — NN TRADING CORE                      │
│   (CPU core 0–1, runs continuously, target cycle time: 50–200ms)            │
│                                                                              │
│  Market Data ──► Feature Engineering ──► NN Inference ──► Risk Check        │
│       ▲                                       │                │             │
│       │          ┌────────────────────────────┘                │             │
│       │          ▼                                             ▼             │
│  Feedback   Trade Decision                              Execution Engine     │
│  (online    (long/short/hold + size)                   (CCXT + Kite AI)     │
│   learning) │                                                                │
│             └──► Persistent Model Update (checkpoint every N trades)        │
└─────────────────────────────┬───────────────────────────────────────────────┘
                              │
               Shared news impact queue (Redis pub/sub)
               Written by Process 2, read by Process 1
               Three priority levels: NEUTRAL / SIGNIFICANT / SEVERE
                              │
┌─────────────────────────────▼───────────────────────────────────────────────┐
│                         PROCESS 2 — LLM NEWS AGENT                          │
│   (CPU core 2–3, runs continuously, decoupled from trading loop)            │
│                                                                              │
│  News Streams ──► Credibility Engine ──► LLM Impact Analysis                │
│                                              │                               │
│                         ┌────────────────────┼────────────────────┐         │
│                         ▼                    ▼                    ▼         │
│                    NEUTRAL              SIGNIFICANT             SEVERE       │
│                  (no action)      (queue next cycle)     (interrupt now)     │
└─────────────────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────────────────────┐
│                         PROCESS 3 — RISK MANAGER                            │
│   (Lightweight, synchronous gate — called by Process 1 before every order)  │
│   Hard limits: drawdown, position size, correlation, trades/hour             │
└─────────────────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────────────────────┐
│                         KITE CHAIN LAYER                                     │
│   Agent DID │ On-chain trade log │ Prediction log │ Autonomous payments      │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Why Python multiprocessing (not asyncio or threading)

Python's GIL (Global Interpreter Lock) prevents two threads from executing Python bytecode simultaneously. For CPU-bound work like NN inference and feature engineering, `threading` gives you concurrency but not true parallelism. `asyncio` is single-threaded by design. it interleaves I/O-bound tasks, but the NN and the LLM call would still share the same CPU core.

`multiprocessing` spawns genuine OS-level processes with separate memory spaces and separate GILs. Each process runs on its own CPU core. This is the right tool.

```python
from multiprocessing import Process, Queue
import torch

# Process 1: NN trading loop
def nn_trading_process(news_queue: Queue, model_path: str):
    model = TradingNN.load(model_path)      # load persistent model
    while True:
        features = build_feature_vector()   # technical + order book signals
        news_impact = drain_news_queue(news_queue)  # non-blocking read
        if news_impact.severity == "SEVERE":
            handle_severe_interrupt(news_impact)
            continue
        features = merge_news_features(features, news_impact)
        decision = model.infer(features)
        if risk_manager.approve(decision):
            execution_engine.execute(decision)
        model.online_update(decision, outcome)  # persistent learning

# Process 2: LLM news loop
def llm_news_process(news_queue: Queue):
    while True:
        article = news_pipeline.next()
        impact = llm_analyse(article)       # Claude API call — takes 200–800ms
        if impact.severity != "NEUTRAL":    # only write to queue if actionable
            news_queue.put(impact)

if __name__ == "__main__":
    q = Queue()
    p1 = Process(target=nn_trading_process, args=(q, "model.pt"))
    p2 = Process(target=llm_news_process, args=(q,))
    p1.start()
    p2.start()
    p1.join(); p2.join()
```

### Latency targets

| Operation | Target latency | How achieved |
|---|---|---|
| Feature engineering (one cycle) | < 20ms | pandas vectorised ops, pre-cached data |
| NN inference (forward pass) | < 10ms | PyTorch CPU inference on small model |
| Risk manager check | < 5ms | Pure Python dict lookups, no I/O |
| Order submission (market order) | < 100ms | CCXT + pre-authenticated session |
| Full NN trading cycle (end-to-end) | < 150ms | All of the above |
| LLM news analysis | 200–800ms | Runs in parallel — does not block trading |
| News queue read (by NN process) | < 1ms | Redis in-memory read |

---

## 2. Multi-Agent Design

### Single agent vs multiple agents

A single monolithic agent with all logic in one process is simpler to build but creates fatal coupling: if the LLM call hangs, trading stops. If the NN crashes, news processing stops. For a trading system, this is unacceptable.

The design uses **four specialised agents** (processes), each with a single responsibility:

| Agent | Process | Responsibility | Failure behaviour |
|---|---|---|---|
| **NN Trading Agent** | Process 1 | Feature engineering → NN inference → trade decision | Logs error, skips cycle, continues |
| **LLM News Agent** | Process 2 | News ingestion → credibility → LLM impact analysis → queue | Logs error, retries with backoff |
| **Risk Manager** | Inline in Process 1 | Hard gate before every order | Blocks trade, never crashes |
| **Execution Agent** | Subprocess of Process 1 | Order routing, fill confirmation, on-chain logging | Retries, logs failure to chain |


#### Agent communication

```
NN Trading Agent  ◄──── news_queue (Redis pub/sub) ────  LLM News Agent
       │
       ├──── risk_check() ────► Risk Manager (inline)
       │
       └──── place_order() ──► Execution Agent
                                      │
                                      └──► Kite Chain (async, fire-and-forget)
```

The news queue is the only inter-process communication channel. It is intentionally one-directional: the LLM News Agent only writes, the NN Trading Agent only reads. This eliminates deadlocks.

---

## 3. News Severity Tiers & Interrupt Protocol

This is the core innovation of the parallel architecture. The LLM doesn't just process news — it classifies every article into a severity tier that determines how the NN trading loop responds.

### The three tiers

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  TIER 1: NEUTRAL                                                             │
│                                                                              │
│  Examples: routine earnings beat, minor product announcement,                │
│            analyst upgrade, gradual regulatory clarification                 │
│                                                                              │
│  LLM output: { severity: "NEUTRAL", impact_delta: 0.0 }                     │
│                                                                              │
│  NN response: Nothing. Queue is not written to. NN cycle continues           │
│               with unchanged feature vector.                                 │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│  TIER 2: SIGNIFICANT                                                         │
│                                                                              │
│  Examples: Fed rate decision, exchange listing/delisting, large hack         │
│            ($50M+), significant regulatory action, major partnership         │
│                                                                              │
│  LLM output: {                                                               │
│    severity: "SIGNIFICANT",                                                  │
│    asset: "BTC-USD",                                                         │
│    direction: "down",                                                        │
│    magnitude_pct_low: 3.0,                                                   │
│    magnitude_pct_high: 8.0,                                                  │
│    confidence: 0.78,                                                         │
│    t_min_minutes: 5,                                                         │
│    t_max_minutes: 30                                                         │
│  }                                                                           │
│                                                                              │
│  NN response: Current cycle COMPLETES normally. At the START of the          │
│               next cycle, news impact vector is injected as additional       │
│               features into the NN input. NN re-infers with new context.     │
│               Position sizing also scales by news_confidence weight.         │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│  TIER 3: SEVERE                                                              │
│                                                                              │
│  Examples: Exchange collapse (FTX-level), war declaration, emergency         │
│            regulatory ban on crypto, Bitcoin protocol exploit,               │
│            stablecoin depeg ($10B+)                                          │
│                                                                              │
│  LLM output: {                                                               │
│    severity: "SEVERE",                                                       │
│    asset: "ALL",                                                             │
│    direction: "down",                                                        │
│    action: "CLOSE_ALL_POSITIONS",                                            │
│    confidence: 0.91                                                          │
│  }                                                                           │
│                                                                              │
│  NN response: IMMEDIATE interrupt. Current cycle is abandoned mid-flight.    │
│               Risk manager executes emergency protocol:                      │
│               close all open positions at market, halt new trades,           │
│               flag state as SUSPENDED. Operator alert sent.                  │
│               NN does NOT resume until human acknowledges OR                  │
│               confidence_of_recovery > threshold.                            │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Interrupt implementation

```python
# Shared atomic flag — multiprocessing.Value for safe cross-process reads
from multiprocessing import Value, Queue
import ctypes

class AgentState:
    def __init__(self):
        self.severe_flag = Value(ctypes.c_bool, False)   # atomic bool
        self.news_queue  = Queue()

# In NN Trading Agent — checked at the TOP of every cycle
def nn_cycle(state: AgentState, model: TradingNN):
    # SEVERE check — atomic read, near-zero cost
    if state.severe_flag.value:
        execute_emergency_protocol()
        wait_for_human_acknowledgement()
        return

    # Build features
    features = build_feature_vector()

    # SIGNIFICANT check — non-blocking queue drain
    while not state.news_queue.empty():
        news = state.news_queue.get_nowait()
        if news.severity == "SIGNIFICANT":
            features = inject_news_features(features, news)

    # NN inference + execution
    decision = model.infer(features)
    ...

# In LLM News Agent — sets the flag on SEVERE
def handle_severe(state: AgentState, impact: NewsImpact):
    state.severe_flag.value = True          # atomic write
    state.news_queue.put(impact)            # also queue the details
    logger.critical(f"SEVERE event: {impact}")
```

### Severity classification prompt (LLM)

```
You are a financial severity classifier for a live trading system.

Given a news article and its trust score, classify the severity of its market impact.

News: {news_text}
Trust score: {trust_score}
Affected assets: {asset_context}

Rules:
- NEUTRAL: routine news, minimal market impact, confidence < 0.5, or magnitude < 1%
- SIGNIFICANT: clear directional catalyst, confidence > 0.6, magnitude 1–10%,
               does NOT require immediate position closure
- SEVERE: systemic risk, exchange collapse, regulatory ban, magnitude > 10%,
          requires immediate position review

Respond in raw JSON only:
{
  "severity": "SIGNIFICANT",
  "asset": "BTC-USD",
  "direction": "down",
  "magnitude_pct_low": 3.0,
  "magnitude_pct_high": 8.0,
  "confidence": 0.78,
  "t_min_minutes": 5,
  "t_max_minutes": 30,
  "rationale": "..."
}
```

---

## 4. Neural Network — Design & Persistent Learning

### Architecture choice

| Model type | Accuracy | Training time | Inference speed | Online learning | Complexity |
|---|---|---|---|---|---|
| XGBoost / LightGBM | High | Minutes | < 5ms | Limited | Low |
| Feedforward NN (MLP) | Medium-high | Hours | < 5ms | Yes (SGD) | Low-medium |
| **LSTM (sequence model)** | **High** | **Hours** | **< 15ms** | **Yes** | **Medium** |
| Transformer (Temporal Fusion) | Very high | Days | 20–50ms | Limited | High |
| CNN on price series | High | Hours | < 10ms | Limited | Medium |
| Ensemble (XGBoost + LSTM) | Very high | Hours | < 20ms | Partial | High |

**Recommendation: LSTM with online fine-tuning, backed by XGBoost for regime-based routing.**

Why LSTM: financial signals are sequential — the RSI at t=5 means something different depending on what it was at t=1 through t=4. LSTMs are built for this. They are fast enough at inference time, well-understood, and support gradient-based online updates.

Why XGBoost alongside: XGBoost trains in minutes on labelled historical data, provides excellent feature importance for debugging, and acts as a fast fallback if the LSTM is mid-update. It also handles the regime classification task cleanly.

### Network architecture

```python
import torch
import torch.nn as nn

class TradingLSTM(nn.Module):
    """
    Input: feature vector of shape (sequence_length, n_features)
    Output: [long_prob, short_prob, hold_prob, position_size_logit]
    """
    def __init__(self, n_features: int = 64, hidden_size: int = 128,
                 num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=4, batch_first=True)
        self.fc_direction = nn.Linear(hidden_size, 3)       # long / short / hold
        self.fc_size      = nn.Linear(hidden_size, 1)       # position size [0, 1]
        self.dropout      = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        lstm_out, _ = self.lstm(x)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        pooled      = attn_out[:, -1, :]            # take last timestep
        pooled      = self.dropout(pooled)
        direction   = torch.softmax(self.fc_direction(pooled), dim=-1)
        size        = torch.sigmoid(self.fc_size(pooled))
        return direction, size
```

### Feature vector (n_features = 64)

The NN receives a window of the last N candles (e.g. 60 candles on the 5-minute chart = 5 hours of context), each represented as a feature vector. All features are normalised to [0, 1] or [-1, 1] before entering the network.

| Feature group | Features | Count |
|---|---|---|
| Price data | OHLCV (normalised to % change from open) | 5 |
| Moving averages | EMA 9, 21, 50, 200 (as % distance from price) | 4 |
| Momentum | RSI, MACD, MACD signal, Stoch RSI | 4 |
| Volatility | ATR (normalised), BB width, BB %B | 3 |
| Volume | Volume ratio (vs 20-period avg), OBV slope | 2 |
| Trend | ADX, +DI, -DI | 3 |
| Fibonacci | Distance to nearest Fib level, level type | 2 |
| S/R levels | Distance to nearest support, distance to nearest resistance | 2 |
| Chart patterns | One-hot encoded pattern flags (10 patterns) | 10 |
| Order book | Bid/ask imbalance, bid depth ratio, ask depth ratio | 3 |
| CVD | CVD slope (5-period), CVD divergence flag | 2 |
| Regime | One-hot regime state (6 states) | 6 |
| News impact | news_direction [-1,+1], news_magnitude [0,1], news_confidence [0,1], news_age_minutes | 4 |
| Macro | Fear/greed index, BTC dominance, funding rate, open interest change | 4 |
| Time | Hour of day (sin/cos encoded), day of week (sin/cos) | 4 |
| **Total** | | **62** |

News features default to `[0.0, 0.0, 0.0, 999.0]` (no news, age = very old) when no significant news is active. When the LLM injects a SIGNIFICANT event, these four values update and stay active for `t_max_minutes` before decaying back to zero.

### Persistent learning - the model never resets

This is a hard requirement. The model must load its previous state on every startup and continue learning from there.

```python
import torch
import os
from pathlib import Path

MODEL_PATH = Path("models/trading_lstm_latest.pt")
CHECKPOINT_DIR = Path("models/checkpoints/")

class PersistentTradingModel:
    def __init__(self):
        self.model = TradingLSTM(n_features=62)
        self.optimiser = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        self.replay_buffer = ReplayBuffer(max_size=10_000)
        self._load_or_initialise()

    def _load_or_initialise(self):
        if MODEL_PATH.exists():
            checkpoint = torch.load(MODEL_PATH)
            self.model.load_state_dict(checkpoint["model_state"])
            self.optimiser.load_state_dict(checkpoint["optimiser_state"])
            self.trade_count = checkpoint["trade_count"]
            self.cumulative_pnl = checkpoint["cumulative_pnl"]
            logger.info(f"Loaded model — {self.trade_count} trades, PnL: {self.cumulative_pnl:.2f}")
        else:
            # First ever run — initialise from scratch
            self.trade_count = 0
            self.cumulative_pnl = 0.0
            logger.info("No existing model found — initialising fresh")
            self._pretrain_on_historical_data()

    def checkpoint(self):
        """Called every 50 trades and on clean shutdown."""
        CHECKPOINT_DIR.mkdir(exist_ok=True)
        state = {
            "model_state":     self.model.state_dict(),
            "optimiser_state": self.optimiser.state_dict(),
            "trade_count":     self.trade_count,
            "cumulative_pnl":  self.cumulative_pnl,
            "timestamp":       datetime.utcnow().isoformat(),
        }
        torch.save(state, MODEL_PATH)
        # Rolling checkpoint — keep last 10
        torch.save(state, CHECKPOINT_DIR / f"checkpoint_{self.trade_count}.pt")
        logger.info(f"Model checkpointed at trade {self.trade_count}")

    def online_update(self, experience: TradeExperience):
        """
        Called after each trade closes. Uses the trade outcome as a training signal.
        experience contains: features_at_entry, decision_taken, actual_pnl
        """
        self.replay_buffer.add(experience)
        self.trade_count += 1

        # Mini-batch update every 10 trades
        if self.trade_count % 10 == 0 and len(self.replay_buffer) >= 32:
            batch = self.replay_buffer.sample(32)
            loss = self._compute_loss(batch)
            self.optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimiser.step()

        # Checkpoint every 50 trades
        if self.trade_count % 50 == 0:
            self.checkpoint()
```

### Initial training on historical data

On first startup (no saved model), the model pre-trains on historical OHLCV data before going live. This is not optional — a completely untrained model will lose money.

```python
def pretrain_on_historical_data(model: TradingLSTM, days: int = 30):
    """
    Labels: did price move > 0.5% in the next 15 minutes?
    Yes up   → label 0 (long)
    Yes down → label 1 (short)
    No       → label 2 (hold)
    """
    data = fetch_historical_ohlcv("BTC/USDT", "5m", days=days)
    features, labels = build_supervised_dataset(data)
    # Standard supervised training loop
    train_supervised(model, features, labels, epochs=50)
    model.checkpoint()
    logger.info(f"Pre-training complete on {len(labels)} samples")
```

### Learning approach comparison

| Approach | Speed | Stability | Hackathon feasibility |
|---|---|---|---|
| Full retrain from scratch every day | Slow | High | Poor — hours of training |
| Frozen model, no updates | Fast | High | Poor — never learns |
| Online SGD on every trade | Fast | Low (noisy) | Medium |
| **Mini-batch replay (every 10 trades)** | **Fast** | **High** | **Good — recommended** |
| Separate offline retraining thread | Medium | Very high | Good if time permits |

---

## 5. Signal Pipeline - Inputs to the NN

All signals are computed mathematically - no LLM, no ML at this stage. The output of this pipeline is the feature vector that enters the NN.

### 5.1 Market Data Feed

Foundation for all technical and order flow analysis.

| Provider | Latency | Cost | L2 book | Trades tape |
|---|---|---|---|---|
| **Binance WebSocket (direct)** | **< 50ms** | **Free** | **Yes** | **Yes** |
| Bybit WebSocket | < 50ms | Free | Yes | Yes |
| Alpaca (US stocks) | < 100ms | Free tier | Partial | Yes |
| CCXT REST (fallback) | 200–500ms | Free | Yes | No |

**Recommendation:** Binance WebSocket for L2 book + trade tape + OHLCV (all from one connection). Alpaca for US equities. CCXT as REST fallback only.

- **Build time:** 3–5 hours
- **Complexity:** 3/10
- **Vibe code recommendation:** ✅ **Vibe code** - verify WebSocket reconnection logic manually.

---

### 5.2 Technical Indicator Engine

All indicators computed on each cycle using pandas-ta / TA-Lib on rolling OHLCV windows. All outputs normalised before entering the feature vector.

#### Moving averages
| Signal | Computation | Feature encoding |
|---|---|---|
| EMA 9, 21, 50, 200 | Standard EMA | (EMA - price) / price → % distance |
| Golden/death cross | EMA50 vs EMA200 | Binary flag |
| VWAP | Volume-weighted | (price - VWAP) / VWAP |
| VWAP deviation band | ±1, ±2 std from VWAP | Normalised position within bands |

#### Momentum & oscillators
| Indicator | Parameters | Normalisation |
|---|---|---|
| RSI | Period 14 | Divide by 100 → [0, 1] |
| MACD | 12/26/9 | Normalise by ATR |
| Stochastic RSI | 14/3/3 | Divide by 100 → [0, 1] |
| ADX | Period 14 | Divide by 100 → [0, 1] |
| CCI | Period 20 | Clip to ±200, divide by 200 → [-1, 1] |

#### Fibonacci retracements
Swing high and low detected over a rolling lookback window. Key levels: 23.6%, 38.2%, 50%, 61.8%, 78.6%. Features: (price - nearest_level) / price, one-hot for which level.

#### Chart pattern recognition
TA-Lib pattern functions return confidence in [-100, 0, 100]. Converted to one-hot flags per pattern type. Patterns: bull flag, bear flag, pennant, ascending/descending triangle, head & shoulders, inverse H&S, double top/bottom, wedge, symmetrical triangle.

#### Support & resistance
Dynamic level detection via rolling pivot points and volume POC (VPVR). Features: distance to nearest support, distance to nearest resistance — both normalised by ATR.

#### Volume analysis
Volume ratio (current / 20-period SMA), OBV slope, absorption candle detection (large volume + small body).

- **Build time (all technical signals):** 10–14 hours
- **Complexity:** 5/10
- **Vibe code recommendation:** ✅ **Vibe code** for indicator calculations. ⚠️ **Hybrid** for normalisation logic — incorrect normalisation silently breaks NN training without obvious error messages.

---

### 5.3 Order Book & Microstructure Features

| Feature | Computation | Latency |
|---|---|---|
| Bid/ask imbalance | (bid_vol - ask_vol) / (bid_vol + ask_vol) | < 1ms |
| Bid depth ratio | bid_vol_top5 / bid_vol_top20 | < 1ms |
| Whale detection | order_size > 5× avg_order_size | < 1ms |
| Liquidity sweep flag | Wick pierced S/R, close reversed | < 2ms |
| CVD slope | Δ(cumulative_vol_delta) / N periods | < 2ms |
| CVD divergence | CVD direction ≠ price direction | < 1ms |

All computed directly from the L2 book and trade tape stream. No external API calls. Combined latency < 10ms.

- **Build time:** 6–8 hours
- **Complexity:** 5/10
- **Vibe code recommendation:** ⚠️ **Hybrid** — vibe the aggregation logic, manually validate CVD divergence and sweep detection against known examples from historical data.

---

### 5.4 Market Regime Detector

Runs inside the NN Trading Agent, outputs a one-hot regime vector injected into the feature set. The NN learns which strategies work in which regime — this is not hardcoded.

| Regime | Trigger conditions | One-hot index |
|---|---|---|
| Strong uptrend | ADX>25, price>EMA200, EMA50>EMA200 | [1,0,0,0,0,0] |
| Strong downtrend | ADX>25, price<EMA200, EMA50<EMA200 | [0,1,0,0,0,0] |
| Ranging | ADX<20, BB width <30th percentile | [0,0,1,0,0,0] |
| High volatility | ATR>95th percentile 30-day | [0,0,0,1,0,0] |
| News-driven | Active SIGNIFICANT/SEVERE event | [0,0,0,0,1,0] |
| Low liquidity | Volume<50th percentile + OTC hours | [0,0,0,0,0,1] |

- **Build time:** 4–5 hours
- **Complexity:** 4/10
- **Vibe code recommendation:** ✅ **Vibe code** indicator computation. ⚠️ **Manual** classification thresholds — these directly gate which signals the NN emphasises.

---

### 5.5 On-chain & Macro Features

Lower-frequency signals, updated every 15–60 minutes and cached. Not re-fetched on every cycle.

| Signal | Source | Update freq | Feature encoding |
|---|---|---|---|
| Crypto fear & greed | Alternative.me | Daily | Divide by 100 → [0, 1] |
| BTC dominance | CoinMarketCap | Hourly | Raw percentage / 100 |
| Funding rate (BTC perp) | Binance | Real-time | Clip to ±0.5%, normalise |
| Open interest Δ | Binance | Real-time | % change / 10 → [-1, 1] |
| Exchange net flow | On-chain | Hourly | Normalised Δ inflow |

- **Build time:** 3–4 hours
- **Complexity:** 3/10
- **Vibe code recommendation:** ✅ **Vibe code**

---

## 6. LLM News Agent

The LLM News Agent runs in Process 2, completely decoupled from trading. Its only output is writing to the shared news queue when it detects SIGNIFICANT or SEVERE events.

### 6.1 News Ingestion Pipeline

| Source | Latency | Coverage | Cost |
|---|---|---|---|
| RSS (Reuters, BBC, CoinDesk, Yahoo Finance) | 1–5 min | Authoritative, slow | Free |
| X (Twitter) filtered stream | Seconds | Fast, noisy | $100+/mo or basic tier |
| Telegram channel scraping (Telethon) | Seconds | Very fast for crypto | Free (unofficial) |
| CryptoPanic aggregator | ~1 min | Good crypto coverage | Free tier |

- **Build time:** 4–5 hours
- **Complexity:** 4/10
- **Vibe code recommendation:** ✅ **Vibe code** — plumbing only, low risk.

---

### 6.2 Credibility Engine

Assigns a trust score (0.0–1.0) before the LLM analyses the article. High-trust sources skip the LLM plausibility check and go straight to severity classification. This saves ~200ms per article from known reliable sources.

```
trust_score = 0.4 × source_base_score
            + 0.3 × cross_reference_score   # confirmed by other sources?
            + 0.2 × historical_accuracy      # was this source right before?
            + 0.1 × llm_plausibility_score   # only for unknown sources
```

```python
BASE_TRUST_SCORES = {
    "reuters.com":       0.95,   # fast lane — skip LLM plausibility
    "bbc.com":           0.93,   # fast lane
    "bloomberg.com":     0.92,   # fast lane
    "coindesk.com":      0.80,   # fast lane
    "yahoo_finance":     0.85,   # fast lane
    "cryptopanic":       0.60,   # LLM plausibility check required
    "telegram_unknown":  0.30,   # LLM plausibility check required
    "twitter_unknown":   0.25,   # LLM plausibility check required
}

FAST_LANE_THRESHOLD = 0.85  # above this, skip plausibility check
```

Trust scores update via EMA after each prediction is validated against actual price movement:
```python
def update_trust_score(current: float, new_score: float, alpha=0.08) -> float:
    return (1 - alpha) * current + alpha * new_score
```

- **Build time:** 8–10 hours
- **Complexity:** 6/10
- **Vibe code recommendation:** ⚠️ **Hybrid** — scaffold DB and scoring pipeline with AI. Manually review the EMA alpha and trust score formula weights.

---

### 6.3 LLM Impact Analysis

Called only when trust_score > MIN_TRUST_TO_ANALYSE (0.40) and article has not been seen before. Produces a structured severity classification (Section 3).

The LLM call is the slowest step in this process (200–800ms). This is acceptable because it runs in parallel with the NN and only writes to the queue when it has something actionable. A slow Claude response delays the queue write by up to 800ms — in most cases this is irrelevant because SIGNIFICANT events play out over minutes to hours.

**LLM model choice for news agent:**

| Model | Speed | Cost per call | Quality |
|---|---|---|---|
| claude-haiku | ~150ms | Very low | Good for classification |
| **claude-sonnet** | **~400ms** | **Medium** | **High — recommended** |
| claude-opus | ~800ms | High | Very high, overkill for classification |

**Recommendation:** Use `claude-haiku` for the credibility plausibility check (fast, cheap, simple). Use `claude-sonnet` for the full severity + impact analysis (higher stakes, needs accuracy).

- **Build time:** 6–8 hours
- **Complexity:** 6/10
- **Vibe code recommendation:** ⚠️ **Hybrid** — vibe the API call scaffolding. **Manually write both prompts** (plausibility check and severity classification). These directly determine what the NN sees as market-moving events.

---

### 6.4 Prediction Feedback Loop (LLM Agent)

After t_max_minutes from a SIGNIFICANT/SEVERE classification, the agent checks the actual price movement and scores the prediction. Updates:
- Source trust score (credibility engine)
- Internal LLM prompt calibration notes (stored in memory, periodically reviewed)
- On-chain prediction log

```python
def score_news_prediction(pred: NewsImpact, actual_move_pct: float,
                           actual_onset_minutes: int) -> float:
    direction_correct = (pred.direction == "down") == (actual_move_pct < 0)
    magnitude_score   = 1.0 - abs(actual_move_pct - pred.magnitude_pct_mid) / max(pred.magnitude_pct_mid, 0.01)
    timing_score      = 1.0 if pred.t_min <= actual_onset <= pred.t_max else 0.3
    return 0.4 * float(direction_correct) + 0.35 * max(0, magnitude_score) + 0.25 * timing_score
```

---

## 7. Risk Manager

Runs inline within the NN Trading Agent. Every trade decision from the NN passes through the risk manager before reaching the execution engine. It is synchronous and must complete in < 5ms — no I/O, no external calls.

```python
RISK_LIMITS = {
    "max_portfolio_drawdown_pct":  15.0,  # halt all trading if exceeded
    "max_daily_loss_pct":          5.0,   # reset next day
    "max_single_position_pct":     20.0,  # no single asset > 20% of portfolio
    "max_correlation_exposure":    0.70,  # limit on correlated positions
    "min_nn_confidence":           0.55,  # NN output probability threshold
    "max_trades_per_hour":         20,    # prevent runaway loops
    "min_position_usd":            15.0,  # below exchange minimum notional
    "max_position_usd":            5000.0,
}
```

Additional check: if a SEVERE flag is active, the risk manager blocks ALL new trades regardless of NN output. This is a redundant safety net on top of the interrupt protocol.

- **Build time:** 4–5 hours
- **Complexity:** 4/10
- **Vibe code recommendation:** ❌ **Manual code** — every check written by hand, every check unit tested. A bug here causes a margin call.

---

## 8. Execution Engine

Receives approved trade decisions, routes to exchange, manages order types, confirms fills, and triggers the on-chain log.

### Order type selection logic

```python
def select_order_type(decision: TradeDecision, regime: MarketRegime) -> OrderType:
    if regime.state == "news_driven_trend" and decision.urgency == "HIGH":
        return OrderType.MARKET       # speed > price in news-driven moves
    if decision.size_usd > 500:
        return OrderType.TWAP         # large orders: minimise market impact
    return OrderType.LIMIT            # default: better fills
```

### Order normalisation (prevent silent exchange failures)

```python
def normalise_order(exchange, symbol: str, size_usd: float) -> tuple[float, float]:
    market       = exchange.market(symbol)
    min_notional = market["limits"]["cost"]["min"] or 10.0
    price        = exchange.fetch_ticker(symbol)["last"]
    qty          = exchange.amount_to_precision(symbol, size_usd / price)
    if float(qty) * price < min_notional:
        raise ValueError(f"Order below minimum: ${float(qty)*price:.2f} < ${min_notional}")
    return float(qty), price
```

- **Build time:** 6–8 hours
- **Complexity:** 6/10
- **Vibe code recommendation:** ⚠️ **Hybrid** — vibe CCXT boilerplate. Manually review order submission, size rounding, and TWAP splitting logic.

---

## 9. Persistent Memory & State

### Storage architecture

| Store | Use | Why |
|---|---|---|
| **PostgreSQL** | Trade history, source trust scores, prediction log, model metadata | Relational, persistent, queryable |
| **Redis** | Inter-process news queue, real-time feature cache, market data cache | Sub-millisecond, pub/sub support |
| **ChromaDB** | RAG over historical similar events (for LLM context) | Vector similarity search |
| **PyTorch `.pt` file** | NN model weights, optimiser state, trade count | Native format, fast load |
| **Rolling checkpoint files** | Last 10 model checkpoints (safe rollback) | Recovery from catastrophic update |

### What persists across restarts

On every startup, the following are restored:

```python
class AgentState:
    # From PostgreSQL
    source_trust_scores:   dict[str, float]   # never reset
    trade_history:         list[Trade]          # full history
    prediction_log:        list[Prediction]     # for accuracy tracking
    cumulative_pnl:        float

    # From PyTorch file
    nn_model_weights:      dict                 # never reset
    nn_optimiser_state:    dict                 # never reset — preserves momentum
    total_trade_count:     int

    # From Redis (reconstructed if Redis restarted)
    feature_cache:         dict                 # rebuilt on first cycle
    active_news_impacts:   list                 # reconstructed from DB
```

### Model rollback capability

If the model's live performance degrades sharply (> 5% drawdown in 24h), the system can automatically roll back to the last checkpoint where performance was acceptable:

```python
def auto_rollback_if_degraded(model: PersistentTradingModel, 
                               recent_pnl: float, threshold: float = -0.05):
    if recent_pnl < threshold:
        checkpoints = sorted(CHECKPOINT_DIR.glob("*.pt"))
        if len(checkpoints) >= 2:
            prev = torch.load(checkpoints[-2])
            model.model.load_state_dict(prev["model_state"])
            logger.warning(f"Auto-rolled back to checkpoint at trade {prev['trade_count']}")
```

---

## 10. Kite AI Chain Integration

Every trade decision is signed with the agent's private key and logged on-chain. This creates a verifiable, tamper-proof performance record — critical for the hackathon demo.

```python
class KiteChainClient:
    async def log_trade_decision(self, decision: TradeDecision,
                                  features_snapshot: dict,
                                  nn_confidence: float) -> str:
        """Logs: signal inputs, NN confidence, decision, rationale. Returns tx hash."""

    async def log_news_prediction(self, prediction: NewsImpact) -> str:
        """Logs news classification for verifiable accuracy tracking."""

    async def log_prediction_outcome(self, prediction_id: str,
                                      actual_move: float) -> str:
        """Closes the prediction loop on-chain — prediction vs reality."""

    async def pay_for_data(self, provider_address: str, amount_wei: int) -> str:
        """Autonomous micropayment for data feeds — agent pays for its own tools."""

    async def get_agent_reputation(self) -> dict:
        """Returns on-chain stats: trade_count, win_rate, prediction_accuracy."""
```

On-chain logging is fire-and-forget from the NN Trading Agent's perspective. It calls `log_trade_decision()` after execution and does not wait for confirmation before continuing. The Kite chain interaction runs in the Execution Agent's own async loop.

- **Build time:** 6–10 hours
- **Complexity:** 7/10
- **Vibe code recommendation:** ⚠️ **Hybrid** — vibe the Web3.py boilerplate. **Manually review all transaction signing.** Never log or expose the private key.

---

## 11. Build Timeline — 4 Weeks

### Total estimated build time: ~120 hours

With a 4-week window, this is achievable comfortably as a 2-person team (~30 hours/person/week) or as a dedicated solo build (~30 hours/week).

### Week 1 — Foundation & data pipeline

**Goal: everything feeds data, nothing breaks, paper mode works end-to-end**

| Task | Hours | Owner | Vibe/Manual |
|---|---|---|---|
| PostgreSQL + Redis setup, schema design | 3h | Both | Vibe |
| Binance WebSocket (OHLCV + L2 book + trades) | 4h | Dev 1 | Vibe |
| Alpaca stock feed | 2h | Dev 1 | Vibe |
| All technical indicators (pandas-ta + TA-Lib) | 6h | Dev 1 | Vibe |
| Feature vector builder + normalisation | 4h | Dev 1 | **Hybrid** |
| News RSS + Telegram ingestion | 4h | Dev 2 | Vibe |
| Credibility engine (source DB + base scores) | 4h | Dev 2 | Hybrid |
| Multiprocessing scaffold (Process 1 + 2 skeleton) | 3h | Both | **Manual** |
| Redis pub/sub queue between processes | 2h | Both | Hybrid |
| Paper trading mode stub | 3h | Dev 1 | Vibe |
| **Week 1 milestone: feature vector prints every 5 seconds, news ingested, two processes running** | | | |

---

### Week 2 — Neural network + order book signals

**Goal: NN makes real decisions (in paper mode), order book signals feeding in**

| Task | Hours | Owner | Vibe/Manual |
|---|---|---|---|
| LSTM architecture implementation | 4h | Dev 1 | **Manual** |
| Historical data fetch + supervised labelling pipeline | 5h | Dev 1 | Hybrid |
| Initial pre-training on 30 days of BTC data | 3h | Dev 1 | Hybrid |
| Persistent model save/load (checkpoint system) | 3h | Dev 1 | **Manual** |
| Online learning (mini-batch replay, every 10 trades) | 4h | Dev 1 | **Manual** |
| Order book microstructure signals (imbalance, CVD, sweep) | 6h | Dev 2 | Hybrid |
| Market regime detector | 4h | Dev 2 | Hybrid |
| Fibonacci + S/R level detection | 5h | Dev 2 | Hybrid |
| Chart pattern recognition (TA-Lib flags + 5 key patterns) | 5h | Dev 2 | Hybrid |
| **Week 2 milestone: NN making paper trades based on full feature vector, model checkpointing** | | | |

---

### Week 3 — LLM news agent + risk manager + execution

**Goal: full parallel system running, LLM interrupts working, risk gates working**

| Task | Hours | Owner | Vibe/Manual |
|---|---|---|---|
| LLM severity classifier (prompt + parser) | 4h | Dev 2 | **Manual** (prompt) |
| LLM impact predictor (full structured output) | 4h | Dev 2 | **Manual** (prompt) |
| News feature injection into NN feature vector | 3h | Dev 1 | Hybrid |
| Severity tier interrupt protocol (SIGNIFICANT + SEVERE) | 4h | Both | **Manual** |
| Atomic SEVERE flag (multiprocessing.Value) | 2h | Dev 1 | Manual |
| Risk manager (all hard limits + SEVERE gate) | 5h | Dev 1 | **Manual** |
| Execution engine (CCXT + order types + normalisation) | 6h | Dev 2 | Hybrid |
| TWAP order splitting | 3h | Dev 2 | Hybrid |
| Prediction feedback loop (outcome scoring + trust update) | 4h | Dev 2 | Hybrid |
| Model auto-rollback on performance degradation | 2h | Dev 1 | Hybrid |
| **Week 3 milestone: full parallel system live in paper mode, LLM interrupts tested** | | | |

---

### Week 4 — Kite chain + testing + demo

**Goal: on-chain, hardened, demo-ready**

| Task | Hours | Owner | Vibe/Manual |
|---|---|---|---|
| Kite AI chain integration (DID, trade log, payments) | 8h | Dev 2 | Hybrid |
| On-chain prediction log + outcome closure | 3h | Dev 2 | Hybrid |
| Monitoring dashboard (PnL, positions, NN confidence, queue) | 4h | Dev 1 | Vibe |
| 48h+ continuous paper trading test | ongoing | Both | — |
| Unit tests: risk manager (all limits) | 3h | Both | Manual |
| Unit tests: interrupt protocol (all three tiers) | 2h | Both | Manual |
| Unit tests: feature normalisation | 2h | Dev 1 | Hybrid |
| Unit tests: NN checkpoint save/load | 2h | Dev 1 | Manual |
| XGBoost fallback model (fast backup if LSTM unavailable) | 4h | Dev 1 | Vibe |
| Demo preparation + on-chain proof-of-performance | 3h | Both | — |
| Buffer / polish / bug fixes | 8h | Both | — |
| **Week 4 milestone: agent running live (small capital or testnet), on-chain logs visible, demo ready** | | | |

---

## 12. Common Problems & Debugging

### 12.1 Processes cannot communicate — queue deadlock

**Symptom:** NN Trading Agent stops reading from news queue. LLM News Agent's `queue.put()` blocks indefinitely.  
**Root cause:** `multiprocessing.Queue` has a maximum size. If the NN process stops draining it, the LLM process blocks on `put()`.  
**Fix:**
```python
# Always use maxsize + put_nowait to prevent blocking
news_queue = Queue(maxsize=100)

# In LLM News Agent — never block
try:
    news_queue.put_nowait(impact)
except Full:
    logger.warning("News queue full — dropping item. Is NN process alive?")
    # Also check if NN process is alive
    if not nn_process.is_alive():
        logger.critical("NN process died — restarting")
        nn_process = Process(target=nn_trading_process, args=(news_queue,))
        nn_process.start()
```
**Prevention:** Process health monitoring — each process writes a heartbeat timestamp to Redis every 5 seconds. A watchdog thread checks them.

---

### 12.2 NN model checkpoint corrupted on unclean shutdown

**Symptom:** `torch.load()` raises an exception on startup. Model cannot be loaded.  
**Root cause:** System killed mid-write to `.pt` file — partial write leaves a corrupted file.  
**Fix:** Always write to a temp file and rename atomically:
```python
def safe_checkpoint(state: dict, path: Path):
    tmp_path = path.with_suffix(".tmp")
    torch.save(state, tmp_path)
    tmp_path.replace(path)      # atomic on POSIX systems
```
**Prevention:** Keep rolling checkpoints. If `latest.pt` is corrupted, fall back to `checkpoints/checkpoint_N.pt`.

---

### 12.3 Feature normalisation breaks after market microstructure shift

**Symptom:** NN starts producing extreme confidence scores (>0.98 or <0.02) for all decisions. Trades become erratic.  
**Root cause:** Normalisation used fixed min/max from training data. Live data outside that range clips or saturates features.  
**Fix:** Use rolling normalisation with a long window (1000 candles):
```python
class RollingNormaliser:
    def __init__(self, window=1000):
        self.buffer = deque(maxlen=window)
    
    def normalise(self, value: float) -> float:
        self.buffer.append(value)
        if len(self.buffer) < 10:
            return 0.5
        arr = np.array(self.buffer)
        return float(np.clip((value - arr.mean()) / (arr.std() + 1e-8), -3, 3) / 3 * 0.5 + 0.5)
```

---

### 12.4 SEVERE interrupt triggers on false positive

**Symptom:** Agent closes all positions and halts trading on a news item that turns out to be incorrect or misclassified.  
**Root cause:** LLM misclassifies a SIGNIFICANT event as SEVERE. Trust score of source was inflated.  
**Fix:** Add a confirmation delay for SEVERE classification from non-ultra-high-trust sources:
```python
def handle_severe_candidate(impact: NewsImpact, trust_score: float):
    if trust_score >= 0.90:
        set_severe_flag_immediately(impact)    # Reuters/Bloomberg — act now
    else:
        # Queue as SEVERE_PENDING — confirm after 30 seconds
        schedule_confirmation(impact, delay_seconds=30)

def confirm_severe(impact: NewsImpact):
    # Re-check: has another trusted source confirmed? Has price moved?
    corroborated = check_cross_reference(impact)
    price_moved  = check_price_impact(impact.asset, threshold_pct=2.0)
    if corroborated or price_moved:
        set_severe_flag_immediately(impact)
    else:
        # Downgrade to SIGNIFICANT
        downgrade_to_significant(impact)
```

---

### 12.5 Online learning causes model to overfit to recent trades

**Symptom:** Model performs well for a few hours then degrades. It starts chasing very short-term patterns that don't generalise.  
**Root cause:** Mini-batch replay buffer is too small, or learning rate too high. Recent trades overwhelm historical patterns.  
**Fix:**
```python
# Use a large replay buffer with prioritised sampling
# Prioritised = sample by TD-error (replay experiences the model was most wrong about)
replay_buffer = PrioritisedReplayBuffer(max_size=10_000, alpha=0.6)

# Reduce learning rate progressively
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=1000)

# Add L2 regularisation to prevent weight drift
optimiser = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
```
**Prevention:** Track rolling 24h Sharpe ratio. If it drops > 30% from the peak in the last 48h, pause online learning and alert.

---

### 12.6 Chart pattern false positives flood NN feature vector

**Symptom:** Pattern flags are nearly always 1.0 — NN can't learn meaningful signal from them.  
**Root cause:** Pattern detection thresholds too loose, or not regime-gated.  
**Fix:**
```python
def validate_pattern(pattern: PatternSignal, regime: MarketRegime) -> bool:
    if pattern.pattern in ["bull_flag", "bear_flag", "pennant"]:
        if regime.adx < 20:             # continuation patterns need a trend
            return False
    if pattern.pattern in ["head_and_shoulders", "double_top"]:
        if pattern.rsi_at_detection < 65:   # reversal needs overbought
            return False
    if "flag" in pattern.pattern:
        if pattern.pole_size_pct < 3.0:     # pole must be meaningful
            return False
    return True
```

---

### 12.7 LLM JSON output is malformed

**Symptom:** `json.JSONDecodeError` from news agent — impact classifications dropped.  
**Fix:**
```python
def extract_json(raw: str) -> dict:
    raw = re.sub(r"```json|```", "", raw).strip()
    start = raw.index("{")
    end   = raw.rindex("}") + 1
    return json.loads(raw[start:end])
```
**Prevention:** `temperature=0` for all structured output calls. Prompt: "Respond with raw JSON only. No preamble, no markdown, no explanation."

---

### 12.8 Exchange order fails silently

**Symptom:** `place_order()` returns no error but order doesn't appear on exchange.  
**Root cause:** Below minimum notional, wrong decimal precision, or authentication session expired.  
**Fix:** Use `normalise_order()` (Section 8) before every order. Verify response contains an order ID, not just `{}`.

---

### 12.9 Redis queue overwhelmed during high-news periods

**Symptom:** Inter-process queue fills during a major market event. LLM agent starts dropping news items.  
**Root cause:** Multiple SIGNIFICANT events queued faster than the NN agent processes them.  
**Fix:** Priority queue — SEVERE always jumps to front:
```python
import heapq

class PriorityNewsQueue:
    SEVERITY_PRIORITY = {"SEVERE": 0, "SIGNIFICANT": 1, "NEUTRAL": 2}
    
    def put(self, impact: NewsImpact):
        priority = self.SEVERITY_PRIORITY[impact.severity]
        heapq.heappush(self._queue, (priority, impact))
    
    def get(self) -> NewsImpact | None:
        if self._queue:
            _, impact = heapq.heappop(self._queue)
            return impact
        return None
```

---

### 12.10 Kite AI transaction timeout on high gas

**Fix:** Gas bumping with retry:
```python
async def send_with_retry(web3, tx, max_attempts=3):
    for attempt in range(max_attempts):
        try:
            gas_price = await web3.eth.gas_price
            tx["gasPrice"] = int(gas_price * (1.1 + attempt * 0.2))
            signed = web3.eth.account.sign_transaction(tx, PRIVATE_KEY)
            return await web3.eth.send_raw_transaction(signed.rawTransaction)
        except Exception as e:
            if attempt == max_attempts - 1:
                raise
            await asyncio.sleep(2 ** attempt)
```

---

## 13. Tech Stack

### Core
- **Language:** Python 3.11+
- **Parallelism:** `multiprocessing` (OS-level processes, genuine parallel CPU usage)
- **Async (within each process):** `asyncio` + `aiohttp`
- **NN framework:** PyTorch 2.x
- **Gradient boosting (fallback + regime):** XGBoost / LightGBM
- **LLM:** Anthropic Claude API — `claude-haiku` for plausibility checks, `claude-sonnet` for severity + impact classification

### Data & signals
- **Market data:** Binance WebSocket (direct, L2 + trades + OHLCV), Alpaca (stocks)
- **REST fallback:** CCXT unified API
- **Technical indicators:** pandas-ta, TA-Lib (C-backed, faster for batch)
- **Portfolio optimisation:** PyPortfolioOpt
- **News:** feedparser (RSS), tweepy (X API), Telethon (Telegram)

### Storage
- **Primary DB:** PostgreSQL (via SQLAlchemy async) — trade history, trust scores, predictions
- **Inter-process queue:** Redis (pub/sub, priority queue) — news impact between processes
- **Feature cache:** Redis — latest normalised feature vector, macro data cache
- **Vector store:** ChromaDB — RAG for historical similar events (LLM context)
- **Model weights:** PyTorch `.pt` files with rolling checkpoints

### Infrastructure
- **On-chain:** Web3.py (Kite AI chain)
- **Exchange execution:** CCXT unified API
- **Monitoring:** FastAPI + simple HTML dashboard (or Grafana + Prometheus for the advanced path)
- **Deployment:** Docker + docker-compose, one container per process

### Development
- **Testing:** pytest + pytest-asyncio
- **Type checking:** mypy (mandatory for NN module, risk manager, execution engine)
- **Logging:** structlog (JSON structured logs — essential for replaying agent reasoning)
- **Profiling:** py-spy (for identifying latency bottlenecks in the feature engineering pipeline)

---

## 14. Testing Strategy

### Unit tests — write these manually, non-negotiable

- Risk manager: every limit individually (drawdown, daily loss, position size, min confidence, SEVERE gate)
- NN checkpoint: save → corrupt → auto-recover from previous checkpoint
- Feature normalisation: values inside range, values outside range (clip behaviour)
- Interrupt protocol: NEUTRAL (no action), SIGNIFICANT (inject features), SEVERE (halt immediately)
- News queue priority: SEVERE always dequeued before SIGNIFICANT
- Pattern validation: each pattern type rejected in wrong regime
- Trust score EMA: convergence, alpha sensitivity, boundary behaviour
- LLM JSON extraction: preamble, fences, truncated JSON, empty string

### Integration tests — hybrid vibe is fine

- Full cycle: market data → features → NN inference → risk check → paper execution
- News cycle: article → credibility → LLM → queue write → NN feature injection
- Persistent learning: close trade → online update → checkpoint → restart → correct weights loaded
- SEVERE interrupt: inject mock SEVERE event → verify all positions closed → verify SUSPENDED state

### Paper trading — mandatory before any live capital

- Run continuously for **minimum 72 hours** (longer than previous plan, given the NN's online learning needs time to demonstrate stability)
- Monitor: NN confidence score distribution (should be spread, not concentrated at extremes), trade frequency (< 20/hour), position sizes (< 20% per asset), 24h rolling Sharpe
- Check: model checkpoint files being written every 50 trades
- Check: SEVERE flag resets correctly after acknowledgement

### Red flags — do not go live if any of these occur

- NN confidence > 0.97 on more than 5% of decisions (model overfit or feature saturation)
- Model weight L2-norm growing each checkpoint (gradient explosion — learning rate too high)
- Any unhandled exception propagating from the risk manager
- SEVERE interrupt not triggering within 2 seconds of mock injection
- NN and LLM processes not both showing alive heartbeats after 1 hour of paper mode
- Trust scores for all sources converging to extremes within 48 hours
