# The Wolf of Wall Street — Project Retrospective (`meta.md`)

> A comprehensive, honest account of what this project set out to do, every major approach
> we tried, where each one broke, how we refined, and the numbers behind every verdict.
> Written 2026-07-04. This is the "why" document — the code is the "what". If you read one
> file to understand the *thinking* behind the system, read this one.

---

## Table of contents

1. [The goal](#1-the-goal)
2. [The first approach: a neural network that predicts price](#2-the-first-approach)
3. [The reckoning: how we discovered we were wrong](#3-the-reckoning)
4. [The pivot: from prediction to a portfolio of edges](#4-the-pivot)
5. [The strategy library — every strategy, every number](#5-the-strategy-library)
6. [Feature engineering — the signal audit and what carries information](#6-feature-engineering)
7. [Model architectures compared — a theoretical standpoint](#7-model-architectures)
8. [The risk machinery — the one thing that consistently worked](#8-the-risk-machinery)
9. [The ML meta-layer — how ML earned its (small) place back](#9-the-ml-meta-layer)
10. [The graveyard — rejected ideas and why](#10-the-graveyard)
11. [Papers we read and how each shaped a decision](#11-papers)
12. [The master results table](#12-master-results-table)
13. [Current state and what's next](#13-current-state)
14. [Meta-lessons](#14-meta-lessons)
15. [Running decision log](#15-running-decision-log) — *every new consideration/edit gets an entry here*

---

## 1. The goal

**Objective:** maximize realized profit per unit time (CAGR) from an autonomous trading system,
subject to a hard risk constraint — the system's Sharpe ratio must comfortably exceed the market's
(the S&P 500, ~0.5–0.6 over long horizons).

The governing identity that ended up organizing everything:

```
CAGR  ≈  Sharpe × σ_book        (for volatility well below the Kelly ceiling)
```

Two independent levers. **Sharpe** comes from skill and diversification; **σ_book** (the volatility
we choose to run) converts that skill into return via leverage/sizing. Much of the project was a
long education in which lever is actually movable — and the punchline, established empirically, is
that raising Sharpe past a floor is brutally hard at retail scale, while the volatility lever and
the risk machinery are where the reliable gains live.

The system was always multi-component: a PyTorch model, an LLM news pipeline, a FastAPI backend,
a Next.js dashboard, crypto (Binance) + US equity (Alpaca) data, and paper/live execution. But the
*intellectual* core — "what do we actually trade on?" — is what this document tracks.

---

## 2. The first approach

**Premise:** train a neural network to predict short-term price direction, then trade the prediction.

This is the intuitive starting point and it is what the original codebase was built around. The
model that carried it (`backend/agents/improved_model.py::ImprovedTradingLSTM`) was, for its class,
genuinely well-engineered:

- **Input:** a 90-dimensional feature vector per 5-minute bar × a 60-bar lookback (5 hours), covering
  price action, moving averages, momentum (RSI/MACD/StochRSI/ADX), volatility (ATR/Bollinger),
  volume/OBV, Fibonacci levels, 10 candlestick patterns, 8 order-book microstructure slots, a 6-class
  regime one-hot, 4 news scalars, 4 macro scalars, cyclical time encodings, an 8-dim higher-timeframe
  (1h/4h) block, a 16-dim semantic news embedding, and 4 earnings-calendar features
  (`backend/signals/feature_spec.py`).
- **Trunk:** symbol embedding → LSTM (later switchable to GRU or a causal TCN) → LayerNorm →
  additive attention → shared MLP.
- **Heads:** one direction head per horizon (H+3 / H+12 / H+48 bars = 15m / 1h / 4h), a position-size
  head, learned ATR-multiple stop-loss / take-profit / trailing heads (trained online via RL), a
  next-K OHLC log-return head (for a "predicted price" chart), and a learnable temperature for
  calibration.
- **Training:** triple-barrier vol-scaled labels (López de Prado style), class-weighted focal loss,
  PnL-magnitude weighting, calendar recency weighting, label smoothing, per-horizon loss weights,
  a per-symbol chronological train/val/**untouched-test** split with embargoes, and cost-aware
  net-alpha checkpoint selection.
- **Online RL:** Advantage-Weighted Regression on a replay buffer, Sortino-shaped rewards, MC-dropout
  uncertainty, fractional-Kelly sizing on the lower credible bound of the edge.

It is worth stating plainly: **none of this was incompetent.** The engineering was sound and much of
it survives. The problem was upstream of all of it — the *target*.

Alongside the model sat an ambitious "predicted-price chart" program (Phases 17–18): MC-dropout
predictive bands, an OHLC-delta head, and a deferred design for an autoregressive candle transformer
with per-channel quantile tokenisation (`trading-agent/potential_implementation.md`). That design was
correctly gated behind an empirical trigger (build it only if the cheaper hybrid's 5-candle MAE was
barely better than a random walk) and never triggered.

---

## 3. The reckoning

Three independent discoveries, in roughly this order, dismantled the "predict direction" premise.

### 3a. Train/serve skew — the model that graded itself on a different exam

The offline `pretrain.py` and the live `technical.py` computed features *differently*, and offline
z-scored the inputs while live did not. An 8-slot vs 10-slot order-book block and a regime one-hot at
different indices meant **offline-trained weights were being fed subtly different vectors live**. The
model's reported training performance was real; its live performance was measuring something else.

Fix (Phase 0): a single canonical feature pipeline (`backend/features/pipeline.py`, `feature_spec.py`
as the single source of truth) with a byte-for-byte live==offline parity test. This was necessary
plumbing — but fixing it only made the *real* verdict legible.

### 3b. The signal audit — direction is ~0.53 AUC

With features unified, `scripts/signal_audit.py` measured the actual information content. The verdict
was unambiguous and repeated across every subsequent test:

- Single-name **directional AUC ≈ 0.53–0.54** at the 1h–4h crypto horizons — barely above a coin flip.
- **No learnable intraday edge in single-name US stocks** at 5m at all.
- **~45% of the 90 features are dead** (|IC| < 0.003) — exactly the blocks that were structurally
  zero in the offline data: order book (no historical L2), news scalars + embeddings (no bar-aligned
  historical corpus), macro, earnings, the constant spread. These were later pruned structurally into
  the 49-feature "active" set.

An information coefficient of ~0.02–0.05 (which is what 0.53 AUC corresponds to) is *not nothing* —
but by Grinold's fundamental law, `IR ≈ IC·√breadth`, a tiny IC only becomes a real information ratio
through **breadth**, not through squeezing the predictor. This reframing became the project's compass.

### 3c. The TA cookbook and Fourier — 0 for 588

To rule out "we just picked the wrong indicators", `scripts/strategy_zoo_research.py` tested 12
distinct classical TA families (MA cross, MACD, Supertrend, Ichimoku, PSAR, ADX-trend, Donchian,
ATR-channel, ROC-momentum, RSI-MR, Bollinger-MR, OBV-trend) across **49 sector-diverse assets**,
cost-aware, scored with the **Deflated Sharpe Ratio** (Bailey & López de Prado — which discounts a
backtest Sharpe by the expected best-of-N-trials under the null).

**Result: 0 of 588 (strategy × asset) trials survive DSR > 0.95.** Zero in every sector including
health (0/120) and finance (0/120). Per-strategy median Sharpe after costs was mostly *negative*
(MACD −0.20, Ichimoku −0.28, ATR-channel −0.38, OBV −0.43). The few positive raw performers were
crypto trend/momentum names — but their DSR (~0.29–0.33) placed them squarely inside the luckiest-of-588
envelope, and they were the trend-on-trending-crypto pattern we would later recognize as **beta, not
alpha**. A Fourier dominant-cycle test gave mean |IC| = 0.020, signed +0.004 — the noise floor.
Markets are not stationary-cyclic; a Fourier low-pass just re-derives a moving average.

**Conclusion of the reckoning:** predicting absolute price direction from public OHLC data at retail
scale and cost is not a solved problem waiting for a better model — it is a *competed-away* problem.
The engineering was fine; the target was wrong.

---

## 4. The pivot

The owner-approved reframe (plan: `swirling-seeking-wolf.md`) was decisive: **demote the
"NN-predicts-direction" model from the primary alpha source to at most one A/B candidate, and rebuild
the core as a portfolio of economically-motivated, largely-uncorrelated rule-based strategies, sized
by risk, with ML repurposed as a meta-layer** (signal-quality filtering, regime, sizing) — never
again to predict raw direction.

This is what a real systematic shop does, and the fundamental law says why: with IC structurally
small, the path to Sharpe is breadth across *uncorrelated* bets plus disciplined risk management. The
target outcome was reframed from "a big prediction number" to "measurable regression alpha with
controlled beta and drawdown, and a portfolio Sharpe driven by diversification."

Everything strategy-agnostic was kept (the feature pipeline, the cost-aware backtest engine with
α/β decomposition and Deflated Sharpe, the walk-forward harness, the data feeds, the execution
primitives, the risk manager). The "NN predicts 5m direction" decision path was retired as primary.

What followed was ~six weeks of building and *measuring* strategies against a non-negotiable gate:
walk-forward, per-fold Sharpe, regression α/β, Deflated Sharpe > 0.95, realistic costs, and one
untouched hold-out. The rest of this document is what that gauntlet produced.

---

## 5. The strategy library

Every strategy was built as a `Strategy` subclass emitting a causal per-bar position path, then run
through the portfolio backtester. Here is each one, in the order tested, with the numbers.

### 5.1 Time-series momentum (crypto) — `ts_momentum.py`
Donchian breakout + EMA trend filter + Chandelier (ATR) trailing stop, the Moskowitz–Ooi–Pedersen
edge. **1h was a graveyard:** Sharpe −0.15 to −0.83, maxDD −45.8%, DSR 0.001 — the 2022–26 crypto
chop is death for 1h breakout. **The fix that salvaged it: 4h bars + an ADX ≥ 25 regime gate + a short
leg.** Sharpe +0.55 to +0.94, β ≈ 0, maxDD −13 to −15%, 3/4 folds positive, stable across the
parameter neighborhood. But **DSR only 0.14–0.30** (short 2.7-year sample + honest trial penalty) —
never promotable standalone. Verdict: a **low-confidence β≈0 diversifier**, valuable precisely because
the rest of the book is long-biased.

### 5.2 Cross-sectional momentum — `cross_sectional.py`
Rank a universe, long top-k / short bottom-k in equal dollars (Σw = 0, β ≈ 0 by construction). This
one taught the sharpest lesson about **overfitting to a regime**:
- **Stocks (8 semis, 1h, ~1.4y):** Sharpe **+4.65**, +175% return, 4/4 folds, maxDD −5.5%, α +0.20.
  Spectacular — and a **mirage**: DSR only 0.374, a single AI-dispersion regime, and when the universe
  was broadened to multiple sectors it **collapsed to −0.09**. The standout was regime luck.
- **Crypto:** thin-positive (+0.17) on a small universe, then **−0.39 over 20 names / 2.7y (DSR 0.002)**
  — crypto names are too mutually correlated for rank-dispersion to work.

### 5.3 Statistical arbitrage — `stat_arb.py`
Engle-Granger cointegration on a train prefix (no look-ahead), rolling-OLS hedge ratio, z-scored
spread, Ornstein-Uhlenbeck half-life gate, market-neutral. **Logic verified correct** (profits +0.86
on synthetic OU data) but **the regime was wrong**: on real semis 2023–25 it FAILS both naive
(DSR 0.000, α −43.5%/yr) and with the OU gate (α −26.7%). The 2022–25 AI boom made semis a
*dispersion/momentum* regime — poison for spread reversion. Its failure was itself evidence that
cross-sectional *momentum* (not reversion) was the live stock regime. Shelved.

### 5.4 Mean reversion (crypto) — `mean_reversion.py`
Fade close z-score extremes, gated to ranging (low-ADX) regimes, exit when a trend emerges.
**Crypto verdict: −90% drawdown, DSR 0.** Fading a momentum asset loses. It also exposed a latent bug:
naive vol-targeting *amplifies* a loser (levered 2× into a losing streak), which motivated the
drawdown de-gearing overlay. Shelved.

### 5.5 Funding carry (crypto) — `funding_carry.py`
Delta-neutral perp funding capture (receive the crowded side). **The only genuinely market-neutral
survivor by DSR (0.690, β ≈ 0) — but economically tiny (~+0.2%/yr** after the cost of side-flips).
Real, uncorrelated, and too small to matter alone.

### 5.6 Daily factor research — `factor_research.py`
On the real 502-name S&P 500 panel, 21 years from 2005 (survivorship caveat noted):
- **reversal_1m:** Sharpe +0.39, 5/6 folds, DSR 0.350 — the standout, strongest *with breadth*.
- **momentum_12_1:** +0.31, DSR 0.234, 3/6 — weakly positive.
- **low_vol:** −0.46, 0/6, DSR 0.000 — genuinely dead (price-only, growth regime).
- **Combined {momentum, reversal}, scale-fixed:** 6/6 folds positive, Sharpe **+0.46**, maxDD −18.5%,
  corr 0.37 (diversification *worked* — combined beat either alone). **BUT** the α/β decomposition gut
  it: the books leak +0.14…+0.25 market beta, and once regressed out, **reg alpha is ~0 to negative**.
  The entire +2.5%/yr return was incidental market beta — a 0.20-beta tilt whose Sharpe (~0.5) *beats*
  the book's 0.46. **Verdict: price-only daily factors have no positive market-neutral alpha on liquid
  large-caps over 21 years.** (A scale bug — double-normalisation of dollar-neutral weights — was found
  and fixed here; Sharpe/DSR are scale-invariant so verdicts held, but it had silently killed the risk
  overlays, a lesson in itself.)

### 5.7 Fundamental factors — `fundamental_factor_research.py`
SEC EDGAR XBRL point-in-time fundamentals (strict no-look-ahead: every figure used only on/after its
filing date). Value (E/P + B/P) and quality (gross-profitability + ROE):
- **value:** reg α **+0.1%/yr**, β +0.06.
- **quality:** reg α **+0.3%/yr**, β +0.02 — genuinely market-neutral.
- These are the **first genuine near-zero-beta positive-alpha factors in the entire project** —
  qualitatively unlike the price-only factors (which were beta in disguise).
- **But economically marginal:** the honest pure-alpha number (value+quality) is **+0.2%/yr at β 0.04**,
  and the recent OOS folds (2022–26, the value "winter") are the weakest. Real, clean, too small to be
  a money cannon.

### 5.8 Managed beta — `managed_beta.py` — **THE decisive win**
After proving market-neutral alpha is ~0 everywhere at retail scale, the owner chose the honest pivot:
**stop pretending to pick winners; harvest the long-run risk premium of trending assets, but deliver
it better than buy-and-hold.** A Faber 200-day trend filter (invested above trend, in cash below —
crashes happen in downtrends), dead-band whipsaw guard, combined with portfolio vol-targeting +
drawdown de-gearing + a VIX/correlation macro overlay.

**Result (daily 2010–2025, curated 7-asset SPY/QQQ/TQQQ/TLT/GLD/BTC/ETH):**

| Metric | Managed book | Buy & Hold |
|---|---|---|
| Sharpe | **1.10** | 0.89 |
| max drawdown | **−27%** | −45% |
| Calmar | **0.62** | 0.37 |
| CAGR | 16.8% | ~16.8% |
| alpha vs B&H | **+7.4%/yr** | — |
| folds positive | 5/6 | — |

The trend filter helps most on crash-prone trenders (BTC 0.79→0.97, ETH 0.54→0.71; TQQQ drawdown
−82%→−52%) and gives back a little Sharpe on smooth equity bulls while halving their drawdown. On a
2000-start with two real bears, cash-below-trend took SPY/QQQ from −69% to −31% drawdown.

**Two sub-findings that mattered:**
- **Shorting HURTS monotonically** (down-exposure 0: Sharpe 1.10 → −0.5: 0.87 → −1.0: 0.67). Shorting
  drift-up assets bleeds on V-recoveries; cash-below-trend captures the protection without the bleed.
  Every short-overlay idea tested (short-after-green, flip-at-exit, inverse ETFs) was rejected for the
  same root cause: shorting fights the positive risk premium the book harvests.
- **The macro overlay** (VIX + cross-asset correlation de-risk) is a modest incremental win on top of
  vol-targeting (1.10 → 1.13, maxDD −27% → −23%, GFC fold 1.30 → 1.88).

This is **not alpha and is not sold as alpha** — against the market the beta is ~1. The claim is
narrow and true: higher Sharpe and shallower drawdown than holding the same assets. It is exactly what
trend-following / tactical-allocation books monetise, and it is the deliverable.

### 5.9 The combined book — diversification finally pays — `combined_book.py`
The fundamental law made concrete. Over the hard 2022-01→2026-05 window (1611 trading days):

| Sleeve | Sharpe | maxDD |
|---|---|---|
| managed-beta (standalone) | 0.65 | −27% |
| ts-momentum 4h (standalone) | 0.52 | −15.5% |
| **correlation between them** | **+0.098** | — |
| **combined 50/50** | **0.80** | **−19.5%** |

Two mediocre-but-uncorrelated streams beat either alone (+23% Sharpe, ~8pts shallower drawdown). This
was the first constructive win of the whole arc — the diversification benefit is structural and *less
overfit-prone* than either standalone Sharpe. Caveat honestly recorded: the ts-momentum ADX gate was
tuned on this same window and its DSR is low, so it is a low-confidence diversifier, small allocation
if any.

### 5.10 Multi-asset daily TSMOM — `tsmom_multiasset.py` — rejected
A third-sleeve attempt: MOP-style long-short daily time-series momentum on the broad ETF+crypto
universe. **Rejected: Sharpe ~0.00 over the full 2010–26 in every lookback variant** (63/126/252,
12m-only, 12-1 skip-month), −0.32 on 2022–26. Same root cause as every short test: the short leg
bleeds against drift-up assets and the long side duplicates managed-beta's trend filter.

### 5.11 FX/commodity trend — candidate, thin
A cleaner third-sleeve candidate: daily trend on UUP/UDN/FX-majors + DBC/USO/UNG/SLV/GLD/DBA/CPER.
**Structurally the thesis held** — correlation 0.035 to the crypto sleeve, 0.163 to managed-beta, the
*most orthogonal stream found*. But standalone Sharpe was only +0.12 on 2022–26, so the book moved
0.27 → 0.28 — a marginal, low-confidence diversifier (and the test window excludes FX trend's best
regimes: the 2014-15 dollar rally, the 2022 commodity spike).

### 5.12 Lead-lag networks — the "Coca-Cola moves Pepsi" idea — `leadlag.py`
Tested the dynamic cross-asset propagation hypothesis (a shock in a leader predicts a follower's next
bar) at **five timescales**, with the follower's own momentum neutralised and costs charged:

| Scale | IC in-sample | IC out-of-sample | Sharpe gross | Sharpe **net** |
|---|---|---|---|---|
| 5m | 0.000 | 0.000 | −1.0 | **−147** |
| 15m | +0.018 | −0.010 | −1.3 | **−71** |
| 30m | +0.028 | +0.001 | −1.5 | **−42** |
| 4h | +0.038 | −0.010 | +0.24 | **−10.7** |
| **1d** | +0.086 | **+0.030** | **+0.42** | **−0.85** |

Intraday lead-lag is dead exactly as the HFT-latency-race prior predicts (you cannot win a
microsecond race at 15bps, no colo). But the **daily row is real**: cross-asset lead-lag *genuinely
persists out-of-sample* (IC_oos +0.030) with positive gross Sharpe — killed only by 60–70%/day
turnover. That is an implementation problem (holding periods, smoothing, dead-bands), not a dead
signal, and it is the one lead-lag thread worth pulling further.

---

## 6. Feature engineering

The `signal_audit.py` IC study is the empirical spine of the whole feature story. Ranked findings:

- **The single best feature found: multi-scale momentum.** `mom_864` (log return over 864 bars) had
  IC ≈ **0.056**, higher than the best base feature (0.044). This productionised into the 9-feature
  `momentum_features` block and the 58-feature "hybrid" contract (49 active base + 9 momentum).
- **Where the (thin) edge concentrates:** HTF 1h/4h EMA/RSI/MACD, EMA distance, VWAP distance, ATR,
  the regime one-hot, and calendar (weekday/hour) — all real but ~0.53–0.54 AUC.
- **Dead on arrival (|IC| < 0.003):** the ~41 structurally-stubbed features (order book, news scalars,
  news embeddings, macro, earnings, constant spread, rsi_divergence, spare pattern). Pruned into
  `ACTIVE_FEATURE_INDICES`.
- **Funding rate:** the one genuinely-new *positioning* axis (not a restatement of the asset's own
  price). Standalone |IC| only **0.009–0.016** → rejected as a directional feature. But funding
  *capture* (the carry strategy) and funding *cost* (the leverage cap) are different, real uses.
- **`accel` (2nd derivative):** IC 0.004 — dead, weaker than velocity everywhere. Dropped.
- **News:** the 20 news dims were structurally zero in training (no bar-aligned historical corpus), so
  the net never learned from news. This is *why* news was moved out of the model and into a live LLM
  risk overlay (Section 9) — news is rare and fat-tailed; a data-starved net cannot learn it, but LLM
  reasoning catches the asymmetric crash-avoidance value.

**The consolidation:** from a 90-feature everything-vector, the honest contract became **58 features**
(49 information-bearing base + 9 multi-scale momentum, z-scored). The legacy 90-wide vector was kept
only for the demoted LSTM A/B path.

---

## 7. Model architectures

A theoretical comparison of every temporal core we built or considered, grounded in what the evidence
said actually matters at this signal-to-noise ratio.

| Architecture | Complexity | Data-efficiency at our scale | Verdict here |
|---|---|---|---|
| **LSTM** (original) | O(T), sequential | Good, but overfit-prone deep | Demoted; memorized (train 0.04 / val 0.16) → shrunk 256/3 → 128/2 |
| **GRU** | O(T), fewer params | Slightly better regularized | Config option; never materially different |
| **TCN** (causal dilated conv) | O(T), parallel, big receptive field | **Best inductive bias / fewest params** | Production trunk; the switchable core |
| **QuantileTCN** | O(T) | Same trunk, calibrated output | The meta-layer model (Section 9) |
| **Transformer / PatchTST** | O(T²) | Worst — needs patching + heavy reg | Deferred; A/B arm only |
| **Mamba / SSM** | O(T), selective state | Competitive, weaker on cross-variable structure | Research arm; the one real reason is cheap long context |
| **Autoregressive candle transformer** | O(T²) + tokeniser | Marginal MAE win (~3–8%) | Deferred by design; trigger never fired |

**Theoretical read:** at 0.53 AUC, architecture is *not the binding constraint*. The signal is so weak
that inductive bias and regularization dominate expressivity — which is exactly why the TCN (strong
prior, few parameters, parallel, full receptive field via dilations) beat the deep LSTM, and why the
last productive architectural change was not a fancier model but a **regularization/wiring fix**: a
last-timestep skip connection into the QuantileTCN head. Additive attention starts near-uniform, so a
signal in the freshest bar is diluted ~T× early in training; a planted last-bar signal was *unlearnable*
at T=60 without the skip and learned in three epochs with it. Where Mamba genuinely earns a look is the
one thing the TCN can't do cheaply — extending context from 60 bars to ~2,000 (a week) without
dilations exploding — but that is a hypothesis to test, not a foregone win.

The **objective function** proved more important than the trunk. We ran two:
- **Pinball (quantile):** predict {p10, p50, p90} of the vol-normalized forward return; edge = p50,
  uncertainty = p90−p10. Calibrated, principled — and it **failed** as a live gate (OOS timing worse
  than baseline).
- **DMN Sharpe (Lim/Zohren/Roberts):** a position head trained to *directly maximize net Sharpe* with a
  turnover cost term. The loss *is* the deliverable. This is what produced the one positive ML result
  (Section 9). Predicting the metric you care about beat predicting an intermediate quantity you then
  threshold.

---

## 8. The risk machinery

The recurring, unglamorous conclusion of the entire project: **the thing that consistently added
risk-adjusted value was not any alpha source — it was the risk machinery.** Four layers, all causal:

1. **Volatility targeting** — scale exposure toward a target annualized vol using trailing realized
   vol (the single most-documented Sharpe improver, Moreira–Muir). The biggest lever on CAGR.
2. **Drawdown de-gearing** — when the book is in a peak-to-trough drawdown beyond a threshold, scale
   the next bar down toward a floor; restore as equity recovers. Caps the depth of a losing run (the
   fix for the −90% mean-reversion amplification).
3. **Funding-aware leverage cap** — perp leverage is financed by funding; a levered position paying
   funding gives back the CAGR the leverage bought. An EMA-smoothed (≈3-day half-life), sign-aware,
   *continuous-ramp* cap de-levers names on the paying side of sustained funding — no threshold to
   whipsaw around.
4. **News/macro overlay** — a tighten-only risk multiplier from the LLM news layer and the VIX/corr
   macro filter.

**The decisive quantitative finding of the final phase** came from `risk_stack_sweep.py`: on the
2022–26 combined book, **Sharpe is ~0.80–0.83 across the entire leverage grid** (leverage-invariant,
exactly as theory says), while **CAGR scales with the vol target** — and **financing cost is the
dominant marginal factor.** With realistic financing charged on the levered portion:

| Financing rate | Best constrained pick (−25% DD budget) | CAGR |
|---|---|---|
| 0%/yr (fantasy) | tv 0.30, lev ≤2.0 | +17.3% |
| 2%/yr | tv 0.15, lev ≤3.0 | +12.3% |
| 5%/yr (realistic) | tv 0.15 | **+11.0%** (Sharpe 0.71, DD −23.2%) |
| 8%/yr | tv 0.15 | +9.7% |

This is *why* the funding cap is offense, not defense: borrowing only when smoothed funding is cheap
is how the live book beats the static financing assumption. The profit lever is real, movable, and
needs no model — but it is bounded by financing, and that is the honest ceiling.

Drawdown-constrained growth theory (Grossman-Zhou, Busseti-Ryu-Boyd risk-constrained Kelly) formalises
the whole picture: maximizing log-wealth under a drawdown constraint is equivalent to acting more
risk-averse. We put the drawdown budget in the risk layer (a hard architectural guarantee) and the
Sharpe floor in the promotion gate — not in the loss, where a benchmark-relative penalty would make
training non-stationary.

---

## 9. The ML meta-layer

ML earned its place back not as a direction-picker but as a **sizing/gating overlay** — a far
better-posed problem than direction (an error costs a mis-size, not a wrong-way trade).

- **Meta-labeling (López de Prado, AFML Ch. 3)** was the canonical idea: keep the primary signal
  binary, train a secondary model to predict P(trade hits target before stop), size on that. **Built
  and walk-forward tested — negative on our book:** in-sample AUC ≈ 0.90 but **OOS AUC ≈ 0.50**, and
  sizing on that noise HURT Sharpe (5-ETF 0.93 → 0.69; 7-asset 1.54 → 0.96). The in-sample→OOS AUC
  collapse is the cleanest single demonstration in the whole project that there is no OOS-predictable
  structure — in direction *or* trade-quality — on an already-clean daily trend primary. Kept as
  scaffolding; not wired to live sizing.
- **The DMN overlay — the one positive ML result.** The QuantileTCN's DMN-Sharpe variant, used to
  *scale* (never originate) the 4h TS-momentum sleeve's position between a 0.25 floor and 1.0, produced
  **+0.37 Sharpe of genuine timing skill** on the untouched test tail (−0.29 vs a −0.66 baseline at the
  same ~0.65 mean weight — the gain is skill, not de-levering, verified against a constant-scaling
  control). The pinball variant, by contrast, showed *negative* timing skill. Caveats recorded
  honestly: one window, a saturated tanh head acting as a binary agree/disagree filter, and a
  two-objective trial count for the DSR. It is wired live behind a flag, paper-only, off by default.
- **Online adaptation, done safely.** Not weight-level online gradient descent (the AWR experiment
  showed per-trade weight updates are noise at this SNR) — instead **online conformal recalibration**
  of the uncertainty interval toward target coverage, plus an **EWMA conviction-shrinkage** that
  collapses the overlay toward neutral if its live hit-rate decays to a coin flip and symmetrically
  recovers. Adaptation at the calibration layer, frozen weights.

The design principle throughout: the model can only shrink or keep the exposure the rule-based
strategy already wants. The worst case of any ML failure is "too conservative", never "blown up".

---

## 10. The graveyard

Rejected ideas, each with the reason — this list is as valuable as the survivors, because it is where
the multiple-testing discipline paid for itself.

| Idea | Verdict | Why |
|---|---|---|
| NN predicts 5m direction | Demoted | 0.53 AUC; competed away |
| 12 classical TA families × 49 assets | 0/588 DSR>0.95 | TA-on-OHLC is dead robustly, not selectively |
| Fourier dominant cycle | Noise floor | Markets aren't stationary-cyclic |
| Dynamic Acceleration Stop / 2nd derivative | Rejected | accel IC 0.004 < velocity everywhere |
| Garman-Klass vol (as vol-target input) | Rejected | Better rank-IC but tracks 3.6% *worse*; proxy ≠ system metric |
| Kelly-Markowitz (Σ⁻¹μ) | Rejected on priors | Textbook error-maximizer with μ≈0; 1/N beats it (DeMiguel) |
| Fractional differentiation (FFD) | Rejected | Crypto already near-stationary; directional IC much weaker than log-returns |
| Hierarchical Risk Parity (HRP) | Rejected for our book | Over-concentrates in the flat/in-cash cluster; needs many assets w/ stable structure |
| Meta-labeling (sizing) | Rejected | OOS AUC 0.50 on clean primary |
| Cross-sectional momentum (broad) | Rejected | Regime-overfit (semis) / too-correlated (crypto) |
| Stat-arb (semis) | Shelved | Dispersion regime poisons reversion |
| Mean-reversion (crypto) | Shelved | Fading a momentum asset = −90% DD |
| Shorting overlays (all variants) | Rejected | Fights the positive risk premium the book harvests |
| Daily multi-asset TSMOM sleeve | Rejected | ~0 Sharpe 16y; duplicates trend filter |
| Intraday lead-lag | Dead | HFT latency race, unwinnable at retail |
| On-chain (stablecoin minting) | Built, gated, unpromoted | Same "front-runs price" claim funding made and failed |
| ExecutionAgent / ArbitrageAgent (infra) | Deferred | Infra-before-alpha is the trap; validate the edge first |

---

## 11. Papers

The literature that directly shaped a decision (not a reading list — a decision log):

- **Moskowitz, Ooi, Pedersen — Time-Series Momentum.** The economic basis for the ts-momentum and
  managed-beta trend sleeves; the most-replicated systematic edge, which is why trend survived when
  reversion didn't.
- **Lim, Zohren, Roberts (2019) — Deep Momentum Networks.** The Sharpe-as-loss + turnover-regularization
  objective; implemented directly as the DMN overlay head, and the source of the project's one positive
  ML result. Successors considered: the Momentum Transformer, X-Trend few-shot regime retrieval.
- **Bailey & López de Prado (2014) — The Deflated Sharpe Ratio.** The multiple-testing gate that killed
  0/588 TA trials and every regime-overfit standout (semis XS +4.65 Sharpe → DSR 0.374). The single
  most-used discipline in the project.
- **López de Prado — Advances in Financial Machine Learning.** Triple-barrier labels, fractional
  differentiation, HRP, and meta-labeling — three of four tested negative here, which is itself a
  finding about our regime and scale.
- **DeMiguel, Garlappi, Uppal (2009) — 1/N.** Why equal-weight beat sample mean-variance and HRP on our
  managed streams; naive diversification is the robust default when μ and Σ are noisy.
- **Grossman & Zhou (1993); Busseti, Ryu, Boyd (2016) — drawdown-constrained / risk-constrained Kelly.**
  The formal frame for putting the drawdown budget in the risk layer, not the loss.
- **Moreira & Muir (2017) — Volatility-Managed Portfolios; VIX-managed variants (2024).** The basis for
  vol-targeting as the dominant Sharpe lever and the implied-vol-blend candidate.
- **Faber — tactical asset allocation / 200-day trend.** The managed-beta core.
- **Grinold — the Fundamental Law of Active Management (IR ≈ IC·√breadth).** The compass after the
  0.53-AUC verdict; the reason the whole project reoriented around breadth and diversification.
- **Poutré, Dionne, Yergeau (2022) — lead-lag arbitrage at high frequency.** The honest prior that
  intraday lead-lag needs a latency race; borne out by the 5m net −147 Sharpe.
- **DeltaLag (2024); lead-lag-aware / Temporal-GNN frameworks.** The generalization the entity graph
  gestures at, and the natural next step *only if* the daily lead-lag signal survives turnover control.
- **Mamba/SSM; PatchTST; TimeSiam; MOMENT; FinCast.** The architecture and self-supervised-pretraining
  research arms — deferred behind the DSR gate, because at 0.53 AUC the trunk is not the constraint.
- **HRT; HARLF (hierarchical RL); TradingAgents, HedgeAgents, FinMem; TradeTrap.** The agentic-systems
  literature — adopted for *structure* (typed control lattice, tighten-only risk precedence) while
  keeping LLMs out of the trade-decision seat, because the benchmarks are cost-unrealistic and TradeTrap
  shows these agents are neither reliable nor faithful under perturbation.

---

## 12. Master results table

Every measured strategy/idea, one row each. Sharpe/DSR are the honest walk-forward numbers; "verdict"
is the live-book decision.

| Strategy / idea | Best Sharpe | Deflated Sharpe | Reg alpha | Beta | Verdict |
|---|---|---|---|---|---|
| NN 5m direction | — (0.53 AUC) | — | — | — | Demoted |
| TA zoo (12×49) | best ~0.7 raw | 0/588 > 0.95 | — | — | Dead |
| ts-momentum 1h | −0.15 to −0.83 | 0.001 | — | ≈0 | Dead |
| ts-momentum 4h+ADX | +0.55 to +0.94 | 0.14–0.30 | +16.6%/yr | ≈0 | **Low-conf diversifier (in book)** |
| xs-momentum stocks (semis) | +4.65 | 0.374 | +0.20 | ≈0 | Rejected (regime-overfit) |
| xs-momentum crypto | +0.17 → −0.39 | 0.002 | — | ≈0 | Dead |
| stat-arb (semis) | +0.02 | 0.000 | −43.5%/yr | ≈0 | Shelved |
| mean-reversion crypto | −90% DD | 0.000 | — | — | Shelved |
| funding carry | +0.04 | 0.690 | ~0 | ≈0 | Real but tiny (+0.2%/yr) |
| factor momentum_12_1 | +0.31 | 0.234 | ~0 | +0.14 | Beta in disguise |
| factor reversal_1m | +0.39 | 0.350 | −1.4%/yr | +0.25 | Beta + negative skill |
| factor combined {mom,rev} | +0.46 | 0.469 | ~0 to −0.6% | +0.20 | No neutral alpha |
| value (fundamental) | — | — | +0.1%/yr | +0.06 | Real, marginal |
| quality (fundamental) | — | — | +0.3%/yr | +0.02 | Real, marginal |
| value+quality | 0.41 | 0.394 | +0.2%/yr | +0.04 | First clean neutral alpha; tiny |
| **managed-beta (combined)** | **1.10 (→1.13 macro)** | — (5/6 folds) | **+7.4%/yr vs B&H** | ~1 | **THE DELIVERABLE** |
| **2-sleeve (mb + ts-4h)** | **0.80** | — | — | ~0.5 | **Live paper book** |
| tsmom-multiasset daily | ~0.00 | — | — | — | Rejected |
| FX/commodity trend | +0.12 standalone | — | — | low | Candidate, thin |
| lead-lag intraday | net −10 to −147 | 0.000 | — | — | Dead |
| lead-lag daily | gross +0.42 / net −0.85 | — | IC_oos +0.03 | — | Real, turnover-bound (open thread) |
| DMN overlay (sizing) | +0.37 timing skill | — | — | — | **Wired, paper, flagged** |
| pinball overlay | −0.86 (worse) | — | — | — | Parked |
| meta-labeling | OOS AUC 0.50 | — | — | — | Parked |

---

## 13. Current state

**The deliverable:** a two-sleeve managed-beta + 4h TS-momentum paper book (Sharpe ~0.80 on the hard
2022–26 window, maxDD ~−19.5%), governed by the full risk stack (vol-target + drawdown de-gear +
funding cap + news/macro overlay), with the DMN sizing overlay available behind a flag. It beats the
S&P Sharpe comfortably by construction, because leverage doesn't change Sharpe and the book's ~0.8
already clears the ~0.55 market Sharpe.

**Runs today** via `start.bat` in paper mode: the NN policy trader is skipped (untrained by design —
`NN_AGENT_ENABLED=false`), and the rule-based StrategyAgent trades on real data, publishing its
net-worth curve, allocation breakdown, and the live cross-asset entity graph to the dashboard.

**The models are NOT trained** beyond quick 30-minute Colab probes — a deliberate ordering decision:
validate the training-free rule-based deliverable in paper first, because the profit lever
(vol-target/financing) needs no model and the model's marginal value is the speculative part.

**Open threads worth pulling, in priority order:**
1. Set the vol target to the chosen drawdown budget (config; the biggest CAGR move, needs no model).
2. A full multi-hour DMN overlay retrain (with the de-saturation fixes and honest trial count), then
   A/B at book level before enabling it hot.
3. Turnover-controlled daily lead-lag — the one research thread that showed genuine OOS persistence.
4. FX/commodity sleeve on full history (its best regimes are outside the 2022–26 test window).
5. The RL smart-order-router — only once the slippage ledger proves the deterministic passive/TWAP
   execution is leaving basis points on the table (the L1 tick logger is already capturing the corpus).

---

## 14. Meta-lessons

The through-lines that generalize beyond this project:

1. **Test the system metric, not the proxy.** Garman-Klass had better rank-IC and still lost on vol
   tracking. Meta-labeling had 0.90 in-sample AUC and 0.50 OOS. A better proxy that doesn't move the
   money number is worse than useless — it's a trap that costs build time.
2. **Deflate for multiple testing, always.** The semis XS book at Sharpe +4.65 was a mirage; the DSR
   said so (0.374) and the broadened re-run confirmed it (−0.09). Without the DSR gate we would have
   shipped regime luck as alpha.
3. **Separate alpha from beta with a regression, every time.** The daily factor book's entire +2.5%/yr
   was incidental market beta. A headline Sharpe that survives α/β decomposition is the only Sharpe
   worth trusting.
4. **Diversification is the most reliable edge.** The first constructive win came not from a better
   predictor but from combining two mediocre uncorrelated streams (corr +0.10) — and the benefit is
   structural, not fitted.
5. **At low SNR, the objective and the regularization matter more than the architecture.** The TCN beat
   the LSTM on inductive bias; the last productive change was a wiring fix, not a bigger model; and
   Sharpe-as-loss beat quantile-then-threshold. Expressivity is not the bottleneck when the signal is
   0.53 AUC.
6. **The risk machinery is the durable edge; alpha at retail scale is ~0 almost everywhere.** After
   testing direction prediction, TA, factors, fundamentals, stat-arb, carry, and lead-lag, the thing
   that consistently added risk-adjusted value was disciplined vol-targeting, drawdown control, and
   trend-filtered beta harvesting. That is not a disappointing conclusion — it is the honest one, and
   it is what actually makes money at this scale.
7. **Keep the model on a leash.** ML's safe role is to *scale* what a robust rule already wants, never
   to originate; then the worst failure mode is "too cautious", not "ruined".

*The blunt summary: we set out to build a machine that predicts the market, discovered that machine
cannot exist at our scale, and built instead a machine that manages risk better than buy-and-hold and
diversifies mediocre edges into a respectable one. That is the whole story, and the numbers above are
the receipts.*

---

## 15. Running decision log

> **Standing rule (owner, 2026-07-04):** every time we consider an idea or make a material edit, this
> section gets an entry — what was considered, the verdict, and the numbers. Newest entries at the top.

### 2026-07-04 — Built the three accepted adaptations (journal/post-mortem, scanner, GBM arm)

**Made:** the three actionables from the entry below, owner-approved with one modification —
the learning loop writes everything to a reviewable JSON so the owner can periodically hand it
to the strong reviewing model for deep post-mortem, while the local LLM gets distilled lessons.

1. **Trade journal + post-mortem loop** (`backend/agents/trade_journal.py`,
   `scripts/postmortem.py`). Append-only JSONL (`training_data/trade_journal.jsonl`) of every
   rebalance / order (with decision price + notional) / mark-to-market / per-symbol outcome.
   The journal *records, never decides*. The post-mortem runs on a WINDOW (default 7d) and
   **classifies before learning**: PROCESS (slippage above the assumed model, gross over cap,
   journal gaps, repeated risk blocks) → always actionable; NOISE (window loss within ±2σ of
   the book's own per-mark vol) → explicitly non-actionable, never becomes a lesson; REGIME
   (loss beyond −2σ, or broad hit-rate decay ≥3 symbols under 35% over ≥10 marks) → allocation-
   level observation. Lessons (LLM-phrased when configured, deterministic otherwise; the LLM
   can only *phrase*, classification is deterministic) go to the existing SkillBook; a JSON
   report lands in `statements/postmortem_<date>.json` — the hand-to-Claude artifact. The mark
   loop also feeds realized per-symbol returns to the overlay's ConvictionShrink (closing the
   `record_outcome` gap flagged below). Test coverage found + fixed a real hole: a
   zero-variance losing stream slipped through a `vol > 0` guard unclassified — now the most
   "beyond expectation" case by construction.
2. **Anomaly/spread scanner** (`backend/signals/anomaly_scanner.py`). Whole perp universe in
   2 REST calls / 15 min; cross-sectional z-flags (|move z| ≥ 3, volume z ≥ 3), spread ≥ 10bps,
   24h quote volume < $5M. **Liquidity flags de-gear (tighten-only, floor 0.5); attention flags
   never de-gear** (momentum sleeves want moves). Fails open to 1.0. Wired into the agent's
   perp targets + `ANOMALY_SCANNER_ENABLED` (default on).
3. **QuantileGBM ensemble arm** (`--with-gbm` in `train_quantile_tcn.py`). The scaffolded
   LightGBM co-model finally used: pooled tabular fit on last-row 58-feature snapshots, duck-
   typed into the overlay ensemble + A/B. Planted-signal test: edge↔target corr > 0.5 on the
   val split; verdict on real data awaits the owner's next training run (judged by the same
   timing-skill-vs-control bar as the TCN arms).

**Numbers:** 9/9 new tests green; suite green. Verdicts pending live data: the scanner's de-gear
and the GBM arm's A/B only earn keep via the paper shadow / next Colab run.

### 2026-07-04 — External multi-agent "scan → research → predict → risk → post-mortem" system

**Considered:** an agent stack the owner read about: (a) a scan agent over ~300,000 markets filtering
by liquidity/volume/resolution and flagging weird moves and wide spreads; (b) two parallel research
agents scraping Twitter/Reddit/RSS, running sentiment, comparing narrative vs market odds; (c) a
prediction agent (XGBoost + LLM) calibrating "true probability vs market price", firing only above a
confidence threshold; (d) a risk agent that blocks oversized trades and sizes by edge; (e) a 5-agent
post-mortem after **every loss** that fixes code/skills/prompts, with Claude bootstrapping fixes so a
local Ollama model can later maintain the skill/prompt layer.

**Context check first:** "300,000 markets" + "narrative vs market odds" identifies this as a
**prediction-market** system (Polymarket/Kalshi-style). That domain differs from ours in two decisive
ways: (1) breadth there is *real* — event markets are near-uncorrelated bets, whereas our measured
crypto/equity universe is correlation-capped (effective N = N/(1+(N−1)ρ̄) → ~1.4 effective bets at
ρ̄≈0.7 no matter how many tickers we scan); (2) the "market odds" are explicit prices set largely by
retail on illiquid events, so an LLM reading news *can* out-calibrate the crowd — in liquid price
markets the crowd includes HFT and the equivalent claim already failed our audits repeatedly.

**Feature-by-feature verdicts:**

| Feature | Verdict | Reasoning / numbers |
|---|---|---|
| (a) Wide scan + anomaly/spread flagging | **ADAPT — as a risk/attention input, not alpha** | Universe breadth ≠ alpha here (crypto XS over 20 names: Sharpe −0.39, DSR 0.002; broad-12 managed book *diluted* Sharpe 1.08→0.94). But an anomaly scanner over the full Binance perp universe (~300 symbols) flagging abnormal moves/spread blowouts is cheap and useful as: a de-gear trigger (spread blowout = liquidity risk), a news–entity-graph cross-check (did a flagged move have a narrative?), and a TS-sleeve universe refresher. We already have the seed: the Variable Attention Engine (vol/R²/volume thresholds) — this generalizes it cross-sectionally. |
| (b) Twitter/Reddit/RSS research agents, narrative vs odds | **PARTIAL — strengthen what exists; gate any new source behind the IC audit** | We already run the RSS→LLM news pipeline, credibility engine, tighten-only news→risk overlay, and entity-graph propagation. The *comparison structure* (narrative vs market-implied expectation) already exists in embryo: `NewsPrediction.outcome_checked` scores predicted vs realized moves. Social sentiment as a *feature* faces the funding-rate precedent — "front-runs price" claims measured |IC| 0.009–0.016 → rejected; any Reddit/Twitter signal goes through `AUDIT_*=1` in signal_audit before a single feature-spec change. No scraping infra before the edge is measured (infra-before-alpha is the trap, per the ArbitrageAgent verdict). |
| (c) XGBoost+LLM probability calibration, confidence-gated firing | **ALREADY HAVE — in stronger, tested form** | Our stack: QuantileGBM (LightGBM, scaffolded) + QuantileTCN/DMN overlay + IR≥1.5 Kelly gate + OnlineConformal (provable coverage tracking) + ConvictionShrink. "Fire only above threshold" = the uncertainty gate. LLMs are poorly calibrated for numeric price probabilities — validated role here is the risk overlay, not probability estimation. One genuine borrowable: the **GBM co-model is scaffolded but unused** — add a QuantileGBM arm to the overlay ensemble A/B (ensemble.py QuantileStacker already exists for exactly this). |
| (d) Risk agent: size-by-edge, block oversized/risky | **ALREADY HAVE — superset** | Fractional Kelly on the lower credible bound of edge, CVaR gate, drawdown halt + de-gear, per-order notional caps, funding-aware leverage cap, tighten-only overlays, vol targeting. Nothing in the description we lack. |
| (e) Post-mortem after EVERY loss, auto-fixing the system | **ADAPT WITH A CRITICAL FIX — per-loss fixing is the online-AWR failure mode in prose** | A correct +EV strategy loses ~40–50% of individual trades; "fix the system so the same mistake never recurs" after each loss = fitting noise (we measured this: per-trade weight updates produced noise; meta-labeling's in-sample 0.90 AUC collapsed to 0.50 OOS). The sound version: **periodic/threshold-triggered post-mortems that first CLASSIFY losses** into (1) process errors (bug, bad fill vs slippage ledger, risk-rule violation, stale data) → always fix; (2) statistical drawdown within expectation → explicitly do NOT touch; (3) regime decay → already handled by ConvictionShrink/de-gear/DSR-gated sleeve retirement. LLM writes lessons to the existing **SkillBook** (recurrence-escalating, decay-weighted — already built) — prompts/skills only, never code or parameters, mirroring the propose→validator-disposes rail. The Claude-bootstraps-Ollama idea maps cleanly: LLM-distilled lessons accumulate in the skill books the local model reads. Known gap this highlights: `record_outcome` is still not wired from trade settlement into the SkillBook/overlay — that wiring is the actionable. |

**Net actionables accepted (pending owner go):** (1) cross-sectional anomaly/spread scanner as a risk
input; (2) wire `record_outcome` from paper-book settlement into SkillBook + overlay shrinkage and add
a periodic classify-then-learn post-mortem job (process/noise/regime triage); (3) add the QuantileGBM
arm to the overlay ensemble A/B. **Rejected:** per-loss system mutation; social-media scraping ahead of
an IC audit; treating LLMs as probability calibrators; equating ticker-count with breadth.
