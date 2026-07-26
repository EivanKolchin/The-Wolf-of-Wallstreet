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

### 2026-07-04 — Frontend/UX cycle: chart crash, news-feedback ranking, copilot, paper wiring, page speed

**Chart crash (`Cannot update oldest data`)** — `lightweight-charts` `series.update()` throws when a
candle's time is ≤ the last bar's, which happens on out-of-order WS ticks after a symbol/timeframe
switch. Fixed by guarding all three `update()` sites in `TradingChart.tsx` (kline, stock tick, aggTrade)
with a `time >= lastKnown` check + try/catch. Verdict: **DONE** (defensive; no happy-path change).

**News feedback → recommendation ranking** — audited the loop end-to-end: ratings persist to
`training_data/news_feedback.json` (source/keyword/severity/rating weights, clamped ±3) and already fed
(a) the RSS pre-filter (`filter_relevant` boost/suppress), (b) the trust threshold + skip in
`run_pipeline_processor` (interest ≤ −1.5 drops the article), (c) the LLM prompt guidance, and (d) the
severity adjustment (bias ≥0.55 → ±1 level). **Missing link found & fixed:** the displayed order at
`/api/news/recent` was pure recency — now blended `recency + 0.25·interest_score`, returns
`interest_score` per item. Verdict: **DONE**; `test_news_feedback + test_news_adaptive = 8 passed`.

**Market copilot** — rebuilt `/api/agent/chat` to match "cheap/local first, escalate only if needed."
Local Ollama (`llama3.1`) answers portfolio/system questions from injected internal context (paper/live
mode, open trades, agent heartbeats, risk status, universe, recent news). External questions (earnings,
**consensus/analyst/price-target**, macro) are detected by `build_search_query` (broadened triggers +
generic question-word fallback gated by `_INTERNAL_MARKERS`), DuckDuckGo-searched, and — when snippets
are thin — enriched with `fetch_page_text` of the top 2 hits. Research → **Gemini 2.5 Flash**
(`tier="flash"`), escalating to **2.5 Pro** (`tier="sonnet"`) only if Flash returns empty. Provider
auto-upgrades `ollama→hybrid_gemini` for this endpoint when a Gemini key is present. Bumped `llm.py`
Gemini 1.5→2.5, added `keep_alive:30m` (cuts Ollama reload latency), widened cloud context to 30k.
Live-tested: internal Q → local model, correct paper/0-trades answer; "analyst consensus on NVDA" →
gemini flash, 5 results, correct Buy/PT synthesis. Frontend shows a per-message route chip. **DONE**.

**Two portfolio values → keep the real one** — dashboard showed both the **Virtual Ledger**
(`/api/portfolio`, NN/DeFi book seeded at `INITIAL_USDC_AMOUNT`=$1000) and the **Strategy Book**
(`/api/strategy/portfolio`, 2-sleeve managed-beta+TSMOM at $100k). With `NN_AGENT_ENABLED=false` the
DeFi book never trades → the $1000 ledger is dead. **Removed the Virtual Ledger card**; Strategy Book is
now the sole portfolio view. Verdict: **DONE** (owner-confirmed the $1000 book is fake).

**Positions / audit paper wiring** — StrategyAgent doesn't write `Trade` rows (weight-book in Redis +
JSONL journal), so positions/audit (which read the `Trade` table) were permanently empty in paper mode.
Rewired: **positions** now shows Strategy Book allocations (asset/side/weight/value) with the `Trade`
view as the live/broker fallback; **audit** reads a new `/api/strategy/journal` (reads the agent's
`trade_journal.jsonl` — rebalances + orders, newest first) with a `Trade`-table fallback. Enriched
`/api/positions` to mark against `portfolio:live_state`, seeded `context.positions` from `/api/positions`
on mount, WS `trade` topic now drops closed trades. **Live-switch reset:** factored the paper wipe into
`_reset_paper_state()`, called automatically from `/api/setup/save` when `PAPER_TRADING` flips true→false
so live books never inherit paper history. **DONE**; journal endpoint live-tested (empty→inactive; seeded
rows→2 events newest-first).

### 2026-07-07 — Full-system audit (owner: "live for a while, no net gains — find every improvement")

**Entity-graph card removed from dashboard** (owner request): `EntityGraphCard.tsx` deleted, dashboard
grid single-column for StrategyBookCard. Backend graph/propagation left in place (feeds the news
overlay; separate decision).

**AUDIT — why the paper book shows no net gains (4 smoking guns + calibration, all evidence-backed):**
1. **PaperBook is in-memory — every backend restart resets equity to $100k** and re-pays the full
   establishment cost (~$22.56 = 2.3bps per restart at gross 0.45). Journal shows 2 resets in 2.7 days;
   chained across resets the book actually made **≈ +0.56%** (07-04→07-07) while the dashboard showed ~$0.
   No cumulative track record can exist without persistence.
2. **The TS sleeve + anomaly scanner + overlay provider all read Binance futures TESTNET data**
   (`BINANCE_FUTURES_TESTNET: bool = True` default, not overridden in .env; mainnet fapi is reachable —
   verified 200). Half the book trades Donchian/ATR/EMA signals computed on synthetic testnet prices.
3. **Live TS sleeve runs UNVALIDATED params**: main.py passes raw `TSMomentumParams()` (ema_trend=100,
   adx_min=0 — no ADX gate) instead of the validated set (entry 48 / exit 24 / ema 50 / **adx_min 25**)
   that salvaged the sleeve (Sharpe +0.94; without the gate it measured −0.83).
4. **Book runs at ≈half (or less) of the intended risk**: tv=0.30 is applied INSIDE the managed sleeve
   only, then ×0.5 sleeve weight; TS sleeve vol-targets its own internal default **0.12** (agent never
   passes ts_kwargs — also means the funding-cap EMA is dead live); no book-level re-target. Position
   math from live allocations → true book vol ≈ 13-16% vs 30% intent; journal-realized ≈ 4% (stale daily
   marks understate). At half-risk + resets, expected drift is invisible by construction.
   Also: `min_rebalance_delta=0.02` is 25-65% of typical position sizes (coarse quantization); risk
   manager's `is_halted` tracks the DEAD NN book (INITIAL_USDC=1000), not the paper book; mark cadence
   has 13.5h gaps (PC sleep) — uptime holes in the curve.
**Agentic layer:** news agent tier="sonnet" under AI_PROVIDER=ollama → llama3.1 locally → ~8 analyses/day
(23 in 3d), severity mis-calibrated (30% SEVERE); post-mortem loop (scripts/postmortem.py) built but
NEVER scheduled; SkillBook only wired to the news LLM-verifier which is OFF; StrategyAgent publishes NO
heartbeat (engine banner reflects the disabled NN agent); copilot context reads `portfolio:live_state`
(NN book) not `strategy:portfolio` (real book). Anomaly flags published to Redis but never surfaced in UI.
**Honest math for the owner:** even at full tv=0.30 & Sharpe 0.8, ~35% of months are negative; days of
sideways equity is the EXPECTED behavior, not a defect — but the 5 findings above must land before the
curve means anything. Plan proposed (P0 measurement integrity → P1 sizing → P2 agents → P3 frontend →
P4 training/go-live); coding awaits owner approval.

### 2026-07-07 (part 2) — Audit fixes IMPLEMENTED P0–P3 (owner: "implement as much as you can")

Entity-graph dashboard card removed (`EntityGraphCard.tsx` deleted). Then the 5 audit findings + agent
layer, all behind config, suite green (35 targeted tests + full FE build):

- **P0.1 PaperBook persistence** (the restart-wipe fix): `PaperBook.to_state/load_state`; StrategyAgent
  persists `strategy:paperbook:state` to Redis on every publish and `_restore_paper_book()` on boot →
  restarts RESUME equity/positions/history instead of resetting to seed + re-paying entry cost. Reset
  endpoint clears the new keys; `paper:reset_requested` skips restore. test_state_persistence_roundtrip.
- **P0.2 Mainnet data**: Binance4hBarProvider / BinanceMultiTFProvider / AnomalyScanner now ALWAYS read
  `fapi.binance.com` (testnet klines are synthetic — the TS sleeve had been trading fabricated prices).
  Testnet still applies to order routing only.
- **P0.3 Validated TS params**: config `STRATEGY_AGENT_TS_*` (ema50/adx25/48-24) wired in main.py,
  replacing the raw `TSMomentumParams()` (ema100/adx0) the agent used to pass → live sleeve now matches
  the +0.94-Sharpe research (was running the −0.83 no-gate config).
- **P0.4 Heartbeat/status**: agent pings `heartbeat:strategy_agent` + publishes `strategy:status`;
  `/api/agent/status` falls back to a synthesized strategy-book status when the NN agent is disabled →
  dashboard banner shows the book that's actually trading (verified: engine="strategy").
- **P1.5 Book-level vol target** (the ~half-risk fix): `_book_realized_vol` (mixed-freq daily-collapsed
  cov of actual holdings) → scale whole combined book to `STRATEGY_AGENT_BOOK_VOL_TARGET` capped by
  max-leverage. .env: sleeve internal tv back to validated 0.15, BOOK_VOL_TARGET=0.30 (the aggressive
  −25% DD intent, now applied where it belongs). test_book_realized_vol_estimate.
- **P1.6 Book-equity drawdown de-gear**: the validated `drawdown_degear` formula applied live off the
  paper book's OWN peak-to-trough equity (RiskManager tracks the dead $1000 NN book). Proportional
  dead-band via `STRATEGY_AGENT_MIN_REBALANCE_DELTA`.
- **P2 agents**: news agent → `hybrid_gemini` when a Gemini key exists + tier "flash" (Gemini 2.5 Flash
  vs local llama3.1 — ~100× throughput, better calibration) + low-confidence-SEVERE demotion guard.
  Post-mortem SCHEDULED: `postmortem.run_once()` extracted + `_run_postmortem_scheduler` (every 24h,
  deterministic, → SkillBook + statements/). Copilot context now reads `strategy_book` (strategy:portfolio
  + status + journal_summary + anomaly_flags) as the LIVE book; prompt points portfolio Qs there, not the
  disabled NN block. Scanner `.publish()` finally called (strategy:anomalies). Verified run_once on the
  real journal: +0.52% window, flagged the uptime gap, wrote 2 lessons.
- **P3 frontend**: `/api/strategy/health` (realized vol vs target, realized Sharpe, current/worst DD,
  uptime coverage + mark gaps, active news de-gears, anomaly counts) + `BookHealthCard.tsx` in the freed
  dashboard slot — the panel that would have surfaced "running at half risk" months ago. FE build green.

**Owner still runs (P4, gated):** overlay DMN training (Colab, notebook exists) → +0.05 book-Sharpe bar;
full NN pretrain (Colab GPU) → evaluate.py net-alpha gate. Start after a restart lets P0 accumulate a
clean multi-day curve. All new behavior reversible via the STRATEGY_AGENT_* / POSTMORTEM_* flags.

### 2026-07-07 (part 3) — profit-push: convex exits + single-name equities (owner-directed)

Owner goal restated precisely: **maximize compound net-worth growth with positive skew** ("small red /
big green"). Correct framing given to owner: that's the Kelly/log-growth + trend-following-convexity
objective; drawdown depth hurts compounding, so capping it SERVES growth (their two asks are aligned);
"push to limits" is bounded by the Kelly-optimal vol (~half-Kelly ≈ 40% vol for Sharpe 0.8, so tv=0.30 is
near the sensible max, not a floor to raise). Also corrected my own earlier number: at Sharpe 0.8 it's
~40% of MONTHS negative (not 1 in 3), but only ~21% of YEARS.

- **CONVEX EXITS (built, tested, live ON):** `TSMomentumParams.long_stop/short_stop` — asymmetric stop:
  tight `initial_atr_mult`=1.5 (cut losers) until the trade is `activation_atr`=1.0 ATR in profit, then a
  wide `trail_atr_mult`=4.0 trail FLOORED AT BREAKEVEN (winner can't revert to a loss). Default
  `convex_exits=False` preserves the validated symmetric Chandelier (all existing tests unchanged); .env
  turns it ON for the live paper book. A/B harness flag `combined_book_research.py --convex-exits` prints
  a symmetric-vs-convex skew/gain-pain/win-loss/maxDD table (`_skew_stats`/`_print_ab_skew`). Owner must
  backtest to confirm the params before trusting live (convex raises turnover). 8 new assertions +
  path-test in test_ts_momentum (14 pass). This is the direct lever for the "small red / big green" shape.
- **SINGLE-NAME EQUITIES (built, smoke-verified):** `STRATEGY_AGENT_STOCK_SYMBOLS` (default "", .env set
  to NVDA AMD TSLA MSTR COIN PLTR) merged into the managed universe in main.py → they trade through the
  SAME validated managed-beta machinery (200-EMA trend filter + book vol-target + per-name news overlay).
  Smoke: NVDA/AMD/TSLA held (above trend), TLT/GLD/some cashed. HONEST CAVEATS given: (1) momentum-BETA,
  NOT validated alpha (single-name XS momentum was overfit); (2) these names are ~0.6-0.8 correlated
  tech/crypto-proxies → CONCENTRATES the book in one theme (not diversification); (3) inner-join alignment
  means a short-history name (COIN/PLTR) truncates the common window (still ≥1200 bars, fine for 200-EMA);
  (4) earnings-gap risk jumps through stops.
- **Leveraged-ETP / LSE timing-arb idea — REJECTED as an arb, honored in spirit:** the owner's "limit
  order fills at LSE open before the leveraged ETP reprices" is a latency arbitrage that does NOT work —
  ETP market-makers quote off live iNAV, so the LSE-open quote already reflects the underlying's overnight
  move; there's no stale price to pick off (and daily-reset leverage has vol-decay that hurts holding).
  Legitimate adjacent truth: leveraged products (TQQQ already in the book; US single-name ETPs TSLL/MSTU/
  CONL/PTIR) express high-conviction trends with more punch — pairs OK with convex short-hold exits, but
  carries decay. NOT built (LSE needs IBKR live routing; can't paper-test).
- **Limit orders:** perps already default to passive post-only limits (EXECUTION_PERP_STYLE=passive);
  AlpacaBroker HAS a limit path (alpaca_broker.py:172) but the managed router currently sends market —
  wiring stocks to prefer limit is a LIVE-cutover item (paper doesn't route, so untestable now).
- **NEXT (owner's ordered list):** Kelly-conviction sizing, then overlay (DMN) training. Book now at gross
  ~1.05 (vol-target working); managed universe = 13 names incl. single stocks.

### 2026-07-08 — Kelly sizing + universe/graph/news/chart expansion (owner-directed)

- **KELLY-CONVICTION SIZING (built, tested, live ON):** `StrategyAgent._conviction_scale` tilts weights
  by SIGNED trend strength — z = (price−EMA50)/EMA50 / return-vol, signed by position direction,
  conviction = clip(1 + gain·tanh(signed_z/2), 1/cap, cap). Applied BEFORE the book vol-target so it's a
  pure tilt (bet bigger on strong aligned trends, smaller on marginal/countertrend) with total book risk
  unchanged; vol-target re-normalizes gross. Fractional-Kelly in spirit (conviction≈edge, vol-target≈1/var).
  Config `STRATEGY_AGENT_CONVICTION_GAIN=0.5`/`_CAP=2.0` (0=off). Test verifies uptrend-long > downtrend-long,
  bounded, sign-aware short, gain=0 neutral (25 strat tests pass).
- **TRADABLE UNIVERSE EXPANDED:** added GOOGL, MSFT (megacap diversifiers), RKLB (space), RGTI (quantum) to
  `STOCK_UNDERLYINGS` + US_EXCHANGE + ETP_MAP (route underlying; GGLL/MSFL/RGTL noted) + improved_model.SYMBOLS
  ids 22-25 (appended for future retrain) + the live `STRATEGY_AGENT_STOCK_SYMBOLS`. **Kioxia = Tokyo-only
  (285A.T, JPY) → NOT US-routable via Alpaca; intentionally excluded** (needs a Japan venue), flagged to owner.
  SanDisk(SNDK)/Micron(MU) already present.
- **ENTITY GRAPH now mirrors the full tradable universe:** added BTC→all-altcoin + ETH→{SOL,AAVE,RENDER}
  sector edges (so every tradable crypto is a node) + NVDA/MSFT→GOOGL AI-platform edges; `view()` seeds nodes
  from `universe.CRYPTO_SYMBOLS+STOCK_UNDERLYINGS` with a `tradable` flag so edge-less names (RKLB/RGTI) still
  appear. 27 nodes / 23 edges. (Backend graph feeds the news overlay; the dashboard card was removed earlier.)
- **NEWS KEYWORDS for ALL tradable stocks:** `backbone.STOCK_KEYWORD_BANK` (15 stocks: company+ticker+people+
  products) + missing cryptos RENDER/NEAR added; combined `KEYWORD_BANK` is now the `extract_symbol_relevance`
  default (was crypto-only → stock news never mapped to stock symbols). Substring-match safe: ambiguous 2-letter
  tokens (be/mu/bare-coin) DELIBERATELY excluded (verified: generic "…would be much better to coin…" → 0 false
  matches; 8 stock headlines all detected; Elon news correctly hits both TSLA+DOGE). `map_asset_to_symbol` now
  maps bare stock tickers to themselves + RNDR/NEAR aliases. RSS feeds already carry stock news (CNBC/Yahoo/NYT/BBC).
- **PORTFOLIO CHART fixed + range buttons:** StrategyBookCard net-worth chart switched to a numeric-`ts` time
  axis with adaptive tick formatting (intraday→clock, ≤120d→date, longer→month-year) + a Tooltip labelFormatter
  (was a pre-formatted category axis that mislabeled). Added **1D/1W/1M/3M/1Y/All** range buttons (default All)
  that window the history client-side + show the window's %-return. PaperBook history retained 1000→9000 points
  (~1yr hourly) so the long ranges have data. FE build green.

**Owner:** RESTART backend+frontend to activate (backend changes: Kelly, universe, keywords, history depth;
frontend: chart). Convex-exit + book params still want a backtest A/B before full trust (paper is safe to run).

### 2026-07-08 (part 2) — News duplicate fix + restart

Owner saw repeated news. Root cause: dedup hash = `sha256(headline+domain)` (so the SAME story from a
different outlet = a different hash → re-emitted) AND the seen-set was in-memory only (every restart
re-emitted all recent stories). Fix, 3 layers: (1) **content-based key** `_content_key(headline)` —
lowercased, punctuation/whitespace-normalized, cross-outlet suffix junk stripped ("business live",
"| Reuters", etc.), no domain → all outlet variants collapse to ONE key (tested: 4 variants → 1 key,
distinct stories stay distinct); (2) **Redis SET-NX persistence** (`news:seen:{key}`, 72h TTL) passed into
NewsIngestionPipeline from main.py → survives restarts (tested: new pipeline instance = "restart" still
suppresses); (3) **display dedup** in `/api/news/recent` (collapse same content-key, overfetch ×5) so
pre-existing DB dupes don't show either. 8 news tests pass. Restarted backend+frontend — log confirms
paper_book_restored (99933.34, persistence working), full expanded universe live, graph propagating
BTC→altcoin news.

**Page-load speed** — launchers ran `npm run dev` (on-demand per-page compile, multi-second first hits).
Switched `start.bat`/`start.sh` to `npm run prod` → new `scripts/prod-start.mjs`: rebuilds only when
`app/`/`components/`/`lib/`/config mtimes exceed `.next/BUILD_ID`, else serves the existing build; **falls
back to `next dev` if the build fails**. Had to **remove `output: 'standalone'`** from `next.config.mjs`
(it rewrites the server chunk layout for container deploys and is incompatible with `next start` — broke
the viem vendor chunk); added an env-gated `distDir` for isolated prod builds. Measured: prod serves
dashboard/positions/audit/settings at **20–50 ms / HTTP 200** vs dev's multi-second compiles. **DONE**;
full prod build green (type-checking on), all routes prerendered.

### 2026-07-09 — Owner bug sweep: chat hang, crypto-only anomalies, frozen net worth, blank perf page, loss drift

Five owner-reported defects, each diagnosed to a root cause before editing. All backend tests green
(**631 passed**); frontend `tsc --noEmit` clean.

- **Copilot chat returned NOTHING (`"why did the agent not go long with sandisk"`).** Reproduced against
  the running backend: the POST **hung 237.5 s with no HTTP response** — not a 500. Root cause: `llm.py`
  called Gemini `generate_content_async(prompt)` with **no timeout** and then did a bare `return
  response.text`; the SDK's `.text` accessor *raises* when a candidate has no text part (safety block /
  RECITATION / MAX_TOKENS-on-thinking), and the final `raise e` propagated into the un-wrapped
  `agent_chat`. Fixes: (1) `_gemini_text()` guard — returns `""`, salvaging text parts off `candidates`
  first; (2) `request_options={"timeout":45}` + generous `max_output_tokens` (2.5 charges thinking to the
  cap); (3) no more `raise e` — degrade to Ollama (hybrid) or `""`; (4) **`asyncio.wait_for` hard bound of
  75 s** on the whole LLM step in `agent_chat` (< the frontend's new 90 s `AbortController`), so the
  backend always wins the race and returns a real message; (5) bounded the page-fetch loop (10 s each).
  Also: `_extract_chat_symbols` only matched **tickers**, so "sandisk" never resolved → the copilot had no
  SNDK context to reason about. Added `_CHAT_NAME_ALIASES` (word-boundary matched, so "sandisks" /
  "googlebot" correctly do NOT match). Verified: `"…long with sandisk"` → `['SNDK']`.
- **Anomaly flags were crypto-only.** The scanner *never asked a stock provider* — it classified whatever
  `fapi.binance.com` returned (~700 perps). Added an Alpaca `/v2/stocks/snapshots` leg over
  `STOCK_UNDERLYINGS`, classified in its **own cross-section** (z-scores are only meaningful within an
  asset class), then merged. Two calibration bugs found by measuring rather than assuming:
  (a) reusing the crypto thresholds flagged **12/15 stocks `wide_spread`** — the free `feed=iex` quote is a
  ~2% single venue that routinely doesn't quote at the touch (measured live, market open, quotes 1–5 s old:
  **SNDK 245 bps, TSLA 98 bps, AMD 40 bps vs NVDA 1.0 bps, MSFT 5.6 bps**). Spread on IEX measures IEX, not
  execution cost → `wide_spread` for stocks is now emitted **only on a consolidated feed** (new
  `ALPACA_DATA_FEED`, default `iex`) **and** only during the regular session. (b) `dailyBar.v` on IEX is
  IEX's slice, so the crypto $5M floor falsely flagged RGTI `illiquid` → stock `min_quote_volume=5e5`.
  Also `move_z/volume_z` **3.0 → 2.5** for stocks: a z-score over n=15 is bounded by (n−1)/√n ≈ **3.61**, so
  3.0 was nearly unreachable and would never have fired. Verified: real data → 0 false positives (max |z| =
  1.84, AMD; the semis AMD/MU/SNDK all +6–7% together = a *sector* move, correctly not idiosyncratic);
  synthetic SNDK +22% → `abnormal_move z=3.12`. Finally, `publish()` now filters to OUR universe — the raw
  scan flags **489/702** perps we never trade, which is *why the panel read as crypto-only*; published set
  went 487-crypto/0-stock → **3 crypto / 8 stock**. De-gear lookups still consult the full flag set.
- **Net worth barely moved.** Root cause: `strategy:portfolio` was republished **only inside
  `rebalance_once()`**, i.e. once per `rebalance_seconds`. The frontend polls every 5 s and faithfully
  re-rendered the same daily snapshot. (`.env` had been set to `3600` explicitly as a stopgap "so the curve
  populates faster" — which bought liveness by re-evaluating the **forming 4h TS candle hourly**, see
  below.) Fix: **decoupled marking from rebalancing.** New `mark_and_publish_once()` + `mark_seconds`
  (20 s): re-marks HELD positions at **live** prices (Binance spot for `…USDT`/`…-USD`, Alpaca for stocks;
  `^VIX` skipped) and republishes, without recomputing targets, routing, journaling, or overlay learning —
  those stay on the daily cadence. Curve points throttled to `history_interval_seconds` (300 s) so the
  history doesn't balloon at a 20 s mark rate. Rebalance now reverts to **86400** and marks on the *same*
  live quotes (injected via a new `live_prices=` arg) so it stays continuous with the fast marks — marking
  on an older bar close would saw-tooth the curve and inject that artifact into the journaled
  `book_return` the overlay learns from. `rebalance_once()` stays **pure / offline-testable** (the loop
  injects; it never fetches) — a first attempt that fetched inline made the unit test mark a fake book at
  real **Citigroup** prices, because its synthetic tickers `A`/`B`/`C` are real NYSE symbols (84% phantom
  drawdown). Caught a second real bug in verification: `BTCUSDT` (TS sleeve) and `BTC-USD` (managed sleeve)
  both price off Binance `BTCUSDT`; a 1:1 map silently dropped one, freezing that leg's mark forever →
  now 1:many.
- **Model Performance page showed nothing.** It aggregates the SQL `Trade` table, which is **0 rows** by
  design (`NN_AGENT_ENABLED=false`; the StrategyAgent paper book publishes to Redis + a JSONL journal and
  never writes `Trade` rows) → 16 zeroed cards. Added a **Live Paper Book** summary at the top of the page
  (total value, P&L, current/worst drawdown, realized Sharpe, vol utilization, gross, rebalances/marks)
  fetched from `/api/strategy/portfolio` + `/api/strategy/health`, plus an explicit empty-state on the NN
  grid. Also fixed a **latent 500**: `_sortino` returns `inf` when a bucket has no losing trade, and
  Starlette serializes with `allow_nan=False` → the *whole page* would have gone blank the moment any
  bucket was all-winners. Guarded like `profit_factor` already was.
- **Numbers under "Net Worth Over Time" were fused/messy.** Regression from the 2026-07-08 switch to a
  numeric time axis: recharts `scale="time"` reads the domain as **milliseconds**, but `ts` is Unix
  **seconds**, so d3 compressed the span 1000× and generated ticks seconds apart — after `fmtTick`'s
  `*1000` every label formatted to the same `HH:MM`. Fixed by feeding the axis ms (`ts*1000`) and dropping
  the `*1000` from the tick + tooltip formatters.
- **"Model is bad at trading."** Framing first: at ~30% vol a −1.0% wobble over 5 days is inside noise
  (book was $98,995, peak +0.83%, worst −1.23%). Ruled OUT: half-risk sizing (book vol-target is working —
  gross 0.606 de-levered from ~50% vol to hit 30%), sign flips, non-persistence, huge financing (paper
  charges 5 bps/rebalance and **no** funding at all → paper is *optimistic*), live NN. Two genuine bugs
  fixed: (1) **forming-bar repaint** — `.env` rebalanced hourly while `Binance4hBarProvider`'s last row is
  the *in-progress* 4h candle, so a Donchian breakout could trigger intrabar and close back inside the
  channel (a false entry the closed-bar backtest never saw), corrupting the entries of the β≈0 diversifier
  sleeve. `_get_ts_bars` now drops the forming candle (`df.iloc[:-1]`) — robust at *any* cadence — and the
  rebalance is back to daily. (2) **stale zero-marks** — when a price fetch failed/cached, every held
  symbol contributed 0, `book_return=0.0` was journaled, and `_after_mark` fed those fabricated flats into
  the overlay's online conviction loop; **~35–40% of journal marks were exactly 0.0** (6 consecutive frozen
  on 07-08). `mark_to_market` now sets `last_mark_repriced`; stale marks record no curve point and are not
  learned from (this is also why the curve *looked* permanently red: freeze, then gap down). Housekeeping:
  the StrategyAgent got its **own** `RiskManager` seeded from `STRATEGY_AGENT_EQUITY` (it was sharing the
  one seeded from the $1000 NN book — a halt on that phantom book would silently freeze the real book), and
  the news overlay no longer records/logs de-gears on flat (`|w|≈0`) targets, which were pure no-ops
  flooding the journal.
- **Still open (owner's call, NOT changed):** the news overlay is a *structurally one-sided* drag — it
  de-gears on opposing news but almost never upsizes (needs conf ≥0.7 AND horizon ≤240 min), and it shaved
  the held ETH long on **38 of 49 fires (mean scale 0.920, min 0.790)** atop a signal this project already
  measured at ~0 directional IC. Options: A/B `STRATEGY_AGENT_NEWS_OVERLAY=false`, or require
  `severity=="SEVERE"` before the resize branch. Also: `trade_journal.jsonl` is contaminated by unit-test
  fixtures (orders for symbols `"A"`/`"B"`, notional 66622.23, `decision_price: null`) that the
  deterministic post-mortem reads — tests should write to a temp journal.

**Owner:** RESTART the backend to activate (all of the above are backend-process changes except the two
frontend files). `.env` now: `STRATEGY_AGENT_REBALANCE_SECONDS=86400`, `_MARK_SECONDS=20`,
`_HISTORY_SECONDS=300`.

### 2026-07-09 (part 2) — INCIDENT: the test suite clobbered the live paper book

**What happened.** Running `pytest` (while the owner's backend was up) destroyed the live paper book:
equity **98,995.08 → 15,698.61 (−84.3%)**, phantom positions `A`/`B`, `last_prices` holding
`A=131.72`, `B=36.49`, `C=63.888` — i.e. real **Agilent / Barnes / Citigroup** quotes.

**Mechanism (three latent faults lining up).**
1. `redis_client.get_redis()` pings the **REAL** Redis first (`main.py` auto-starts one on :6379) and only
   falls back to FakeRedis when that ping *fails*. With the backend up, tests hit production Redis.
2. `StrategyAgent.rebalance_once()` ended with `await self._publish_portfolio()` — i.e. Redis I/O inside
   the method whose own docstring promises it is "unit-testable with fakes and free of redis/db/broker
   coupling". `test_strategy_agent.py` drives it with a `PaperBook(equity=100_000)` over the synthetic
   universe `A`/`B`/`C`, so the test's book was written straight to `strategy:paperbook:state`.
3. Those synthetic tickers are **real NYSE symbols**. A transient version of `_live_prices` (since reverted)
   was briefly called inside `rebalance_once`, so the fake book got marked at real Citi/Agilent prices → the
   84% wipe. The backend then restarted (~15:15), `_restore_paper_book()` happily adopted the wreck, and the
   agent kept rebalancing its real universe on a $15.7k equity.
   (The pre-existing `A`/`B` rows in `trade_journal.jsonl` from 07-07, flagged earlier as "test fixtures
   leaking into the production journal", were the same bug — a warning that went unheeded.)

**Fixes (defence in depth).**
- `rebalance_once()` no longer publishes. `run()` publishes after it returns. Pinned by a regression test
  (`test_rebalance_once_never_touches_redis`) that monkeypatches `get_redis` to raise.
- `_restore_paper_book()` **rejects** a persisted state holding positions outside `universe ∪ ts_universe`
  (would have blocked this exact poisoning), and now **consumes** `paper:reset_requested`. Nothing else
  cleared that flag once `NN_AGENT_ENABLED=false` (only `nn_agent` did) — which is why
  `_reset_paper_state()` had resorted to a fragile `ex=60` TTL; a sticky flag would wipe the book on
  *every* restart.
- `tests/conftest.py` now isolates the whole suite **before** `backend.core.config` imports: `REDIS_URL`
  → closed port :6399 (forces the FakeRedis fallback), `DATABASE_URL` → temp sqlite, plus an autouse
  fixture pinning `redis_client._fake_redis_instance` (checked *first* by `get_redis()`, so it also covers
  modules that did `from ... import get_redis`) and redirecting `trade_journal.DEFAULT_PATH` to a tmpdir.
  New `tests/test_store_isolation.py` asserts all of this. **Verified empirically**: real Redis is
  reachable, yet a canary written through `get_redis()` during a full suite run never appears on :6379.

**Repair.** Archived the poisoned journal (746 rows → `trade_journal.poisoned-20260709T224447.jsonl`),
truncated the live one, deleted `strategy:{paperbook:state,portfolio,status}`, and set
`paper:reset_requested=true`. Next backend start seeds a clean book at `STRATEGY_AGENT_EQUITY = $100,000`.
Discarded history: 78 healthy marks, 07-04→07-09, ending **−1.0%** (peak +0.83%, worst −1.23%) — preserved
in the archive. **640 tests pass.**

**Lesson:** any method a unit test can call must not reach a shared store, and a fallback that only
triggers "when the real thing is unreachable" is not isolation — it is a coin flip on whether prod is up.

### 2026-07-26 — MEASUREMENT CONTRACT established (owner-directed adversarial review, Phase 0)

Owner set a hard objective: **ARR >=20%** with **P(month-over-month decline) <= 5-10%**, and required a
binding measurement contract before any further analysis. Contract below governs every subsequent claim.

**The contract.** Sharpe = *monthly* arithmetic **excess** returns, net of modelled costs, rf = `^IRX`
(13-week T-bill, daily, ffilled, /252), annualised `mu/sigma * sqrt(12)`. Monthly frequency chosen
deliberately: it matches the binding constraint, and it dissolves the PPY ambiguity below. Every Sharpe
must carry a 95% CI (Lo 2002 eq.9 with Mertens skew/kurtosis terms). Lo autocorrelation correction:
APPLIED on samples >=10y; on the 53-month window reported as a *band*, since rho1 = -0.109 has SE 0.137.
Acceptance for any proposal: purged walk-forward + deflated Sharpe at the honest trial count + cost sweep
at 1x/2x/4x. Fails any one => labelled speculative.

**Three measurement defects found (all inflate or obscure the headline):**
1. **rf = 0 everywhere.** `grep -riE "risk_free|riskfree|rf_rate|excess_ret"` over the whole backend:
   **zero hits**. Over 2022-26 cash paid **3.97%/yr**. Managed-beta sleeve Sharpe **0.655 -> 0.418**
   (-36%). Vol-matched combined book **0.80 -> ~0.56**. Roughly a third of the headline is T-bill yield.
2. **Annualisation factor mismatched to observation frequency.** `align_panel(how="outer")` puts crypto
   weekend bars in the index: **22.4% of rows are Sat/Sun**, so the series runs **365.5 obs/yr** on the
   2022-26 window while `PPY = 252`. meta.md's "1611 trading days" is 1612 *calendar* days = **4.41 yr,
   not 6.4 yr**. Direction is conservative for Sharpe (understates ~1.20x) but it corrupted the sample
   length, hence every CI and MinTRL. Monthly estimation makes it moot.
3. **`ann_ret = mean * 252` is printed as "CAGR"** (`combined_book_research.py:188`). True CAGR is lower
   by variance drag: 10.95% arith vs **10.04%** geometric on 2022-26.

**Reproducibility gap.** The return series behind every number in S12 is **never persisted** (no
`to_csv`/`to_parquet` in the research scripts) and the parquet price cache
(`training_data/equity_daily/`) was **empty** — 0 files. The entire results table was unfalsifiable from
the repo as committed. Repopulated the cache (29 symbols, yfinance, 2010-01-01) and **reproduced
S5.9's managed-beta 0.65 exactly (0.655)**, which validates the pipeline. The 4h crypto sleeve still
cannot be rebuilt (no Binance kline cache), so the *combined* book remains unverified end-to-end.

**Measured, managed-beta sleeve, monthly excess (n=53 months, 2022-01..2026-05):**
| quantity | value |
|---|---|
| annualised SR | **+0.432**, 95% CI **[-0.41, +1.27]**, t = 1.01 |
| monthly skew / exkurt | **+1.71** / +5.03 (Jarque-Bera p<0.0001) |
| **P(monthly excess < 0)** | **52.8%** (owner needs <=5-10%) |
| MinTRL to reject SR=0 @95% | **14.5 yr** 1-sided / 20.6 yr 2-sided |
Full 2010-26 sample (n=199 months): SR **+0.846** CI [+0.44, +1.25], t=4.06, P(mth<0)=45.2%.

**Required return/vol ratio for the owner's constraint — all three distributions:**
| P(mth<0) | Gaussian | Student-t(4), unit-var | EMPIRICAL (2022-26) |
|---|---|---|---|
| 0.05 | **5.70** | 5.22 | **4.07** (bootstrap CI 2.94-4.91) |
| 0.10 | **4.44** | 3.76 | **3.26** (bootstrap CI 2.62-4.53) |
**The Gaussian assumption is NOT flattering the owner — it is penalising him.** Two reasons: standardized
t(4) is *less* extreme than Gaussian at the 5-10% quantile (crossover is ~2.5 sigma, further out than the
constraint bites), and the empirical monthly distribution is **positively** skewed. Caveat that matters:
that positive skew is partly *manufactured* by the in-sample-tuned `drawdown_degear` truncating the left
tail, plus TQQQ/crypto convexity — exactly the property that decays OOS. And the alpha=0.05 empirical
quantile rests on the **~2.7th order statistic of 53 months**; its CI spans nearly 2:1. Not admissible
as a point estimate.

**The multiple-testing ceiling — the decisive number.** Logged trials in this repo alone: 588 TA + 120 +
50 + 6 + 26 table rows ~= **>=790**, true count higher (unlogged param variants). Under the null of zero
skill, E[max of N=800 trial Sharpes] = **3.19 SEs** (analytic Bailey-Lopez de Prado; Monte-Carlo 20k reps
confirms 3.18) = **SR_ann 1.37** on a 53-month sample. Every candidate ever produced — mb 0.43, mb rf=0
0.655, combined book 0.80, full-sample 0.846 — sits **below the Sharpe that pure selection noise
manufactures at this trial count.** No result in S12 has cleared its own search cost.

**Verdict for all later phases.** Required SR 4.44-5.70 (Gaussian) or 3.26-4.07 (empirical, wide CI).
Best honest measured SR **0.43, CI containing zero**. Gap ~8-10x, and *un-leverageable* — leverage moves
CAGR and vol together, leaving SR fixed (already proven in S8's leverage-invariance sweep). MinTRL 14.5y
vs 4.4y of selection-contaminated backtest and **18 journal records (~17 days) of paper**. The 20% ARR
target is reachable via vol-target/financing; the **P(monthly decline)<=5-10% constraint is not reachable
with any strategy in this repo**, and must be renegotiated or the objective rebuilt around a
fundamentally higher-SR source. Stated plainly at the top of every subsequent phase.

### 2026-07-26 (part 2) — PHASE 1: adversarial attack on the combined book (owner: "break the combined book")

Rebuilt BOTH sleeves independently (native Binance 4h klines via `data-api.binance.vision`, 21 symbols,
9667 bars each, 2022-01→2026-05 — bypasses the 5m-zip path; same OHLCV, same 00:00-UTC boundaries) and
re-ran the book under the 2026-07-26 measurement contract.

**REPRODUCTION GATE — PASSED EXACTLY.** mb **0.651** (S5.9: 0.65), ts **0.521** (0.52), combined
**0.796** (0.80), corr(mb,ts) **+0.098** (0.098). The book is real and the numbers below attack the
actual deliverable, not a strawman. Series persisted to `combined_book_returns.csv` — the first time
this book's returns have existed on disk.

**WHAT I COULD NOT BREAK (stated plainly — a real attempt was made):**
1. **No look-ahead anywhere in the risk stack.** `vol_target_scale` shifts leverage by one bar;
   `drawdown_degear` computes dd from pre-bar equity; `_managed_net` applies `pos[t-1]` to `ret[t]`;
   `_causal_equal_weights` shifts weights; `_to_calendar` compounds weekend crypto FORWARD into the
   next calendar row (never backward). All genuinely causal. This machinery is sound.
2. **The ADX gate is NOT a tuned spike.** adx_min 0→25 is a flat plateau (SR +0.288→+0.378), degrading
   only above 28. Not the overfit knob I expected to find.
3. **`drawdown_degear` does NOT manufacture the positive skew.** degear on/off is bit-identical
   (it is inert, as S8 already noted). **My Phase-0 caveat blaming degear for the +1.71 skew was WRONG
   and is retracted.** The left-tail truncation comes from ManagedBeta's Faber 200-EMA cash-below-trend
   rule, not the outer overlay.

**WHAT BROKE:**
1. **COST SENSITIVITY — FAILS the contract.** TS sleeve SR: **+0.297 (1x) → +0.007 (2x) → −0.603 (4x)**.
   Breakeven **2.0x = 30bps/side**. Sign is not stable ⇒ speculative by the contract's own rule.
2. **The slippage ledger is EMPTY OF REAL DATA.** All 114 rows of `training_data/slippage_log.jsonl`
   have `decision_price == fill_price == 100.0` (0 real fills) yet are tagged `status:"live"`. The
   ledger built specifically to validate the 7.5-15bps assumption **validates nothing**, and the sleeve
   dies at 2x that assumption. 30bps/side across 21 perps incl. ALGO/FIL/RENDER is not a stretch.
3. **The ADX-gate rescue narrative is FALSE.** The audit credits the gate with saving the sleeve
   (+0.94 with / −0.83 without). Measured: **no gate +0.528 vs gate +0.521 — the gate is worth −0.007.**
   The rescue came from the **1h→4h timeframe change**, made at the same time. Two changes, one
   credited. And 25 isn't optimal — adx_min=20 gives +0.378 vs +0.297.
4. **REGIME CONCENTRATION.** Fold 2 (2023-02→2024-03, 14mo) alone = SR **+1.036**. The other 39 months
   = SR **+0.211**. One window carries the book.
5. **DEFLATED SHARPE FAILS AT EVERY TRIAL COUNT** (contract/excess series): N=1 **0.852**, N=10 0.298,
   N=100 0.069, **N=810 → 0.0158**. Even pretending this was the only idea ever tested, it fails.
6. **P(monthly<0) = 56.6% for the COMBINED book — WORSE than either sleeve** (mb 52.8%, ts 49.1%).
   *The structural insight:* diversification raised Sharpe but LOWERED the monthly win rate, because
   the book's positive skew (+1.17) pushes the median below the mean. **For a constraint on
   P(monthly decline), positive skew at fixed Sharpe is the ENEMY** — it is exactly what made the
   Phase-0 "empirical required ratio" look easier (4.26 vs Gaussian 5.70) while making the realised
   outcome worse. Both facts are the same fact.

**BUG (new):** `managed_beta_daily` calls `vol_target_scale(combined, 0.15, PPY=252)` on a ~365-row/yr
calendar → annualised vol is understated by √(365/252)=1.20 → leverage overstated by the same factor,
so the book runs materially hotter than its stated 15% target (measured ann vol 16.7% at √252).

**VERDICT.** The book is honestly built and honestly measured — and it is not an edge. Contract SR
**+0.461, 95% CI [−0.40, +1.33]**, DSR 0.016, cost-fragile at 2x, regime-concentrated in one 14-month
window, realised P(monthly<0) **56.6%** against a ≤5-10% requirement. Gap to the owner's constraint:
**9.6x (at 10%) to 12.4x (at 5%)**. Nothing found here is fixable by tuning; the failure is that
managed beta + a cost-fragile crypto trend sleeve is a ~0.5-Sharpe object and the objective needs ~4.4.

### 2026-07-26 (part 3) — PIPELINE AUDIT + CORRECTED BASELINE (owner: line-by-line inflation audit)

**HEADLINE: 58% of the reported Sharpe was risk-free omission, a calendar bug, and cost optimism.
Corrected SR = +0.332, 95% CI [−0.57, +1.23]. The interval contains zero.**

**CONFIRMED #1 — THE MONDAY BUG (the one real code defect; `portfolio.py:28` + `portfolio.py:63`).**
`align_panel(how="outer")` reindexes every symbol onto the UNION calendar and NaN-pads. With crypto in
the universe the union contains Sat/Sun, so equities are NaN there. `_bar_returns` then computes
`close[Mon]/close[Sun] = x/NaN → NaN`, and line 67 `np.nan_to_num(r, nan=0.0)` silently turns it into
**0.0**. Measured on SPY 2010-26: **621 of 839 Mondays zeroed (74%)**, 36.5% of all SPY bar-returns
exactly 0.0. The 218 survivors are all pre-2014-09, before BTC-USD joins the index — so across the
entire 2022-26 study window **every Friday→Monday equity gap return is deleted.** That is precisely
where weekend/gap risk lives, so the bug understates vol AND drawdown. Fixing it moves the managed-beta
sleeve's maxDD from **−27.0% to −39.9%** and the combined book's from −13.8% to −19.5%. Sharpe cost:
**−0.085**. Root cause is `nan_to_num` treating "no data" as "no return"; absence must propagate, not
be coerced to zero.

**CONFIRMED #2 — rf = 0** (no `risk_free` anywhere in the backend). Cost: **−0.335**, the single
largest component. **CONFIRMED #3 — cost realism**: flat 5bps/15bps with no size scaling, no financing
on the levered portion, no perp funding. Replacing with half-spread + Almgren √-impact + 5% financing +
10%/yr funding costs **−0.044** at $1M (the flat 15bps was actually *conservative* at small size — and
wildly optimistic at scale, see capacity). **CONFIRMED #4 — survivorship**: all 29 MB + 21 crypto names
hand-picked in 2026; no bankrupt/acquired/delisted-to-zero names with terminal returns. **CONFIRMED #5
— restated prices**: `equity_daily.py:105` `auto_adjust=True` → dividends retroactively rescale all
prior closes, so the 200-EMA trend filter sees a series no live trader could have seen (small).
**CONFIRMED #6 — vehicle mismatch**: TS sleeve runs `allow_short=True` on SPOT klines; shorting spot
needs borrow, live uses perps → funding. Now charged. **CONFIRMED #7 — no market impact / capacity**.

**RULED OUT (a real attempt was made):** one-bar offset IS applied (`_managed_net` / `strategy_net_returns`
use `pos[t-1]·ret[t]`; both `_positions_one` are fully determined by data through `close[t]`); no
centred windows, no `shift(-n)`, no backfill, no `center=True`, no full-sample scaler/quantile/PCA
anywhere in the deliverable path; Donchian channels are `.rolling().max().shift(1)`; `_to_calendar`
compounds weekend crypto FORWARD only; stops are evaluated on the CLOSE, not intrabar, so the backtest
never assumes a stop-price fill and eats gaps in full (conservative); `down_exposure=0` ⇒ no equity
shorts ⇒ no borrow cost owed. Fundamentals/index-membership: N/A, not used by this book.

**CORRECTED BASELINE — the reference point for every later phase:**
| | reported | corrected |
|---|---|---|
| Sharpe (contract) | 0.796 (rf=0) | **+0.332  CI [−0.57, +1.23]** |
| CAGR / vol | — | +8.2% / 12.7% |
| maxDD | −13.8% | **−19.5%** |
| P(monthly<0) | 56.6% | **50.9%** (target ≤5-10%) |
Annual: 2022 **−11.2%**, 2023 +23.6%, 2024 +15.5%, 2025 +5.2%, 2026ytd +5.8%; total +41.0%/1100 days.

**ATTRIBUTION — Sharpe from bugs/bias vs real edge:** 0.796 → −0.335 (rf) → −0.085 (Monday bug) →
−0.044 (cost realism) = **+0.332**. **58% of the headline was artifact.** What remains is not
distinguishable from zero (t≈0.7), fails DSR at every trial count, and is regime-concentrated.

**CAPACITY CEILING (Almgren √-impact, γ=1.0):** SR by AUM — $0.1M +0.388, $1M +0.332, $5M +0.233,
**$20M +0.055**, $50M −0.147, $100M −0.371. **Alpha is fully consumed by impact at ≈$20M.**

### 2026-07-26 (part 4) — IS THE CORRECTED RESULT DISTINGUISHABLE FROM LUCK? (owner-directed)

Built an 864-variant grid on the CORRECTED pipeline (8 managed-beta configs x 36 TS configs x 3 blend
weights, 1104 days) — the search space that plausibly produced the shipped book — and ran the full
overfitting battery against it. Shipped config = `mb200_0.0|ts48_3.0_25|w0.5`, SR +0.402 on this grid's
cost basis (spread+fee, no size-aware impact; the $1M size-aware baseline is +0.332).

**VERDICT: P(edge is real) ~ 10-25%. P(edge meets the owner's constraint) = 3e-19.**

| test | result | verdict |
|---|---|---|
| **DSR** (corrected: SR0 = sigma_ACROSS-TRIALS x E[max_N], denom = SE(SR)) | 0.42 @ sigma=0.15 → **0.004** @ sigma=0.50 | FAIL at every count |
| **PBO** (CSCV, S=16, 12870 combos) | **75.0%** | SEVERE — selection worse than random |
| **MinBTL** @ N=810 | **63.2 years**; have 4.38 | FAIL by 14x |
| **White's Reality Check** | p = **0.119** | cannot reject "no edge" |
| **Hansen's SPA** | p = **0.664** | cannot reject "no edge" |
| **Purged 5-fold, 200d embargo** | 4/5 folds +, mean +0.290, sd 1.016, all CIs span 0 | inconclusive by construction |
| **DoF** | **76 free params / 5.5 independent 200d blocks = 13.8:1** | catastrophic |

**DSR METHOD FIX (my own error, corrected).** Parts 2-3 used the single-trial SE as sigma_SR. Bailey-LdP
require the **cross-trial SD of Sharpes**. Measured across the 864 variants actually run: **0.142 ann**.
That is much SMALLER than the single-trial SE because the variants are highly correlated, so the honest
DSR is **0.42, not 0.016**. But 0.142 understates true search diversity — the project's real search
spanned TA zoo / stat-arb / factors / XS-momentum / crypto with Sharpes from −0.83 to +4.65, so
sigma_SR is realistically 0.3-0.5, giving **DSR 0.11 → 0.004**. Range reported honestly; all fail.

**PBO is the most damning number** because it is non-parametric and needs no sigma_SR assumption. The
IS-best variant lands at the **27.5th percentile OOS** (median), and degrades **IS +1.101 → OOS +0.435**.
Picking the in-sample winner is *worse than picking at random*.

**PARAMETERS SIT ON A BROAD PLATEAU, NOT A SPIKE — and that is not the good news it sounds like.**
All 864 variants are positive (min +0.095, mean +0.451, sd 0.142, max +0.836). So the book is NOT a
curve-fit spike. But the shipped config sits at the **38th percentile of its own grid** — it was never
even well-tuned — and the whole plateau has CIs containing zero. A broad plateau of
indistinguishable-from-zero means the parameters don't matter *because there is nothing to tune*. The
result is **robustly mediocre**, not fragilely good. Best cells: ts_entry=96 (0.475-0.586 vs 48's
0.274-0.494) and mb_ema=100 (0.541/0.547); nobody would have chosen the shipped cell on merit.

**REGIME DEPENDENCE — the book is a long-volatility structure.** By vol tercile of its own trailing
63d realised vol: low **−1.47bp/day**, mid **−0.39bp/day**, high **+7.82bp/day** (SR −0.190 / −0.042 /
**+0.803**). *All* the return comes from high-vol regimes. By trend: SPY uptrend +0.596, downtrend
−0.381 (so the "steps out of downtrends" claim does NOT show up as downtrend outperformance). By year:
2022 **−1.136**, 2023 +1.017, 2024 +0.626, 2025 +0.227, 2026 +1.111. Drop 2023 → +0.160; drop the best
5% of months → **−0.116** (caveat: dropping the top 5% hurts any positive series — it is the magnitude
plus the +0.83 monthly skew that makes it notable, not the sign flip alone).

**WHY PURGED CV CANNOT EXONERATE THIS.** Naive k-fold is invalid here — (a) the 200d EMA means a test
day's signal is built from ~200 days sitting in TRAIN, (b) positions are path-dependent (trailing stop
+ trend state cross any cut), (c) returns are serially dependent within a held trade, (d) regimes are
persistent so random folds leak future regime. But even done properly, **nothing is fitted per fold** —
the parameters were chosen after seeing all folds. Purged CV therefore tests regime STABILITY, not
leakage; the leakage is at the SELECTION level and only CSCV/PBO measures it. And with a 200d lookback
over 1104 days there are only **~5.5 independent blocks**; a 5-fold scheme with a full 200d embargo
would consume 91% of the sample. Purged CV is close to impossible on this data.

**KILL LIST:** ADX gate (worth −0.007, Phase 1); the specific shipped parameter values (38th pct of
their own grid — no basis for preferring them); TS sleeve as a standalone edge (dies at 2x costs, cost
assumption backed by a ledger with 0 real fills); DMN overlay (+0.37 "timing skill" = 2 trials on ONE
window with a saturated tanh — worthless next to PBO 75%; do NOT enable); lead-lag (already net −0.85);
any claim resting on 2023. **SURVIVES:** sleeve diversification (corr +0.09 is structural, not fitted,
and doesn't require either sleeve to have alpha) and vol-targeting (a theorem about leverage, not an
edge). Neither is alpha.

### 2026-07-26 (part 5) — "what is our Sharpe, and how do we get to 1.6?" (owner Q)

**CURRENT LIVE SHARPE = UNDEFINED. The agent has not traded.** `trade_journal.jsonl` holds 18 records,
ALL from a single day (2026-07-09: two rebalances 12 minutes apart, 16 orders), equity flat at exactly
100000.00, **0 marks**. `postmortem_20260710` and `postmortem_20260715` both report `n_marks: 0`,
`window_return: 0.0`, `per_mark_vol: 0.0`. Nothing has happened since 2026-07-09 — **17 days of no
activity**. The 2026-07-09 "RESTART needed" was never done, or the mark loop is dead. The ONLY Sharpe
this project has is the backtest: **+0.332, CI [−0.57, +1.23]**.

**WHY 1.6 IS UNREACHABLE — the fundamental law, with your own measured inputs.**
`SR_book = s*sqrt(N)/sqrt(1+(N-1)*rho)`, ceiling as N→inf is **s/sqrt(rho)**.
At your measured s=0.33, rho=0.09: **ceiling = 1.10**. No number of sleeves reaches 1.6.
| rho | ceiling @ s=0.33 | N needed for 1.6 |
|---|---|---|
| 0.09 (measured) | **1.10** | **IMPOSSIBLE** |
| 0.05 | 1.48 | IMPOSSIBLE |
| 0.03 | 1.91 | 78 sleeves |
| 0.00 | inf | 24 sleeves |
To hit 1.6 at rho=0.09 every sleeve must be a **0.48-Sharpe standalone**; best ever measured is 0.33.

**LEVER TESTED AND CONFIRMED — idle-cash yield, ~+0.07 to +0.14 Sharpe, FREE.** The managed-beta sleeve
holds **45.3% average idle cash** (trend filter → cash below 200-EMA) which earns **zero** in the
backtest. At the window's 3.97% T-bill that is 1.80%/yr uncredited on the sleeve; at the book's 12.7%
vol, +0.142 Sharpe sleeve-level / ~+0.07 book-level. Pure accounting, no risk added. Needs
implementation-level cash-sweep accounting to bank; the range reflects that, not a point estimate.

**LEVER TESTED AND REFUTED — vol-targeting direction.** Hypothesis from part 4: the book earns
+7.82bp/day in high vol and −1.47bp/day in low vol, while vol-targeting sizes ~1/vol, i.e. largest in
the LOSING regime. Tested pro-vol sizing (scale ∝ vol, clipped 0.5-2.0, shifted): **SR 0.292 vs 0.332
shipped — WORSE. Hypothesis rejected.** (Note: the "flat sizing" arm of that test was vacuous — I
multiplied an already-vol-targeted series by 1.0. Only the pro-vol arm is informative. Properly
removing the overlay requires a rebuild.)

**LEVER PLAUSIBLE, MECHANISM SOUND, MAGNITUDE UNVALIDATED — turnover reduction.** Part-4 surface shows
ts_entry=96 at 0.475-0.586 vs entry=48 at 0.274-0.494. Mechanism is arithmetic (SR=(mu−c)/sigma; lower
turnover ⇒ lower c, sigma unchanged) and is NOT exhausted — Phase 1 showed the TS sleeve dies at 2x
costs, so cost is a first-order term. But the specific magnitude is in-sample selection on my own grid.

**EXHAUSTED / WILL NOT WORK:** more indicators or ML (810+ trials, IC≈0); leverage (SR-invariant, S8);
more names in the SAME asset classes (S: broad-12 DILUTED the book 1.08→0.94).

**HONEST CEILING:** with cash-yield credited and turnover cut, **~0.45-0.55** is defensible for this
architecture. 1.0 would need rho≈0.03 across 10+ genuinely different books. **1.6 net of costs on
liquid retail-accessible trend+beta: I do not know how to get there and do not believe it exists at
this scale.** META-POINT: every test run adds to the trial count — this session alone added 864+.
The search is itself the adversary; DSR falls with each new idea tried.
