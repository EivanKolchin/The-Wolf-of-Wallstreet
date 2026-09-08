# The Wolf of Wallstreet

A self-hosted systematic trading system: a Python backend that researches, backtests and paper-trades a
multi-asset book, and a Next.js dashboard to watch it. It runs on a laptop against free or cheap data
(Binance, Alpaca, yfinance, SEC EDGAR), and it is wired for live execution but ships in paper mode.

It is also a research record. The project started as a neural network that tried to predict short-term
price direction. That did not work. The reasons it did not work are written down below with the numbers
attached, because the negative results took most of the effort and are the most useful thing in the
repository. If you only read one section, read [The research record](#the-research-record).

**Status:** paper only, run on demand. No live capital has ever been routed. The models are not trained
beyond short probe runs, by choice.

## Contents

- [What this is](#what-this-is)
- [The short version of the result](#the-short-version-of-the-result)
- [How it works](#how-it-works)
- [Repository layout](#repository-layout)
- [Running it](#running-it)
- [Configuration](#configuration)
- [Paper mode and live mode](#paper-mode-and-live-mode)
- [The research record](#the-research-record)
- [Where the project is today](#where-the-project-is-today)
- [Known problems and gotchas](#known-problems-and-gotchas)
- [Lessons](#lessons)
- [Disclaimer](#disclaimer)

## What this is

Three things live in this repository:

1. **A research stack.** Vectorised backtesters, a walk-forward harness, a strategy library, a feature
   pipeline shared byte-for-byte between offline training and live inference, and the statistics needed
   to not fool yourself: deflated Sharpe, purged and embargoed splits, and alpha/beta regression against
   the benchmark. The heavier overfitting battery (probability of backtest overfitting, White's reality
   check, Hansen's SPA) was run as a one-off audit rather than committed; its results are below.
2. **A live paper-trading engine.** A FastAPI service that holds target weights for a two-sleeve book,
   marks it against live prices, applies a risk stack, and routes orders to a paper book or, if you
   supply keys and flip two flags, to real brokers.
3. **A dashboard.** Next.js 14, reads the backend over REST, shows the book, positions, news,
   performance and an audit trail.

It is not a product and not a beginner's tool. It assumes you can read a backtest critically and are
comfortable with the idea that most of what it measured came back negative.

## The short version of the result

We tested direction prediction, classical technical analysis, cross-sectional and time-series momentum,
statistical arbitrage, mean reversion, funding carry, daily price factors, fundamental factors, lead-lag
networks, meta-labelling and a gradient-boosted meta-layer. Against a multiple-testing-honest bar,
essentially none of it produced alpha at retail scale on public data.

What did survive:

- **Trend-filtered beta harvesting.** Holding a curated multi-asset basket only while each asset is above
  its 200-day trend, and sitting in cash otherwise, beat buy-and-hold on every risk-adjusted metric we
  measured. That is not alpha. It is beta, managed.
- **Diversification.** Two mediocre and nearly uncorrelated sleeves (correlation +0.10) beat either one
  alone. The benefit is structural rather than fitted, which is why it is the piece we trust most.
- **The risk machinery.** Volatility targeting, drawdown de-gearing and a funding-aware leverage cap
  moved the return-per-unit-risk number more reliably than any signal we built.

And the honest caveat on the headline number. The two-sleeve book reported a Sharpe of 0.796 on the
2022 to 2026 window. A line-by-line audit found that 58% of that was artifact: no risk-free rate was
subtracted (worth 0.335), a calendar bug silently zeroed 74% of Monday equity returns (0.085), and the
cost model ignored financing and funding (0.044). The corrected figure is:

| | reported | corrected |
|---|---|---|
| Sharpe (monthly excess, net of costs) | 0.796 | **+0.332, 95% CI [-0.57, +1.23]** |
| CAGR / volatility | | +8.2% / 12.7% |
| Max drawdown | -13.8% | **-19.5%** |
| P(monthly return < 0) | 56.6% | 50.9% |

The confidence interval contains zero. A full overfitting battery on the 864-variant grid that plausibly
produced this configuration put the probability of backtest overfitting at 75%, meaning the in-sample
winner lands below the median out of sample. Minimum backtest length to justify the result is 63 years
against the 4.4 years of data available. The honest reading is that this book is robustly mediocre
rather than fragilely good, and that its remaining edge is not distinguishable from luck.

Every Sharpe quoted further down is annualised, walk-forward, and computed with a risk-free rate of zero
unless it is explicitly labelled "corrected". Subtract roughly 0.3 to compare against cash.

## How it works

### The runtime

`trading-agent/backend/main.py` starts a FastAPI service on port 8000 and forks two worker processes
that share a severe-event flag:

- **`NNTradingAgent`**, the original neural policy trader, now a misnomer. It is gated behind
  `NN_AGENT_ENABLED`, which is `false` in the configuration this project actually runs. With it off, the
  process skips the model entirely and instead hosts the rule-based `StrategyAgent`, which is the part
  that trades, along with the macro feed, the tick logger and the post-mortem scheduler. Note that
  `.env.example` still ships `NN_AGENT_ENABLED=true`, so a fresh clone starts the untrained model path.
  Set it to false.
- **`LLMNewsAgent`**, which polls RSS, classifies headlines with an LLM and publishes impacts.

Around them the service runs an async macro feed, an optional L1 tick logger, a websocket updater for
the dashboard and a post-mortem job on a 24 hour timer. Before any of that it does a set of pre-flight
steps that matter more than they sound: a DNS check, a detached watchdog process, an auto-start for
Redis (a portable Windows build if Docker is not present) and Ollama, and a probe of port 8000 that
kills a stale backend, because otherwise a new process starts, fails to bind, and the dashboard quietly
keeps talking to the old one.

Redis carries cross-process state (book snapshots, recent news, heartbeats, anomaly flags) and is where
the paper book actually lives. SQLite via SQLAlchemy holds the slower-moving records: news predictions,
source trust scores, keyword weights, agent events. Append-only JSONL under `trading-agent/statements/`
and `trading-agent/training_data/` holds the trade journal, the API and LLM call logs and the slippage
ledger.

### The book

Two sleeves, combined 50/50 by default in `backend/strategies/combined_book.py`:

- **Managed beta** (`managed_beta.py`, `managed_book.py`). Hold each asset in a curated basket (SPY, QQQ,
  TQQQ, TLT, GLD, BTC, ETH plus optional single names) while it is above its 200-period EMA, hold cash
  below it, with a dead band to stop the signal flapping around the line. Tactical asset allocation in
  the Faber sense. Testing a short leg below trend made it monotonically worse, so exposure below trend
  is zero, not negative.
- **Time-series momentum** (`ts_momentum.py`). A 4h Donchian breakout on crypto perpetuals with an EMA
  trend filter, an ADX regime gate and an ATR chandelier trailing stop. Long and short, beta near zero.

`StrategyAgent` (`backend/agents/strategy_agent.py`) turns those into target weights once a day,
reconciles them against the current book, skips deltas below a dead band, and routes the rest. A second
faster loop (20 seconds) marks held positions at live prices and republishes the portfolio, which is
deliberately separate from rebalancing: marking often and trading rarely is the point. `rebalance_once`
is kept pure, with no network calls, because the test fixtures use tickers `A`, `B` and `C` and those
are real NYSE symbols.

### The risk stack

All causal, applied in order, in `backend/risk/` and the portfolio layer:

1. **Volatility targeting** per sleeve, then a book-level re-target against the book's own realised
   volatility. This is the main lever on return, and it is a sizing decision rather than a prediction.
2. **Drawdown de-gearing.** Beyond a threshold below the running peak, scale the next bar down toward a
   floor and restore as equity recovers.
3. **Funding-aware leverage cap** (`funding_cap.py`). Perpetual leverage is financed by funding, so a
   levered position on the paying side gives back what leverage bought. An EMA-smoothed (about 3 day
   half-life), sign-aware, continuous ramp de-levers those names. Continuous rather than threshold-based
   so there is nothing to whipsaw around.
4. **News and macro overlay** (`news_overlay.py`). A tighten-only multiplier. It can shrink, flatten or
   halt an asset, never enlarge beyond a small capped upsize, and an optional LLM verifier is wired so
   that it can only tighten further.
5. **Hard limits** in `RiskManager`: max drawdown, max daily loss, position size, trade frequency,
   minimum notional. The agent aborts a rebalance while halted.

### Data

- **Crypto:** Binance klines at 5m, 1h and 4h, funding-rate history, and book ticker for the tick logger.
  Historical bulk data comes from the public Binance Vision archive, so no key is needed.
- **US equities and ETFs:** Alpaca. On the free tier this is the IEX feed, roughly 2% of consolidated
  volume, so quoted spreads and daily volume are single-venue figures and must never be read as absolute
  liquidity.
- **Daily bars for research:** yfinance, cached to parquet under `trading-agent/training_data/equity_daily`.
  The cache key includes the start date, so asking for a different start silently rebuilds it.
- **Fundamentals:** SEC EDGAR XBRL company facts, free, back to about 2009. Used strictly point in time,
  where a figure only becomes visible on or after its filing date, so a late restatement cannot leak
  backwards.
- **News:** RSS polling with hash-based de-duplication and a finance and macro keyword filter.

### Execution

`backend/execution/` holds an Alpaca broker, a Binance USD-M futures broker (post-only passive limit at
the near touch, poll, cancel, market sweep on timeout), an optional TWAP child-order slicer, and a
`LiveRouter` that converts target weights into notional orders, caps per-order size, marks shrinking
orders reduce-only and routes perpetuals to Binance and everything else to Alpaca.

Live routing is fail-safe by construction: `make_live_router` returns `None` unless `PAPER_TRADING=false`
**and** valid per-venue keys are present, so the flag on its own cannot move money. Every fill writes a
decision-price versus fill-price record to a slippage ledger, which exists so the backtest's assumed
cost can eventually be checked against real fills rather than trusted.

An IBKR adapter and an on-chain execution engine also exist and are registered at startup for status
reporting, but neither is reachable from the router. See [Known problems](#known-problems-and-gotchas).

### News and the LLM layer

The LLM never picks trades. It reads headlines and produces severity, direction, confidence and a time
horizon, which the overlay converts into a risk multiplier. An entity graph propagates a headline from
its source asset to related tickers, but edge weights are earned: a curated prior is multiplied by a
rolling 90-day return correlation, so a stale narrative decays to zero on its own. An LLM edge-discovery
agent may nominate new graph edges, and the correlation validator disposes of them.

This design is the direct result of a measurement. The 20 news dimensions in the training feature vector
were structurally zero offline, because there is no bar-aligned historical news corpus, so the network
never learned anything from news. News is rare and fat-tailed, which is exactly what a data-starved
network cannot learn and what LLM reasoning is actually useful for. So news was moved out of the model
and into the risk gate.

### The dashboard

Next.js 14 with the app router, TypeScript, Tailwind, TanStack Query, recharts and lightweight-charts.
`/` redirects to `/dashboard` (book value, allocations, book health, an agent chat panel), and there are
pages for `positions`, `performance` (overall, today, per asset, model and news, latency), `news`,
`assets`, `audit` (the journal as an event trail) and `settings` (an editor for the `.env` values needed
on first run). `/signals` is still a stub. Data comes from the backend's REST API at
`127.0.0.1:8000/api` through a shared hook layer, plus a websocket for live updates; the frontend never
touches Redis directly.

### The research harness

Everything in `trading-agent/scripts/` is offline and reproducible from cached data:

| Script | What it answers |
|---|---|
| `signal_audit.py` | Information coefficient and AUC per feature. The study that ended the direction premise |
| `strategy_zoo_research.py` | 12 classical TA families across 49 assets, scored with deflated Sharpe |
| `managed_beta_research.py` | Trend-filtered beta versus buy-and-hold, per asset and combined |
| `combined_book_research.py` | The two-sleeve book, correlation and diversification benefit |
| `risk_stack_sweep.py` | The volatility target and financing grid. Where CAGR actually comes from |
| `factor_research.py`, `fundamental_factor_research.py` | Daily price factors and EDGAR value/quality |
| `leadlag_research.py` | Directed predictive correlation across five timescales |
| `meta_label_research.py` | Triple-barrier meta-labelling, in-sample versus out-of-sample |
| `backtest_portfolio.py` | Walk-forward portfolio harness with per-fold Sharpe, alpha, beta, DSR |
| `train_quantile_tcn.py` | Trains the sizing overlay, pinball and Sharpe objectives, multi-seed |
| `pretrain.py`, `evaluate.py` | Offline model training and the net-alpha promotion check |
| `postmortem.py` | Scheduled review of the live journal into per-agent lesson books |
| `backtest_dashboard.py` | A separate browser view of a backtest run, on port 8765 (`launch_backtester`) |

## Repository layout

```
trading-agent/
  backend/
    main.py            FastAPI service, process supervision, agent wiring
    agents/            StrategyAgent (the live book), news agent, risk agent, LLM agents
    strategies/        Strategy library. Only managed_beta and ts_momentum reach the live book;
                       cross_sectional, stat_arb, mean_reversion, funding_carry, tsmom_multiasset
                       and meta_label are research-only and kept for the record
    backtest/          Vectorised engines, portfolio backtest, statistics (DSR, alpha/beta)
    features/          Shared feature pipeline and store, live equals offline by test
    signals/           feature_spec, macro regime, entity graph, lead-lag
    models/            LSTM/GRU/TCN trunks, QuantileTCN, LightGBM, stacker, conformal layer
    risk/              RiskManager, news overlay, funding cap
    execution/         Broker adapters, live router, slippage ledger
    data/              Market feeds, daily equity loader, derivatives, fundamentals, ticks
    api/               REST routes consumed by the dashboard
  frontend/            Next.js 14 dashboard
  scripts/             Research and training entry points
  tests/               100 files, about 550 tests
models/                Checkpoints, including the TCN overlay seeds
start.bat / start.sh   Launchers
redis_up.bat / .sh     Idempotent Redis container
```

## Running it

### Requirements

- Node.js 18 or newer, 20 recommended
- Python 3.10 or newer
- Redis, for cross-process state. The launcher starts it in Docker; failing that, the backend fetches a
  portable Windows build and starts that instead
- Optional: Ollama for a local LLM, or a Gemini or Anthropic key

### Start

From the repository root, run the launcher for your platform. `start.bat` on Windows, `./start.sh` on
macOS or Linux. Do not mix them: they differ in shell syntax and virtualenv paths, not just in style.

The launcher copies `.env.example` to `.env` if needed, brings up Redis, installs frontend dependencies
with `--legacy-peer-deps` on first run, starts the frontend in its own window, then creates
`backend/.venv`, installs `requirements.txt` and runs `main.py` in a restart loop in a second window.
Open `http://localhost:3000`.

Complete the setup prompt on first load, or fill the same values in Settings, or edit
`trading-agent/.env` by hand. Nothing trades until `STRATEGY_AGENT_ENABLED=true`.

### Manual start

```bash
cd trading-agent/frontend && npm install --legacy-peer-deps && npm run dev
```

```bash
cd trading-agent/backend && python -m venv .venv
.venv/bin/python -m pip install -r ../requirements.txt
PYTHONPATH=.. .venv/bin/python main.py
```

### Running the research instead

The research scripts need no server, no Redis and no keys. They read the parquet cache under
`trading-agent/training_data/` and print walk-forward results to stdout. The cache is not in the
repository, so the first run of anything touching daily bars downloads them through yfinance, and the
crypto sleeves need Binance klines in the cache before they will build.

```bash
cd trading-agent && python scripts/combined_book_research.py
```

Most scripts take flags that switch the universe, the cost model or the variant under test. Read the
docstring at the top of each one: it states the question the script was written to answer and, usually,
the answer it gave.

## Configuration

All configuration is environment variables loaded through the backend settings module. The ones that
matter:

| Variable | Default | Meaning |
|---|---|---|
| `PAPER_TRADING` | `true` | Master safety for the book. With `true`, no order can reach a venue |
| `PAPER_MODE` | `true` | The older equivalent for the legacy neural and on-chain paths. Leave it on |
| `STRATEGY_AGENT_ENABLED` | `false` | Runs the rule-based book. This is the live deliverable, so turn it on |
| `NN_AGENT_ENABLED` | `true` in `.env.example` | The legacy neural trader. **Set it to false.** The checkpoint in `models/` is a random-init file that loads cleanly, so leaving this on runs an untrained model |
| `STRATEGY_AGENT_SYMBOLS` | curated basket | Managed-beta universe |
| `STRATEGY_AGENT_TS_SYMBOLS` | crypto perps | Time-series momentum universe |
| `STRATEGY_AGENT_TARGET_VOL`, `_BOOK_VOL_TARGET` | 0.15 / 0.30 | Sleeve and book volatility targets |
| `STRATEGY_AGENT_REBALANCE_SECONDS`, `_MARK_SECONDS` | 86400 / 20 | Trade rarely, mark often |
| `STRATEGY_AGENT_NEWS_OVERLAY` | on | The tighten-only news risk gate |
| `ALPACA_*`, `BINANCE_FUTURES_*` | unset | Venue credentials. Required before anything can go live |
| `AI_PROVIDER` | `ollama` | `ollama`, `gemini` or `anthropic` for the news and copilot layers |

## Paper mode and live mode

Paper mode is the default and the only mode this system has ever run in. The paper book holds real
target weights, marks against real prices, charges a modelled cost per rebalance and persists itself to
Redis so a restart does not reset equity.

Live mode needs three independent things to line up: `PAPER_TRADING=false`, valid per-venue API keys,
and a router that is only constructed when both of the above hold. This is deliberate. A single flag,
a single typo or a single stale config cannot route an order.

Note that paper is optimistic relative to live: it charges a flat cost per rebalance and no perpetual
funding, and it never misses a fill.

## The research record

### 1. Predict the direction

The original system was a neural network predicting short-term direction: a 90-dimensional feature
vector per 5-minute bar over a 60-bar lookback, an LSTM (later switchable to GRU or a causal TCN) with
attention, direction heads at 15m, 1h and 4h horizons, learned stop-loss and take-profit heads,
triple-barrier volatility-scaled labels, focal loss, embargoed chronological splits, and online
advantage-weighted regression with fractional Kelly sizing.

The engineering was sound. The target was wrong.

### 2. The reckoning

Three findings dismantled the premise, in this order.

**Train/serve skew.** Offline training and live inference computed features differently, and offline
z-scored inputs while live did not. The model's reported performance was real and was measuring a
different exam from the one it sat live. Fixed by extracting one canonical pipeline with a byte-for-byte
parity test between live and offline. That fix did not create the edge, it just made the verdict legible.

**Direction is 0.53 AUC.** With features unified, the signal audit measured actual information content:
0.53 to 0.54 AUC at crypto 1h and 4h horizons, no learnable intraday edge in single-name US stocks at
all, and about 45% of the 90 features dead at |IC| below 0.003. The dead ones were exactly the blocks
that were structurally zero offline: order book, news, macro, earnings, constant spread. The best single
feature found was multi-scale momentum (`mom_864`) at IC 0.056.

**Zero for 588.** To rule out "wrong indicators", 12 distinct classical TA families were tested across
49 sector-diverse assets, cost-aware, scored with the deflated Sharpe ratio. Zero of 588 strategy-asset
trials cleared DSR > 0.95, including zero in health and zero in finance. Median post-cost Sharpe per
family was mostly negative. A Fourier dominant-cycle test came back at the noise floor (mean |IC| 0.020,
signed +0.004). Markets are not stationary-cyclic, and a Fourier low-pass just re-derives a moving
average.

An IC of 0.02 to 0.05 is not nothing, but by Grinold's fundamental law an information ratio comes from
breadth, not from squeezing a weak predictor harder. That reframing became the compass for everything
after.

### 3. The pivot

The premise changed from "predict the market" to "hold a portfolio of economically motivated edges, and
use ML only as a meta-layer". Every strategy and signal family below was built, walk-forward tested and
scored against a multiple-testing bar. Most died.

| Strategy or idea | Best Sharpe | Deflated Sharpe | Regression alpha | Beta | Verdict |
|---|---|---|---|---|---|
| NN 5m direction | 0.53 AUC | | | | Demoted |
| TA zoo, 12 families x 49 assets | best ~0.7 raw | 0 of 588 > 0.95 | | | Dead |
| Time-series momentum 1h | -0.15 to -0.83 | 0.001 | | ~0 | Dead |
| Time-series momentum 4h + ADX | +0.55 to +0.94 | 0.14 to 0.30 | +16.6%/yr | ~0 | **In the book, low confidence** |
| Cross-sectional momentum, semis | +4.65 | 0.374 | +0.20 | ~0 | Rejected, regime-overfit |
| Cross-sectional momentum, crypto | +0.17 to -0.39 | 0.002 | | ~0 | Dead |
| Statistical arbitrage, semis | +0.02 | 0.000 | -43.5%/yr | ~0 | Shelved |
| Mean reversion, crypto | -90% drawdown | 0.000 | | | Shelved |
| Funding carry | +0.04 | 0.690 | ~0 | ~0 | Real but +0.2%/yr |
| Daily factor, 12-1 momentum | +0.31 | 0.234 | ~0 | +0.14 | Beta in disguise |
| Daily factor, 1m reversal | +0.39 | 0.350 | -1.4%/yr | +0.25 | Beta plus negative skill |
| Value (EDGAR, point in time) | | | +0.1%/yr | +0.06 | Real, marginal |
| Quality (EDGAR, point in time) | | | +0.3%/yr | +0.02 | Real, marginal |
| Value + quality | 0.41 | 0.394 | +0.2%/yr | +0.04 | First clean neutral alpha, tiny |
| **Managed beta, combined** | **1.10** | not deflated, 5 of 6 folds | **+7.4%/yr vs buy-and-hold** | ~1 | **The deliverable** |
| **Two-sleeve book** | **0.80** | | | ~0.5 | **The live paper book** |
| Daily multi-asset TSMOM | ~0.00 | | | | Rejected |
| FX and commodity trend | +0.12 | | | low | Candidate, thin |
| Lead-lag, intraday | net -10 to -147 | 0.000 | | | Dead |
| Lead-lag, daily | gross +0.42, net -0.85 | | IC_oos +0.03 | | Real, turnover-bound |
| DMN sizing overlay | +0.37 timing skill | | | | Wired, paper, off by default |
| Meta-labelling | OOS AUC 0.50 | | | | Parked |

Three of those deserve a sentence each, because they were the most instructive failures.

**Cross-sectional momentum on semiconductors** returned a standalone Sharpe of +4.65 over 1.4 years,
four folds out of four positive, beta near zero. Its deflated Sharpe was 0.374, meaning it sat inside the
luckiest-of-N envelope. Re-run on a broader universe it collapsed to -0.09. Without the deflation gate we
would have shipped regime luck as alpha.

**Daily price factors** looked workable at a combined Sharpe of +0.46 across 21 years of the S&P 500,
six folds out of six positive. Then the alpha/beta regression gutted it: the books leaked +0.14 to +0.25
of market beta, and once that was regressed out the alpha was zero to negative. The entire +2.5%/yr was
incidental market exposure, which a 20% SPY position would have delivered at a better Sharpe.

**Meta-labelling** was the López de Prado idea that fit the diagnosis best: if you cannot predict
direction, predict whether a trade will hit its target before its stop, and size on that. In-sample AUC
0.90, out-of-sample AUC 0.50, and sizing on that noise hurt Sharpe everywhere. That gap is the cleanest
single demonstration in the project that there is no out-of-sample predictable structure here, in
direction or in trade quality.

### 4. What survived

Trend-filtered beta and the risk machinery. The managed-beta book over 2010 to 2025 (SPY, QQQ, TQQQ,
TLT, GLD, BTC, ETH) returned Sharpe 1.10 against buy-and-hold's 0.89, max drawdown -27% against -45%,
with the same CAGR. The trend filter helps most where crashes are deepest: TQQQ drawdown went from -82%
to -52%, and over a 2000 start SPY and QQQ went from -69% to -31%. Adding a VIX and cross-asset
correlation overlay took it to 1.13 with a -23% drawdown. All rf = 0.

Adding the 4h crypto momentum sleeve produced the first genuinely constructive result. The two streams
correlate at +0.098, and the 50/50 combination lifted Sharpe from 0.65 to 0.80 on the hard 2022 to 2026
window while cutting max drawdown from -27% to -19.5%. Neither sleeve is good. The combination is better
than either, which is the fundamental law paying out.

The volatility sweep then established where return actually comes from, and what limits it. Sharpe is
flat at 0.80 to 0.83 across the entire leverage grid, exactly as theory says it must be, while CAGR
scales with the volatility target. Financing is the binding constraint:

| Financing rate | Best pick at a -25% drawdown budget | CAGR |
|---|---|---|
| 0%/yr (fantasy) | target vol 0.30, leverage cap 2.0 | +17.3% |
| 2%/yr | target vol 0.15, cap 3.0 | +12.3% |
| 5%/yr (realistic) | target vol 0.15 | **+11.0%** |
| 8%/yr | target vol 0.15 | +9.7% |

That is why the funding cap is offence rather than defence: borrowing only when smoothed funding is
cheap is the difference between those rows.

### 5. ML earns a small place back

Not as a direction picker, but as a sizing gate, where an error costs a mis-size rather than a wrong-way
trade. A Deep Momentum Network head (Lim, Zohren and Roberts) trained directly on net Sharpe with a
turnover penalty, used only to scale the momentum sleeve between a 0.25 floor and 1.0, produced +0.37
Sharpe of timing skill on an untouched test tail, verified against a constant-scaling control so the
gain was skill rather than de-levering. The pinball-loss variant of the same model showed negative
timing skill. Predicting the metric you care about beat predicting an intermediate quantity and
thresholding it.

The caveats were recorded at the time and they matter: one window, a saturated tanh head that had
degenerated into a binary agree/disagree filter, and a trial count that makes the deflated Sharpe
meaningless. Against a 75% probability of backtest overfitting on the parent book, this overlay is not
evidence of anything. It is wired live behind a flag, paper only, and off.

A LightGBM meta-layer was later run over the 5-minute crypto stack as the ML arm of a rebuild. Zero of
90 features cleared |t| > 3.84 over 404 tests on 691,890 bars, and the model returned an excess Sharpe
of -0.390 at DSR 0.000. Nine of the "momentum" columns turned out to be the trailing return under
different names, at rank correlation 1.0000.

### 6. The audit

The most valuable session in the project was the one that attacked its own best result. The findings
are in [The short version](#the-short-version-of-the-result); the method is worth repeating:

- **Subtract the risk-free rate.** Cash paid 3.97%/yr over the window. A grep for `risk_free` across the
  backend returned nothing. About a third of the headline was T-bill yield.
- **Check the calendar.** Outer-joining a crypto and equity panel puts weekends in the index, and
  `nan_to_num` turned "no data" into "no return", deleting 74% of Monday equity returns. Absence must
  propagate, not be coerced to zero. Fixing it moved the sleeve's max drawdown from -27% to -39.9%.
- **Charge realistic costs.** Half-spread plus square-root impact plus financing plus funding. At $1M the
  old flat 15bps was conservative. Capacity analysis put the point where impact fully consumes the alpha
  at roughly $20M.
- **Measure selection, not just fit.** On an 864-variant grid, the probability of backtest overfitting
  was 75%, White's reality check gave p = 0.119 and Hansen's SPA gave p = 0.664. All 864 variants were
  positive and the shipped one sat at the 38th percentile of its own grid, which means the parameters
  were never the thing that mattered.
- **Know the ceiling.** With a measured per-sleeve Sharpe of 0.33 and inter-sleeve correlation of 0.09,
  the fundamental law caps the book at 1.10 no matter how many sleeves are added. Reaching 1.6 would
  require every sleeve to be a 0.48 standalone; the best ever measured here was 0.33.

The one free lever the audit did find: the managed-beta sleeve sits in cash 45.3% of the time and earns
nothing on it in the backtest. Crediting T-bill yield on idle cash is worth roughly +0.07 to +0.14
Sharpe for no added risk, and needs cash-sweep accounting rather than a model.

## Where the project is today

**The deliverable** is the two-sleeve paper book with the full risk stack, plus the research harness that
produced the verdicts above. Corrected Sharpe +0.33 with a confidence interval spanning zero.

**The live state** is honest and unimpressive. The trade journal contains activity on two days,
2026-07-09 and 2026-09-04, with the book at roughly $97.2k of a $100k paper start. There is no realised
live Sharpe because the agent has barely traded. The system is not hosted; it runs when the laptop runs.

**The models are not trained** beyond 30-minute Colab probes. That is an ordering decision rather than an
oversight: the profit lever (volatility target and financing) needs no model, and the model's marginal
value is the speculative part.

**An unmerged rebuild branch** (`rebuild/d1-git-hygiene`, 38 commits ahead of `main`) executes a full
deletion and repair plan against everything above: it removes the on-chain execution path and the wallet
key, retires the neural policy process, consolidates two import roots and the forked data directories,
adds a kill switch and dead-man timer, makes the live configuration equal to the validated one, wires
the risk-free rate and deflated Sharpe into the measurement path, adds a promotion gate so no
configuration reaches live unevaluated, and replaces the ML arm with the LightGBM stack described above.
It is not merged, and this README describes `main`.

**Open threads**, in the order they are worth pulling:

1. Set the volatility target to the chosen drawdown budget. Biggest CAGR move available, needs no model.
2. Credit idle-cash yield. Free Sharpe, needs accounting.
3. Turnover-controlled daily lead-lag. The one research thread that showed genuine out-of-sample
   persistence (IC 0.03) before turnover ate it.
4. FX and commodity trend on full history. Lowest correlation stream found, but tested only on a window
   that excludes its best regimes.
5. An always-on host, so that "the live Sharpe is undefined" stops being the answer.

## Known problems and gotchas

- **Never run `pytest` against a live backend without checking test isolation first.** In July 2026 the
  suite wiped the live paper book, 98,995 down to 15,698, because Redis fallback pinged the real server
  first, a test published its fixture book, and the fixture tickers `A`, `B` and `C` are real NYSE
  symbols that then got marked at real prices. `tests/conftest.py` now forces a closed Redis port, a
  temporary database and a temporary journal, and a test pins that behaviour. Verify it is intact.
- **Two import roots.** `trading-agent/` and `trading-agent/backend/` can both act as the package root
  depending on how you launch, which produces duplicate settings objects, database engines and
  `training_data/` directories. Repaired on the rebuild branch, still present on `main`.
- **Alpaca free tier is IEX only.** Spreads and volumes are single-venue. Do not use them as liquidity
  signals.
- **The cleanup watchdog is indiscriminate.** On shutdown `backend/watchdog.py` runs
  `taskkill /F /T /IM` against `node.exe`, `ollama.exe` and `redis-server.exe` by image name, so it will
  take down unrelated Node processes and any other Redis or Ollama you had running.
- **The on-chain execution path is dead but still wired.** The DeFi engine is still constructed at
  startup and served under `/api/wallet/*`, and the frontend still ships wagmi, viem, a Ramp SDK, a
  wallet panel and a connect modal. None of it is reachable from the live router, and `.env.example`
  still declares `AGENT_PRIVATE_KEY`. Deleted on the rebuild branch.
- **Runtime artifacts are tracked in git on `main`** (database, logs, checkpoints), and there are stale
  duplicates: an out-of-date `trading-agent.db` at the repository root next to the live one inside
  `trading-agent/`, an empty root `training_data/`, and three separate model trees. Also fixed on the
  rebuild branch.
- **The news overlay is one-sided.** It de-gears far more often than it upsizes, on a signal whose
  information coefficient has never been measured, which makes it a drag of unknown size.
- **Backtest cost assumptions are unvalidated.** The slippage ledger exists to check them and has close
  to zero real fills in it.

## Lessons

1. **Test the system metric, not the proxy.** Garman-Klass volatility had better rank IC and still
   tracked 3.6% worse on the thing it was for. Meta-labelling had 0.90 in-sample AUC and 0.50 out.
2. **Deflate for multiple testing, always.** A +4.65 Sharpe book was a mirage and the deflated Sharpe
   said so before the broader re-run confirmed it.
3. **Separate alpha from beta with a regression, every time.** A book can look like skill and be a
   0.20 beta tilt wearing a costume.
4. **Diversification is the most reliable edge available.** The one constructive result came from
   combining two mediocre uncorrelated streams, not from a better predictor.
5. **At low signal-to-noise, the objective and the regularisation matter more than the architecture.**
   The last productive model change was a wiring fix, not a bigger network.
6. **The risk machinery is the durable edge.** After testing direction, TA, factors, fundamentals,
   stat-arb, carry and lead-lag, what consistently added risk-adjusted value was volatility targeting,
   drawdown control and trend-filtered beta.
7. **Keep the model on a leash.** ML may scale what a robust rule already wants, never originate. Then
   the worst failure mode is "too cautious" rather than "ruined".

The blunt summary: this project set out to build a machine that predicts the market, found that machine
cannot exist at this scale with this data, and built instead a machine that manages risk better than
buy-and-hold and diversifies mediocre edges into a slightly less mediocre one. The numbers above are the
receipts, including the ones that say the remaining edge is indistinguishable from luck.

## Disclaimer

This is experimental research software, not financial advice and not a product. It ships in paper mode
and has never traded real capital. If you enable live execution you are operating production trading
infrastructure with your own money, and you are responsible for every order it sends. The authors accept
no liability for any financial loss.

## License

Apache License 2.0. See `LICENSE`.
