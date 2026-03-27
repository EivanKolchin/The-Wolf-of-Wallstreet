# Hackathon Tracks — Coverage Analysis
### Kite AI Hackathon 2026

---

## Track 1 — Agentic Trading & Portfolio Management
*Supported by Kite AI*

> Build autonomous trading agents that analyse markets, execute on-chain trades, manage risk dynamically, allocate capital across DeFi protocols, and perform cross-chain arbitrage.
> Focus: AI-native trading infrastructure, reputation-aware capital delegation, stablecoin-first settlement.

### What the build satisfies at completion

| Requirement | Status | Evidence |
|---|---|---|
| Autonomous trading agent | ✅ Full | NN Trading Agent (Process 1) runs a full cycle every 5 seconds without human input |
| Analyses markets | ✅ Full | 62-feature vector: MAs, RSI, Fibonacci, S/R, patterns, order book, CVD, regime |
| Executes on-chain trades | ✅ Full | ExecutionEngine + KiteChainClient logs every trade to Kite chain with tx hash |
| Manages risk dynamically | ✅ Full | RiskManager with drawdown gate, daily loss limit, SEVERE interrupt protocol |
| AI-native trading infrastructure | ✅ Full | LSTM + online learning + LLM news agent — both AI-native components |
| Reputation-aware capital delegation | ⚠️ Partial | Trust scores update per source, but agent-to-agent delegation not built |
| Stablecoin-first settlement | ⚠️ Partial | USDC settlement not explicitly implemented — trades are via CCXT/Binance. Kite chain tx logs exist but are not USDC payments |
| Allocates capital across DeFi | ❌ Not built | Single exchange (Binance CEX). No DeFi protocol allocation. |
| Cross-chain arbitrage | ❌ Not built | Single chain (Kite AI). No cross-chain logic. |

### Overall Track 1 satisfaction: ~65% at build completion

The core trading infrastructure, risk management, and AI-native decision making is fully built and genuinely autonomous. The gap is on the DeFi/cross-chain/stablecoin side of the track description.

### How to close the gap (priority order)

**1. USDC settlement on Kite chain (high impact, medium effort — ~8h)**
Replace the Binance execution path with a Kite AI native DEX call that settles in USDC. If Kite AI provides a DEX contract or x402 payment endpoint, route trades through it instead of CCXT for at least a demo-mode trade. This satisfies "stablecoin-first settlement" directly.

**2. On-chain reputation score (medium impact, low effort — ~4h)**
Store the agent's cumulative win rate and prediction accuracy on-chain (a simple smart contract mapping `agent_address → ReputationScore`). Any other agent or protocol can then query this on-chain reputation before delegating capital. This satisfies "reputation-aware capital delegation" in spirit.

**3. Uniswap/DeFi allocation (medium impact, high effort — ~16h)**
Add a second execution path: when the NN produces a signal for ETH/USDC with high confidence, route a portion of capital through Uniswap v3 (or a Kite AI native DEX) rather than Binance. This demonstrates DeFi protocol allocation and makes the demo substantially more impressive. Feasible in week 4 if week 3 finishes ahead of schedule.

---

## Track 2 — Agentic Commerce
*Supported by Kite AI*

> Build autonomous AI agents that discover products and services, execute USDC payments via x402, manage subscriptions and usage-based billing, and interact with APIs using programmable constraints.
> Focus: agent-to-API payments, stablecoin settlement, verifiable identity, and real-time execution on Kite AI.

### What the build satisfies at completion

| Requirement | Status | Evidence |
|---|---|---|
| Autonomous AI agent | ✅ Full | Two parallel agent processes, continuous operation |
| Executes payments | ⚠️ Partial | Kite chain transactions exist, but not USDC payments via x402 |
| Agent-to-API payments | ⚠️ Partial | `kite_chain.pay_for_data()` method exists as a stub — not wired to real data API payments |
| Verifiable identity | ✅ Partial | Agent DID (agent_address), all actions signed with private key, tx hashes on-chain |
| Real-time execution on Kite AI | ✅ Full | Every trade logged to Kite chain in real time |
| Discovers products/services | ❌ Not built | Not applicable to trading context |
| Manages subscriptions / usage billing | ❌ Not built | Not applicable to trading context |
| Programmable constraints on API use | ⚠️ Partial | Risk manager limits exist but are not expressed as on-chain programmable constraints |

### Overall Track 2 satisfaction: ~30% at build completion

The verifiable identity and real-time Kite chain execution cross over from Track 1. The commerce-specific elements (x402 payments, subscriptions, API discovery) are not naturally part of a trading agent.

### How to incorporate Track 2 elements

**1. Wire the data payment stub (medium impact, low effort — ~4h)**
The `pay_for_data()` method already exists in KiteChainClient. Wire it to an actual paid data API that accepts on-chain payment. Candidates:
- If Kite AI provides a data marketplace or oracle service, pay for news feed access via USDC on Kite chain
- Alternatively, mock this: agent pays a small USDC amount to a "data provider" wallet every hour as a subscription fee, logged on-chain. Not real commerce but demonstrates the pattern for judges

**2. x402 payment for Claude API calls (high impact, medium effort — ~6h)**
Frame the LLM API calls as paid agent actions: each time the agent calls claude-sonnet for severity classification, it logs an on-chain micropayment record (even if symbolic). Position this as "the agent pays for its own intelligence" — demonstrates genuine AI economic agency, which is the spirit of the Commerce track.

**3. Programmable spending constraints (medium impact, medium effort — ~6h)**
Add an on-chain spending policy: a simple mapping of `agent_address → {max_daily_spend_usdc, max_per_call_usdc}` that the agent reads before any payment. This makes the agent's payment behaviour verifiable and constrainable by the wallet owner — directly satisfies "programmable constraints."

---

## Track 3 — Novel Track
*Supported by Kite AI*

> Build anything unique that doesn't fit the first two tracks. New approaches, unexpected integrations, open applications. If it runs on Kite and does something nobody's seen before, it belongs here.

### What the build satisfies at completion

| Novel element | Status | What makes it novel |
|---|---|---|
| Parallel NN + LLM architecture with interrupt protocol | ✅ Full | Three-tier interrupt (NEUTRAL/SIGNIFICANT/SEVERE) with atomic cross-process flag — not seen in any existing open trading system |
| Persistent online learning LSTM | ✅ Full | NN that never resets and learns from every completed trade — not just inference, but continuous adaptation |
| On-chain AI decision audit trail | ✅ Full | Every NN inference decision logged to Kite chain with full signal context — verifiable AI reasoning |
| LLM credibility engine with feedback loop | ✅ Full | Source trust scores that evolve based on prediction accuracy — self-calibrating news intelligence |

### Overall Track 3 satisfaction: ~80% at build completion

The parallel NN + LLM interrupt architecture and the persistent online learning model are genuinely novel — no existing trading bot, open-source or commercial, does both of these together. The on-chain AI audit trail (logging NN input features + decision at inference time) is also not something seen in existing systems.

### How to strengthen the Novel track angle

**1. On-chain model performance attestation (high impact, medium effort — ~6h)**
Every 50 trades, write a compact performance summary to Kite chain: `{trade_count, win_rate_7d, sharpe_7d, model_version, checkpoint_hash}`. Anyone can verify the agent's claimed performance against the on-chain record — this is a genuinely novel form of AI accountability that no existing system provides.

**2. Agent-to-agent signal sharing (very high impact, high effort — ~20h)**
If another team also builds an agent on Kite chain, enable a paid signal channel: your agent publishes its high-confidence NN decisions to a Kite chain smart contract. Other agents can subscribe (pay USDC) to receive the signal feed. This is agent-to-agent commerce + trading infrastructure combined — a genuinely novel intersection of all three tracks.

**3. On-chain regime history (low effort, medium novelty — ~3h)**
Every time the regime detector fires a change, log the regime transition on-chain. Creates a publicly queryable market regime history that any Kite chain participant can use. Small touch, high novelty value in a demo.

---

## Cross-track Summary Table

| Hackathon requirement | Built | Track 1 | Track 2 | Track 3 |
|---|---|---|---|---|
| AI agent performing tasks | ✅ | ✅ | ✅ | ✅ |
| Settles on Kite chain | ✅ | ✅ | ✅ | ✅ |
| Executes paid actions | ⚠️ stub | ⚠️ | ⚠️ | ✅ |
| Functional UI (web app) | ✅ | ✅ | ✅ | ✅ |
| End-to-end live demo | ✅ | ✅ | ✅ | ✅ |
| Kite chain attestations | ✅ | ✅ | ✅ | ✅ |
| Publicly accessible (Vercel/AWS) | ✅ | ✅ | ✅ | ✅ |
| Reproducible via README | ✅ | ✅ | ✅ | ✅ |
| Autonomous market analysis | ✅ | ✅ | — | ✅ |
| Risk management | ✅ | ✅ | — | ✅ |
| Stablecoin settlement (USDC) | ❌ | ⚠️ | ❌ | — |
| x402 payments | ❌ | — | ❌ | — |
| Agent-to-API payments | ⚠️ stub | — | ⚠️ | ✅ |
| Verifiable agent identity (DID) | ✅ | ✅ | ✅ | ✅ |
| Novel architecture | ✅ | — | — | ✅ |
| Persistent learning | ✅ | ✅ | — | ✅ |
| On-chain audit trail | ✅ | ✅ | ✅ | ✅ |

### Best track to submit under

**Primary: Track 1 (Agentic Trading)** — the build is most complete and most compelling for this track. The architecture, risk system, and on-chain audit trail are exactly what the track description asks for.

**Stretch: Novel Track** — if the parallel NN + LLM architecture and persistent online learning are emphasised in the demo, these are genuinely novel enough to be competitive. Frame the submission as "new AI trading infrastructure paradigm" rather than just a bot.

**If Track 2 elements are added (x402 data payments):** submitting to all three tracks with a note that the project bridges them is possible and would be unusual — potentially a strong differentiator.
