# The Wolf of Wallstreet (WoW)
The Wolf of Wallstreet is an AI assisted trading platform that combines a Python backend trading engine with a Next.js dashboard. It is designed for local development and experimentation with systematic trading, news aware risk controls, and live agentic execution infrastructure.

## Table of contents
- [What this program does](#what-this-program-does)
- [Who this project is for](#who-this-project-is-for)
- [System architecture at a glance](#system-architecture-at-a-glance)
- [Getting started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Fast start with startup scripts](#fast-start-with-startup-scripts)
  - [What `start.sh` vs `start.bat` should be used for](#what-startsh-vs-startbat-should-be-used-for)
  - [Manual startup option](#manual-startup-option)
- [How the system works in simple terms](#how-the-system-works-in-simple-terms)
- [Technical deep dive](#technical-deep-dive)
  - [Backend process model](#backend-process-model)
  - [Neural trading loop](#neural-trading-loop)
  - [News ingestion and AI interpretation](#news-ingestion-and-ai-interpretation)
  - [Risk controls and execution](#risk-controls-and-execution)
  - [API and frontend integration](#api-and-frontend-integration)
- [Configuration notes](#configuration-notes)
- [Paper mode and live mode](#paper-mode-and-live-mode)
- [Troubleshooting](#troubleshooting)



## What this program does
At a high level, the platform continuously watches market data, generates model driven trade decisions, checks those decisions against risk limits, and then executes trades in either paper mode or live mode.

In parallel, a separate news LLM based intelligence process monitors financial RSS sources, classifies incoming headlines with the LLM pipeline, and can raise urgent signals that alter risk behavior or halt trading activity when needed.

The frontend provides a clean browser based control and monitoring surface for setup, status, positions, and related system information.


## Who this project is for
This project is most useful for:
- Developers who want a full stack reference for an autonomous trading workflow.
- Quant and ML practitioners testing model driven execution logic with an interactive UI.

This project is not a beginner financial product and should be treated as an engineering system that requires careful setup and validation.


## System architecture at a glance
The repository includes three major layers:

1. **Backend (`trading-agent/backend`)**  
   FastAPI service plus multiprocessing trading agents, market and news ingestion, risk management, and execution logic.

2. **Frontend (`trading-agent/frontend`)**  
   Next.js 14 TypeScript dashboard that reads API and Redis backed runtime status.

3. **Startup scripts (repository root)**  
   Cross platform launch scripts (`start.sh`, `start.bat`) that help initialize and run frontend and backend services quickly with ease.


## Getting started

### Prerequisites
Install the following before launch if not already installed:
- Node.js 18 or newer (Node.js 20 recommended)
- Python 3.10 or newer
- Git

You should also have a terminal where you can keep service windows open to view logs.


### Fast start with startup scripts
If you want the easiest path, use one of the startup scripts from the repository root.

1. Clone the repository and open a terminal in the repository root.
2. Run exactly one script based on your operating system:
   - **Windows:** `start.bat`
   - **macOS or Linux:** `./start.sh`
   [or alternatively: navigate within folder to your appropiate start script and click to execute]
3. The script launches:
   - The frontend development server
   - The backend service loop in a separate terminal window when available
4. On your browser open: `http://localhost:3000`.


### What `start.sh` vs `start.bat` should be used for
Use this rule:

- **Use `start.bat` if you are on Windows**  
  It handles Windows command semantics, `cmd` process spawning, and virtual environment paths under `.venv\Scripts`.

- **Use `start.sh` if you are on macOS or Linux**  
  It uses Bash, Unix process management, and `.venv/bin` paths.

Do not mix them. Running the wrong script on the wrong operating system will fail because shell syntax and path conventions are different.


### Manual startup option
If you prefer manual control:

1. Move into `trading-agent`.
2. Frontend:
   - `cd frontend`
   - `npm install --legacy-peer-deps`
   - `npm run dev`
3. Backend in a second terminal:
   - `cd trading-agent/backend`
   - `python -m venv .venv` (or `python3 -m venv .venv`)
   - Install dependencies from `../requirements.txt`
   - Start service with `python main.py` (or `.venv` interpreter equivalent)
4. Open `http://localhost:3000`.


## How the system works in simple terms
Think of the system as two coordinated AI workers plus a dashboard:

- **Worker 1: Trading brain**  
  Reads market data and decides whether to go long, short, or hold [based on past data and mathematical models for decision making].

- **Worker 2: News brain**  
  Watches financial news from trusted sources and flags important events, rating each on different things.

- **Risk layer in the middle**  
  Blocks unsafe decisions before execution.

- **Execution layer**  
  Places paper trades (using fake money for testing) or live trades depending on configuration.

- **Dashboard**  
  Shows system state and lets you monitor behavior in real time with a modern interface.


## Technical deep dive

### Backend process model
The backend entrypoint initializes FastAPI, database setup, websocket updater tasks, and then spawns two multiprocessing workers:
- `NNTradingAgent` process for model driven trading decisions.
- `LLMNewsAgent` process for news analysis and event severity signaling.

A shared severe event flag coordinates emergency behavior across processes.


### Neural trading loop
The neural process performs repeated cycles that include:
1. Building feature vectors from market candles, order flow context, and regime information.
2. Maintaining rolling sequences for model inference.
3. Running inference with the persistent trading model to get direction probabilities.
4. Producing a `TradeDecision` object containing direction, size, confidence, and contextual metadata.
5. Sending decision output through risk approval gates before execution.

The agent also pushes visualization oriented prediction payloads to Redis for frontend consumption.


### News ingestion and AI interpretation
News ingestion polls a configured RSS list, deduplicates articles with hash based tracking, filters by finance and macro keywords, and forwards relevant items into downstream analysis.

The LLM based news agent and credibility pipeline classify impact severity. Significant events can bias risk behavior, and severe events can trigger emergency safeguards.


### Risk controls and execution
The risk manager enforces hard constraints including:
- Maximum portfolio drawdown
- Maximum daily loss
- Position size limits
- Trade frequency limits
- Minimum notional filters

Only approved decisions continue to execution.

Execution supports paper or live behavior. For each approved trade, the engine:
1. Normalizes quantity with exchange precision and min notional checks.
2. Selects order type logic based on context.
3. Records trade metadata in the database.
4. Emits audit style trade logging through Kite chain integration.


### API and frontend integration
FastAPI routes provide health, setup, risk, and operational endpoints, while websocket tasks stream live updates used by the Next.js UI.

The frontend is implemented with modern React and TypeScript patterns and is organized into dashboard pages, widgets, and shared libraries for API and state handling.


## Configuration notes
- Startup scripts attempt to create `.env` from `.env.example` when available.
- Backend settings are loaded through the project configuration module.
- External providers may require keys and endpoint configuration before full functionality is available.
- Optional local LLM flows may depend on Ollama availability.


## Paper mode and live mode
- **Paper mode** is the safe default path for simulation and validation. By default $1000 awarded in digital fake money for monitoring and test the model.
- **Live mode** should only be used after extensive testing, strict limit review, and controlled capital exposure.

If you enable live execution, treat this as production trading infrastructure and implement your own operational safeguards.

**We are NOT liable for ANY financial losses due to our program, this program is intended for experimental and educational purposes not profit making**


## Troubleshooting

### Frontend dependency issues
If `npm run dev` fails, try reinstalling dependencies:
- `npm install --legacy-peer-deps`


### Backend restarts repeatedly
Review backend logs for missing dependency, provider credential, database, or Redis connectivity errors.


### No secondary terminal opens on Linux
`start.sh` attempts `gnome-terminal`, `xterm`, or `konsole`, then falls back to inline execution.


### Issues with the trading execution / LLM news agent 
Ensure on start you have completed the setup as asked for on start. if you close this you can complete it in settings and then click save. If there are issues saving this data you can manually enter the required data in the .env file (\trading-agent\.env)


## License
This repository is distributed under the terms of the included `LICENSE` file.
