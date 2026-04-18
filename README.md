# The Wolf of Wallstreet
Smart Trading Agent

## Overview
This is a fully autonomous AI trading system that operates with a dual intelligence architecture. It combines the speed of a neural network with the semantic analysis capabilities of a Large Language Model. The system executes trades onchain and logs events using Kite AI.

The core idea is that the neural network (built with PyTorch LSTMs) acts as the trading brain, running continuous loops to make sub 100ms decisions based on technical analysis and market signals. At the same time, the LLM runs in an isolated background process, scanning financial news via RSS feeds and APIs. When the LLM detects a financially significant event, it signals the neural network via a priority queue to adjust risk limits or halt trading immediately without blocking the normal tick cycle.

## Project Structure
The repository is split into two main sections:
* backend: A Python 3.11 environment running the algorithmic agent, FastAPI, PyTorch, CCXT for market feeds, and Web3.py for blockchain interactions.
* frontend: A Next.js 14 application providing the user dashboard, built with TypeScript, TailwindCSS, lightweight charts, and wagmi for wallet connections.
* infrastructure: Postgres 16 and Redis 7, handled inside a Docker Compose setup.

## Core Features
* Parallel Processing: The NN and LLM run on separate cores using Python multiprocessing. They communicate through a shared priority queue via Redis so the trading loop never hangs waiting for an API response.
* Dynamic News Interruption: The continuous news feed uses local models via Ollama (e.g., Llama 3) or external APIs (e.g., Google Gemini/Anthropic Claude) to categorize headlines into severity tiers (Neutral, Significant, or Severe), adjusting or stopping the trading engine dynamically.
* Autonomous Execution: Capable of executing swaps automatically on Arbitrum using USDC.
* Setup Automation: Cross-platform bootloaders (`start.bat` & `start.sh`) automatically download and install Ollama, Python, and Node dependencies directly from the web, ensuring no massive installers are tracked in the repository.
* Risk Management: Dedicated synchronous gates check hard limits on drawdowns, position sizing, correlation, and trade frequency before any order is submitted.
* Verifiability: Features an on chain agent identity and logs trade predictions and executions on the Kite AI blockchain.

## Getting Started
To get the system running locally, you need Docker, Docker Compose, and Git installed.

1. Clone the repository to your local machine.
2. For an easy automatic setup, run the provided start scripts depending on your OS:
   * Windows: Double click `start.bat` or run it in your terminal.
   * macOS/Linux: Run `./start.sh` in your terminal.
   These scripts will pull the code, build the Docker containers, and start the system.
3. Alternatively, to run manually:
   * Copy the `.env.example` file to `.env`.
   * Start the containers: `docker-compose up --build`
4. Once the application is running, visit http://localhost:3000 in your browser to access the dashboard.
5. A setup modal will appear where you can directly enter your API keys and configuration data.
6. To pretrain the LSTM model (optional but recommended), open a second terminal and run:
   `docker-compose exec backend python scripts/pretrain.py`
7. Connect your web3 wallet to the local setup using the Kite AI chain parameters.

## Paper vs Live Trading
By default, the agent runs in paper mode. In this mode, trades are simulated and portfolio values are tracked only in the local database. No real capital is exposed.

To run in live mode, you must set PAPER_MODE=false in your .env file and fund your Arbitrum wallet with USDC. It is highly recommended to start with a very small amount for the first 24 hours to monitor live execution safely.

## Deployment Notes
* Frontend: Can be easily deployed on Vercel as a standard Next.js application. Keep in mind to set all NEXT_PUBLIC environment variables in the dashboard.
* Backend: Must be deployed on a persistent server like an AWS EC2 instance or a DigitalOcean droplet. Serverless environments are not compatible with the Python multiprocessing requirements of the agent. You can use the provided docker-compose.prod.yml file for a production ready setup with resource limits.
