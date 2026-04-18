# AI Trading Agent

A full-stack algorithmic AI trading platform built around autonomous trade execution using PyTorch LSTMs, Large Language Models for news analysis, and a Next.js UI frontend bridged via Kite AI Chain for auditing.

## Structure
- `/backend`: Python 3.11 core algorithmic agent using FastAPI, PyTorch, CCXT, Web3.py.
- `/frontend`: Next.js 14 frontend using TypeScript, TailwindCSS, lightweight-charts, and wagmi.
- We rely on hosted Postgres and Redis instances to remove Docker overhead entirely.

## Getting Started (Native Setup)

We have removed Docker entirely to prevent architecture mismatches between Windows and Mac (Apple Silicon). Both the Web UI and Python Backend run natively.

1. Prerequisites: Git, Python 3.10+, and Node.js 18+
2. Set up Free Cloud Databases:
   - Create a free Postgres database at [Supabase](https://supabase.com/). Get the Connection URI.
   - Create a free Redis instance at [Upstash](https://upstash.com/). Get the Connection URI.
3. Clone repo
4. `cp .env.example .env` and fill in all values
   - Specifically, ensure `DATABASE_URL` matches your Supabase target, and `REDIS_URL` matches your Upstash target.
5. In the root of the project, run:
   - **Windows:** Double-click `start.bat`
   - **Mac/Linux:** Run `bash start.sh`
6. Visit [http://localhost:3000](http://localhost:3000)
7. Connect MetaMask to Kite AI chain (include chain params)
8. Expected state after setup: paper mode active, NN loading pretrained weights,
   news feed connecting, dashboard showing live data within 2 minutes.

## Live mode vs Paper mode
  Paper mode (PAPER_MODE=true in .env): agent runs full cycle, trades are simulated,
  no real swaps occur, portfolio value tracked in DB only.
  Live mode (PAPER_MODE=false): real Uniswap swaps executed on Arbitrum.
  WARNING: start with small amounts. Recommended max $50 for first 24 hours.

## Vercel deployment instructions (for frontend only)
- Set all NEXT_PUBLIC_ env vars in Vercel dashboard.
- Backend must be deployed separately (Railway, Render, or AWS EC2).
- Note: multiprocessing requires a persistent server — Vercel serverless is NOT compatible with the backend. Use a VPS or container service.

## AWS EC2 one-liner deploy
  `python main.py` or use your preferred process manager (like systemd or PM2).
  (uses the prod variant with resource limits and logging config)
