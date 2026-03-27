# AI Trading Agent

A full-stack algorithmic AI trading platform built around autonomous trade execution using PyTorch LSTMs, Large Language Models for news analysis, and a Next.js UI frontend bridged via Kite AI Chain for auditing.

## Structure
- `/backend`: Python 3.11 core algorithmic agent using FastAPI, PyTorch, CCXT, Web3.py.
- `/frontend`: Next.js 14 frontend using TypeScript, TailwindCSS, lightweight-charts, and wagmi.
- Docker compose environment runs Postgres 16, Redis 7, Next.js frontend, and the Python backend on a shared network.

## Getting Started

1. Prerequisites: Docker, Docker Compose, Git
2. Clone repo
3. `cp .env.example .env` and fill in all values
4. `docker-compose up --build`
5. In a second terminal: `docker-compose exec backend python scripts/pretrain.py`
6. Visit [http://localhost:3000](http://localhost:3000)
7. Connect MetaMask to Kite AI chain (include chain params)
8. Expected state after setup: paper mode active, NN loading pretrained weights,
   news feed connecting, dashboard showing live data within 2 minutes.

## Vercel deployment instructions (for frontend only)
- Set all NEXT_PUBLIC_ env vars in Vercel dashboard.
- Backend must be deployed separately (Railway, Render, or AWS EC2).
- Note: multiprocessing requires a persistent server — Vercel serverless is NOT compatible with the backend. Use a VPS or container service.

## AWS EC2 one-liner deploy
  `docker-compose -f docker-compose.prod.yml up -d`
  (uses the prod variant with resource limits and logging config)
