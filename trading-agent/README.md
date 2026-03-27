# Trading Agent

A full-stack algorithmic trading platform specifically built around autonomous trading execution using a Python agent with a Next.js UI frontend. 

## Structure
- \`/backend\`: Python 3.11 core algorithmic agent using structlog, Pydantic settings.
- \`/frontend\`: Next.js 14 frontend using TypeScript, TailwindCSS, and shadcn/ui.
- Docker containers run Postgres 16, Redis 7, the Next.js frontend, and the Python backend on a shared network.

## Getting Started

1. Copy \`.env.example\` to \`.env\` and fill it with your API keys.
2. \`docker-compose up -d --build\`
3. Next.js app will be accessible at [http://localhost:3000](http://localhost:3000)
4. Fastapi backend will be accessible at [http://localhost:8000](http://localhost:8000)
