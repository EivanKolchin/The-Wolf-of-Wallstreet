#!/bin/bash

echo "=============================================="
echo "   AI Trading Agent - UNIX Startup Script"
echo "=============================================="

# Basic prereq checks
if ! command -v node >/dev/null 2>&1; then
    echo "Node.js not found. Please install Node 18+ (Node 20 recommended) and ensure it is on PATH."
    exit 1
fi

if command -v python3 >/dev/null 2>&1; then
    PYTHON_CMD=python3
elif command -v python >/dev/null 2>&1; then
    PYTHON_CMD=python
else
    echo "Python not found. Please install Python 3.10+ and ensure it is on PATH."
    exit 1
fi

if [ -d "backend" ] && [ -d "frontend" ]; then
    PROJ_DIR="."
elif [ -d "trading-agent/backend" ]; then
    PROJ_DIR="trading-agent"
else
    echo "Error: trading-agent directory not found. Please run this script from the repository root."
    exit 1
fi

pushd "$PROJ_DIR" >/dev/null

if [ ! -f ".env" ]; then
    echo "[Setup] .env file not found. Copying .env.example..."
    if [ -f ".env.example" ]; then
        cp .env.example .env
    else
        echo "[Warning] .env.example not found. Continuing without copy."
    fi
fi

# Trap terminal exits to gracefully kill backgrounded frontend process
trap 'echo "Stopping processes..."; kill 0; exit' SIGINT SIGTERM EXIT

# 1. Start the Frontend in the background
echo "[Frontend] Starting..."
cd frontend
if [ ! -x "node_modules/.bin/next" ] || [ ! -d "node_modules/autoprefixer" ]; then
    echo "[Frontend] Installing dependencies (legacy peer deps)..."
    npm install --legacy-peer-deps
fi

npm run dev &
FRONTEND_PID=$!
cd ..

# 2. Start the Backend in a separate terminal to keep logs visible
echo "[Backend] Starting in a new terminal..."
BACKEND_CMD="cd '$PWD/backend' && if [ ! -x .venv/bin/python ]; then echo '[Backend] Creating virtual environment...' && $PYTHON_CMD -m venv .venv; fi && BACKEND_PY='./.venv/bin/python' && BACKEND_PIP='./.venv/bin/pip' && echo '[Backend] Installing Python dependencies...' && $BACKEND_PIP install --upgrade pip && $BACKEND_PIP install -r requirements.txt && echo '[Backend] Launching service...' && $BACKEND_PY main.py; echo; echo 'Backend exited. Press enter to close.'; read"

if command -v gnome-terminal >/dev/null 2>&1; then
    gnome-terminal -- bash -c "$BACKEND_CMD"
elif command -v xterm >/dev/null 2>&1; then
    xterm -hold -e "$BACKEND_CMD"
elif command -v konsole >/dev/null 2>&1; then
    konsole --noclose -e bash -c "$BACKEND_CMD"
else
    echo "[Backend] No secondary terminal found; running inline."
    bash -c "$BACKEND_CMD"
fi

popd >/dev/null
