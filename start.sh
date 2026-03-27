#!/bin/bash

echo "=============================================="
echo "   AI Trading Agent - UNIX Startup Script"
echo "=============================================="

# Enable exit on failure for critical checks, though we will handle python execution manually
cd trading-agent 2>/dev/null || {
    echo "Error: trading-agent directory not found. Please run this script from the workspace root."
    exit 1
}

# Trap terminal exits to gracefully kill backgrounded frontend process
trap 'echo "Stopping processes..."; kill 0; exit' SIGINT SIGTERM EXIT

# 1. Start the Frontend in the background
echo "[Frontend] Starting..."
cd frontend
if [ ! -d "node_modules" ]; then
    echo "[Frontend] 'node_modules' missing. Installing dependencies..."
    npm install
fi

# Run frontend in the background and pipe output to terminal seamlessly
npm run dev &
FRONTEND_PID=$!
cd ..

# 2. Start the Backend in the foreground block
echo "[Backend] Starting..."
cd backend

# Execute python code
python3 main.py
BACKEND_EXIT_CODE=$?

# Analyze if program errored on boot
if [ $BACKEND_EXIT_CODE -ne 0 ]; then
    echo ""
    echo "=============================================="
    echo "Backend execution failed (Exit Code: $BACKEND_EXIT_CODE)."
    echo "Likely missing libraries. Installing from requirements.txt..."
    echo "=============================================="
    
    # Try fetching the pip requirements
    pip3 install -r requirements.txt
    
    echo ""
    echo "[Backend] Retrying execution..."
    python3 main.py
    
    if [ $? -ne 0 ]; then
        echo "[Backend] Execution failed again. Check the database/Redis configs or terminal logs above."
    fi
fi

cd ../..
