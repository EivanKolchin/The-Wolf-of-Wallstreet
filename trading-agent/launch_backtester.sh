#!/usr/bin/env bash
# Launch the interactive backtest dashboard (strategies vs random vs S&P 500).
# Usage:  ./launch_backtester.sh [port]      (default port 8765)
set -euo pipefail
cd "$(dirname "$0")"
PORT="${1:-8765}"
export PYTHONIOENCODING=utf-8

# Prefer the project venv (POSIX or Windows layout), else fall back to python3.
if [ -x "./backend/.venv/bin/python" ]; then
  PY="./backend/.venv/bin/python"
elif [ -x "./backend/.venv/Scripts/python.exe" ]; then
  PY="./backend/.venv/Scripts/python.exe"
else
  PY="python3"
fi

echo
echo "  Backtest dashboard -> http://127.0.0.1:${PORT}"
echo "  (opening your browser in a moment; press Ctrl+C here to stop)"
echo

# Open the browser ~2s after the server starts (best-effort, cross-platform).
(
  sleep 2
  URL="http://127.0.0.1:${PORT}"
  if command -v xdg-open >/dev/null 2>&1; then xdg-open "$URL"
  elif command -v open >/dev/null 2>&1; then open "$URL"
  elif command -v start >/dev/null 2>&1; then start "" "$URL"
  fi
) >/dev/null 2>&1 &

exec "$PY" scripts/backtest_dashboard.py --port "$PORT"
