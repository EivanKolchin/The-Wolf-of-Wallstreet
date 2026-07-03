#!/bin/bash
# ----------------------------------------------------------------------
#  redis_up.sh — bring up a local Redis for the trading agent via Docker.
#  Idempotent: creates the container once, then just starts it next time.
#  Matches the default REDIS_URL=redis://localhost:6379/0.
#  Exits 0 on success, 1 if Docker is unavailable (caller may continue).
# ----------------------------------------------------------------------
REDIS_NAME=wolf-redis
REDIS_PORT=6379

if ! command -v docker >/dev/null 2>&1; then
    echo "[Redis] Docker CLI not found. Install Docker (or run Redis via your package manager)."
    exit 1
fi

if ! docker info >/dev/null 2>&1; then
    echo "[Redis] Docker engine not running. Start Docker Desktop / dockerd and re-run."
    exit 1
fi

if docker ps -a --format '{{.Names}}' | grep -qx "$REDIS_NAME"; then
    echo "[Redis] Starting existing container $REDIS_NAME..."
    docker start "$REDIS_NAME" >/dev/null
else
    echo "[Redis] Creating container $REDIS_NAME on port $REDIS_PORT..."
    docker run -d --name "$REDIS_NAME" -p "$REDIS_PORT:6379" --restart unless-stopped redis:7 >/dev/null
fi

sleep 2
if docker exec "$REDIS_NAME" redis-cli ping 2>/dev/null | grep -qi PONG; then
    echo "[Redis] Ready (PONG) at redis://localhost:$REDIS_PORT/0"
else
    echo "[Redis] WARNING: no PONG yet; the container may still be starting."
fi
exit 0
