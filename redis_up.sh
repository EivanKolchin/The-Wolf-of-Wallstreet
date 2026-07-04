#!/bin/bash
# ----------------------------------------------------------------------
#  redis_up.sh — best-effort local Redis for the trading agent.
#  NON-BLOCKING: if a Redis is already up we use it; if Docker's engine is
#  responsive we (re)start the wolf-redis container; otherwise we SKIP
#  IMMEDIATELY. The backend falls back to a portable/in-memory Redis, so the
#  launcher never hangs on this step. Always exits 0 (caller continues).
# ----------------------------------------------------------------------
REDIS_NAME=wolf-redis
REDIS_PORT=6379

# 1) Already reachable on :6379 (Docker / package-manager Redis / WSL)? Fast TCP probe.
if timeout 1 bash -c "echo > /dev/tcp/127.0.0.1/$REDIS_PORT" >/dev/null 2>&1; then
    echo "[Redis] Already running on port $REDIS_PORT."
    exit 0
fi

# 2) Docker CLI present at all?
if ! command -v docker >/dev/null 2>&1; then
    echo "[Redis] Nothing on :$REDIS_PORT and no Docker — backend will use its built-in fallback (portable / in-memory)."
    exit 0
fi

# 3) Is the Docker engine responsive? Bounded to 6s so a stopped/starting daemon can't hang the launcher.
if ! timeout 6 docker info >/dev/null 2>&1; then
    echo "[Redis] Docker engine not ready — skipping (backend fallback handles Redis). Start Docker if you want Docker Redis."
    exit 0
fi

# 4) Engine is up — create or start the container (idempotent, quick).
if docker ps -a --format '{{.Names}}' | grep -qx "$REDIS_NAME"; then
    echo "[Redis] Starting existing container $REDIS_NAME..."
    timeout 20 docker start "$REDIS_NAME" >/dev/null 2>&1
else
    echo "[Redis] Creating container $REDIS_NAME on port $REDIS_PORT (first run may pull redis:7)..."
    timeout 90 docker run -d --name "$REDIS_NAME" -p "$REDIS_PORT:6379" --restart unless-stopped redis:7 >/dev/null 2>&1
fi
echo "[Redis] Docker Redis requested at redis://localhost:$REDIS_PORT/0"
exit 0
