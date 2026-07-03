@echo off
REM ----------------------------------------------------------------------
REM  redis_up.bat — bring up a local Redis for the trading agent via Docker.
REM  Idempotent: creates the container once, then just starts it next time.
REM  Matches the default REDIS_URL=redis://localhost:6379/0.
REM  Exits 0 on success, 1 if Docker is unavailable (caller may continue).
REM ----------------------------------------------------------------------
setlocal enabledelayedexpansion
set "REDIS_NAME=wolf-redis"
set "REDIS_PORT=6379"

where docker >nul 2>nul
if errorlevel 1 (
    echo [Redis] Docker CLI not found. Install Docker Desktop ^(or run Redis via WSL/Memurai^).
    exit /b 1
)

docker info >nul 2>nul
if errorlevel 1 (
    echo [Redis] Docker engine not running. Launching Docker Desktop...
    start "" "%ProgramFiles%\Docker\Docker\Docker Desktop.exe"
    echo [Redis] Waiting for the Docker engine ^(up to ~120s^)...
    set /a tries=0
    :waitloop
    timeout /t 4 >nul
    docker info >nul 2>nul
    if not errorlevel 1 goto dockerready
    set /a tries+=1
    if !tries! geq 30 (
        echo [Redis] Docker did not become ready in time. Start Docker Desktop manually and re-run.
        exit /b 1
    )
    goto waitloop
)
:dockerready

docker ps -a --format "{{.Names}}" | findstr /x "%REDIS_NAME%" >nul 2>nul
if errorlevel 1 (
    echo [Redis] Creating container %REDIS_NAME% on port %REDIS_PORT%...
    docker run -d --name %REDIS_NAME% -p %REDIS_PORT%:6379 --restart unless-stopped redis:7 >nul
) else (
    echo [Redis] Starting existing container %REDIS_NAME%...
    docker start %REDIS_NAME% >nul
)

timeout /t 2 >nul
docker exec %REDIS_NAME% redis-cli ping 2>nul | findstr /i PONG >nul
if errorlevel 1 (
    echo [Redis] WARNING: no PONG yet; the container may still be starting.
) else (
    echo [Redis] Ready ^(PONG^) at redis://localhost:%REDIS_PORT%/0
)
endlocal
exit /b 0
