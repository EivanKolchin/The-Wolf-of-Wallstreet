@echo off
REM ----------------------------------------------------------------------
REM  redis_up.bat — best-effort local Redis for the trading agent.
REM  NON-BLOCKING: if a Redis is already up we use it; if Docker's engine is
REM  already running we (re)start the wolf-redis container; otherwise we SKIP
REM  IMMEDIATELY. The backend (main.py) then falls back to a portable Windows
REM  Redis or in-memory FakeRedis, so start.bat never hangs on this step.
REM  Always exits 0 (the caller continues regardless).
REM ----------------------------------------------------------------------
setlocal enabledelayedexpansion
set "REDIS_NAME=wolf-redis"
set "REDIS_PORT=6379"

REM 1) Already reachable on :6379 (Docker / Memurai / portable / WSL)? Fast TCP probe.
powershell -NoProfile -Command "try { $c = New-Object Net.Sockets.TcpClient; $c.Connect('127.0.0.1',%REDIS_PORT%); $c.Close(); exit 0 } catch { exit 1 }" >nul 2>nul
if not errorlevel 1 (
    echo [Redis] Already running on port %REDIS_PORT%.
    endlocal & exit /b 0
)

REM 2) Docker CLI present at all?
where docker >nul 2>nul
if errorlevel 1 (
    echo [Redis] Nothing on :%REDIS_PORT% and no Docker — backend will use its built-in fallback ^(portable Redis / in-memory^).
    endlocal & exit /b 0
)

REM 3) Is the Docker ENGINE already responsive? Bounded to 6s so a stopped/starting
REM    engine (Docker Desktop first-run not finished) can NEVER hang the launcher.
powershell -NoProfile -Command "$o=[IO.Path]::GetTempFileName(); $p = Start-Process -FilePath 'docker' -ArgumentList 'info' -NoNewWindow -PassThru -RedirectStandardOutput $o -RedirectStandardError ($o+'.err'); if ($p.WaitForExit(6000)) { exit $p.ExitCode } else { try { $p.Kill() } catch {}; exit 1 }" >nul 2>nul
if errorlevel 1 (
    echo [Redis] Docker engine not ready — skipping ^(backend fallback handles Redis^). Start Docker Desktop first if you want Docker Redis.
    endlocal & exit /b 0
)

REM 4) Engine is up and responsive — create or start the container (idempotent, quick).
docker ps -a --format "{{.Names}}" | findstr /x "%REDIS_NAME%" >nul 2>nul
if errorlevel 1 (
    echo [Redis] Creating container %REDIS_NAME% on port %REDIS_PORT% ^(first run may pull redis:7^)...
    docker run -d --name %REDIS_NAME% -p %REDIS_PORT%:6379 --restart unless-stopped redis:7 >nul 2>nul
) else (
    echo [Redis] Starting existing container %REDIS_NAME%...
    docker start %REDIS_NAME% >nul 2>nul
)
echo [Redis] Docker Redis requested at redis://localhost:%REDIS_PORT%/0
endlocal & exit /b 0
