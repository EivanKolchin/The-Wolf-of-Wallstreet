@echo off
REM ----------------------------------------------------------------------
REM  One-click backend launcher.
REM
REM  1) Runs DNS pre-flight (probes Alpaca/Binance/LLM hosts).
REM  2) If system DNS can't reach them, installs Cloudflare DoH (1.1.1.1)
REM     as an in-process fallback for the life of THIS process tree only —
REM     no admin rights, no Windows DNS changes.
REM  3) Boots the backend with the project venv + correct PYTHONPATH.
REM ----------------------------------------------------------------------
setlocal
cd /d "%~dp0"

set "PY=%~dp0backend\.venv\Scripts\python.exe"
if not exist "%PY%" (
    echo [start_backend] Project venv not found at backend\.venv — falling back to system python.
    set "PY=python"
)

set "PYTHONPATH=%~dp0;%~dp0backend"

echo.
echo === DNS pre-flight ============================================================
"%PY%" -m backend.core.network_check
echo ===============================================================================
echo.
echo === IB Gateway pre-flight ====================================================
REM Detect/launch IB Gateway and wait up to 120s for the API port to open.
REM If the user doesn't use IBKR (no install), this is a no-op + clear message.
"%PY%" -m backend.core.ibkr_launcher --wait 120
echo ===============================================================================
echo.

echo [start_backend] Launching backend on http://0.0.0.0:8000 ...
"%PY%" backend\main.py

set "RC=%ERRORLEVEL%"
echo.
echo [start_backend] Backend exited with code %RC%.
pause
endlocal
