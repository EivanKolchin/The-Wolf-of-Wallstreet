@echo off
REM Launch the interactive backtest dashboard (strategies vs random vs S&P 500).
REM Usage:  launch_backtester.bat [port]      (default port 8765)
setlocal
cd /d "%~dp0"
set "PYTHONIOENCODING=utf-8"
set "PORT=8765"
if not "%~1"=="" set "PORT=%~1"

set "PY=.\backend\.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=python"

echo.
echo   Backtest dashboard -^> http://127.0.0.1:%PORT%
echo   (opening your browser in a moment; press Ctrl+C here to stop)
echo.

REM open the browser ~2s after the server starts
start "" cmd /c "timeout /t 2 >nul & start "" http://127.0.0.1:%PORT%"

"%PY%" "scripts\backtest_dashboard.py" --port %PORT%
endlocal
