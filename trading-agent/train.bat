@echo off
REM ----------------------------------------------------------------------
REM  One-click training launcher (v2.1 schema).
REM
REM  Trains ImprovedTradingLSTM offline on historical OHLCV with:
REM    - 70-feature v2.1 sequences (real HTF [62:70] + macro [53:57])
REM    - PnL-magnitude-weighted CE loss (Phase 16)
REM    - ATR-multiple exit heads (Phase 17)
REM    - Next-K_FUTURE OHLC delta head (Phase 18 v2)
REM
REM  Output: models/pretrain_v2_best.pt
REM  Promoted to: models/trading_lstm_latest.pt (live model loads this on boot)
REM
REM  Default args produce a useful first training pass. Override:
REM    train.bat --epochs 80 --batch-size 256 --start-year 2022 --start-month 1
REM
REM  Add --skip-download to reuse already-cached CSVs in data/.
REM ----------------------------------------------------------------------
setlocal

REM Repo root = location of this file
cd /d "%~dp0"

REM Pick the Python interpreter
set "PY=%~dp0backend\.venv\Scripts\python.exe"
if not exist "%PY%" (
    echo [WARN] Project venv not found at backend\.venv — falling back to system python.
    set "PY=python"
)

REM Ensure project paths are importable
set "PYTHONPATH=%~dp0;%~dp0backend"

REM DNS pre-flight: probe critical hosts and install Cloudflare DoH fallback
REM into THIS process tree if system DNS is broken. No admin rights needed.
echo Running DNS pre-flight check...
"%PY%" -m backend.core.network_check

REM Default training settings (override on command line)
set "DEFAULT_ARGS=--epochs 40 --batch-size 128"

if "%~1"=="" (
    echo Launching training with defaults: %DEFAULT_ARGS%
    "%PY%" scripts\pretrain.py %DEFAULT_ARGS%
) else (
    "%PY%" scripts\pretrain.py %*
)

set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" (
    echo.
    echo [ERROR] Training exited with code %RC%.
    echo Common causes: missing pandas_ta, no internet for OHLCV download, low RAM.
    pause
    exit /b %RC%
)

echo.
echo [OK] Training finished. Restart the backend to pick up models\trading_lstm_latest.pt
pause
endlocal
