@echo off
setlocal enabledelayedexpansion

echo ==============================================
echo    AI Trading Agent - Windows Startup Script
echo ==============================================

:: Node check
where node >nul 2>nul
if errorlevel 1 (
    echo Node.js not found. Install Node 18 or higher and ensure it is on PATH.
    pause
    exit /b 1
)

echo CHECKPOINT_NODE_OK

:: Python check
set "PYTHON_CMD="
for %%P in (python python3) do (
    if not defined PYTHON_CMD (
        where %%P >nul 2>nul
        if not errorlevel 1 set "PYTHON_CMD=%%P"
    )
)
if not defined PYTHON_CMD (
    echo Python not found. Install Python 3.10 or higher and ensure it is on PATH.
    pause
    exit /b 1
)

:: Detect project directory
set "PROJ_DIR=."
if not exist "%PROJ_DIR%\backend" (
    if exist "trading-agent\backend" (
        set "PROJ_DIR=trading-agent"
    ) else (
        echo Error: trading-agent directory not found. Run this script from the repository root.
        pause
        exit /b 1
    )
)

pushd "%PROJ_DIR%"

:: Ensure .env exists
if not exist ".env" (
    echo [Setup] .env missing. Copying .env.example...
    if exist ".env.example" (
        copy ".env.example" ".env" >nul
    ) else (
        echo [Warning] .env.example not found. Continuing without copy.
    )
)

:: Frontend
echo [Frontend] Starting...
pushd frontend
set NEED_FE_INSTALL=
if not exist "node_modules\.bin\next" set NEED_FE_INSTALL=1
if not exist "node_modules\.bin\next.cmd" set NEED_FE_INSTALL=1
if not exist "node_modules\autoprefixer" set NEED_FE_INSTALL=1
if defined NEED_FE_INSTALL (
    echo [Frontend] Installing dependencies ^(legacy peer deps^)...
    call npm install --legacy-peer-deps
)
start "AI Trading Agent - Frontend" cmd /k "npm run dev || echo Frontend crashed. Please check logs. ^& pause"
popd

:: Backend
echo [Backend] Starting in a new window...
set "BDIR=%CD%\backend"

echo @echo off > ".backend_run.bat"
echo cd /d "%BDIR%" >> ".backend_run.bat"
echo if not exist .venv\Scripts\python.exe ( >> ".backend_run.bat"
echo     echo [Backend] Creating virtual environment... >> ".backend_run.bat"
echo     %PYTHON_CMD% -m venv .venv >> ".backend_run.bat"
echo ^) >> ".backend_run.bat"
echo echo [Backend] Installing Python dependencies... >> ".backend_run.bat"
echo call .venv\Scripts\python.exe -m pip install --upgrade pip >> ".backend_run.bat"
echo call .venv\Scripts\python.exe -m pip install -r "..\requirements.txt" >> ".backend_run.bat"
echo echo [Backend] Launching service... >> ".backend_run.bat"
echo set "PYTHONPATH=%%CD%%\.." >> ".backend_run.bat"
echo .venv\Scripts\python.exe main.py >> ".backend_run.bat"
echo echo. >> ".backend_run.bat"
echo echo Backend exited. Press any key to close this window. >> ".backend_run.bat"
echo pause >> ".backend_run.bat"

start "AI Trading Agent - Backend" cmd /k "call .backend_run.bat"

popd
echo.
echo Frontend and backend windows launched. Keep them open to see logs.
echo Press any key to close this launcher window.
pause
endlocal
