@echo off
setlocal enabledelayedexpansion

if "%~1"=="--backend" goto backend_runner

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
if not exist "node_modules\" (
    echo [Frontend] node_modules not found. Installing dependencies...
    call npm install --legacy-peer-deps
)
start "AI Trading Agent - Frontend" cmd /k "npm run dev || (echo [Frontend] Run failed, attempting dependency install... ^& call npm install --legacy-peer-deps ^& npm run dev) || echo Frontend crashed. Please check logs. ^& pause"
popd

:: Backend
echo [Backend] Starting in a new window...
start "AI Trading Agent - Backend" cmd /k ""%~f0" --backend "%CD%\backend" "%PYTHON_CMD%""

popd
echo.
echo Frontend and backend windows launched. Keep them open to see logs.
echo This launcher window will close automatically...
timeout /t 3 >nul
endlocal
exit /b 0

:backend_runner
set "BDIR=%~2"
set "PYCMD=%~3"
cd /d "%BDIR%"
if not exist .venv\Scripts\python.exe (
    echo [Backend] Creating virtual environment...
    %PYCMD% -m venv .venv
    echo [Backend] Installing Python dependencies...
    call .venv\Scripts\python.exe -m pip install --upgrade pip
    call .venv\Scripts\python.exe -m pip install -r "..\requirements.txt"
)

echo [Backend] Note: Ensure you have added your hosted DATABASE_URL and REDIS_URL to the .env file!
set "PYTHONPATH=%CD%\.."

:backend_loop
echo [Backend] Launching service...
.venv\Scripts\python.exe main.py

if %ERRORLEVEL% neq 0 (
    echo [Backend] Execution failed or deliberate restart. Attempting to check dependencies...
    call .venv\Scripts\python.exe -m pip install -r "..\requirements.txt"
    echo [Backend] Relaunching service...
    goto backend_loop
) else (
    echo [Backend] Service restarting smoothly...
    goto backend_loop
)

echo.
echo Backend exited. Press any key to close this window.
pause
exit /b 0
