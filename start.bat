@echo off
setlocal

echo ==============================================
echo    AI Trading Agent - Windows Startup Script
echo ==============================================

:: Navigate to the project directory
if exist "trading-agent" (
    cd trading-agent
) else (
    echo Error: trading-agent directory not found. Please run this script from the workspace root.
    pause
    exit /b 1
)

:: 1. Start the Frontend in a separate window
echo [Frontend] Starting...
cd frontend
:: Try running npm start/dev. If it immediately errors (e.g. missing modules check), handle it.
if not exist node_modules (
    echo [Frontend] 'node_modules' missing. Installing dependencies...
    call npm install
)
:: Launch Next.js in a separate terminal process
start "AI Trading Agent - Frontend" cmd /c "npm run dev || (echo Frontend crashed. Please check logs. & pause)"
cd ..

:: 2. Start the Backend in this window
echo [Backend] Starting...
cd backend

:: Try to run the backend first
python main.py

:: If it exits with an error code, assume missing libraries and try installing them
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ==============================================
    echo Backend execution failed (Exit Code: %ERRORLEVEL%).
    echo Likely missing libraries. Installing from requirements.txt...
    echo ==============================================
    
    pip install -r requirements.txt
    
    echo.
    echo [Backend] Retrying execution...
    python main.py
    
    if %ERRORLEVEL% NEQ 0 (
        echo [Backend] Execution failed again. Check database configs or Python errors above.
        pause
    )
)

cd ..\..
endlocal
