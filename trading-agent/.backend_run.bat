@echo off 
cd /d "C:\Users\eivan\Desktop\The-Wolf-of-Wallstreet\trading-agent\backend" 
if not exist .venv\Scripts\python.exe ( 
    echo [Backend] Creating virtual environment... 
    python -m venv .venv 
) 
echo [Backend] Installing Python dependencies... 
call .venv\Scripts\python.exe -m pip install --upgrade pip 
call .venv\Scripts\python.exe -m pip install -r "..\requirements.txt" 
echo [Backend] Launching service... 
set "PYTHONPATH=%CD%\.." 
.venv\Scripts\python.exe main.py 
echo. 
echo Backend exited. Press any key to close this window. 
pause 
