@echo off
title AgriSprayAI - Easy Start
color 0A

echo.
echo ========================================
echo    AgriSprayAI - EASY START SYSTEM
echo ========================================
echo.

echo [1/4] Checking project directory...
cd /d "C:\Dharshan Raj P A\College\Projects\TaRp\AI-Pesticide-Spraying-Optimizer"
if not exist "code\api\server.py" (
    echo ERROR: Project files not found!
    echo Please make sure you're in the correct directory.
    pause
    exit /b 1
)
echo OK - Project directory found

echo.
echo [2/4] Checking virtual environment...
if not exist "venv\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found!
    echo Please run: python -m venv venv
    pause
    exit /b 1
)
echo OK - Virtual environment found

echo.
echo [3/4] Starting API Server...
echo Starting API server on http://localhost:8000
echo Keep this window open!
echo.
call venv\Scripts\activate
python code\api\server.py

echo.
echo API Server stopped. Press any key to exit...
pause
