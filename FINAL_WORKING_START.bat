@echo off
title AgriSprayAI - Final Working Start
color 0A

echo.
echo ========================================
echo   AgriSprayAI - FINAL WORKING START
echo ========================================
echo.

echo [1/4] Checking project directory...
cd /d "C:\Dharshan Raj P A\College\Projects\TaRp\AI-Pesticide-Spraying-Optimizer"
if not exist "code\api\server.py" (
    echo ERROR: Project files not found!
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
echo [3/4] Checking models...
if not exist "models\yolov8_baseline\weights\best.pt" (
    echo WARNING: Models not found. Running download...
    call venv\Scripts\activate
    python download_models.py
    if errorlevel 1 (
        echo ERROR: Failed to download models
        pause
        exit /b 1
    )
)
echo OK - Models found

echo.
echo [4/4] Starting API Server...
echo Starting API server on http://localhost:8000
echo Keep this window open!
echo.
call venv\Scripts\activate
python code\api\server.py

echo.
echo API Server stopped. Press any key to exit...
pause
