@echo off
echo 🌱 AgriSprayAI System Starter
echo =============================
echo.

REM Check if Python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python is not installed or not in PATH
    echo Please install Python 3.9-3.11 from https://python.org
    pause
    exit /b 1
)

echo ✅ Python found
echo.

REM Start the system
python start_system.py

pause
