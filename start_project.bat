@echo off
echo 🌱 AgriSprayAI - Easy Startup for Windows
echo ==========================================
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

REM Run the startup script
python start_project.py

pause
