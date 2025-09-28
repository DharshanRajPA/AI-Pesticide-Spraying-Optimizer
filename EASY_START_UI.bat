@echo off
title AgriSprayAI - Web Interface
color 0B

echo.
echo ========================================
echo   AgriSprayAI - WEB INTERFACE START
echo ========================================
echo.

echo [1/3] Checking project directory...
cd /d "C:\Dharshan Raj P A\College\Projects\TaRp\AI-Pesticide-Spraying-Optimizer\ui"
if not exist "package.json" (
    echo ERROR: UI directory not found!
    echo Please make sure you're in the correct directory.
    pause
    exit /b 1
)
echo OK - UI directory found

echo.
echo [2/3] Checking Node.js...
node --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Node.js not found!
    echo Please install Node.js from https://nodejs.org
    pause
    exit /b 1
)
echo OK - Node.js found

echo.
echo [3/3] Starting Web Interface...
echo Starting web interface on http://localhost:3000
echo Keep this window open!
echo.
npm start

echo.
echo Web Interface stopped. Press any key to exit...
pause
