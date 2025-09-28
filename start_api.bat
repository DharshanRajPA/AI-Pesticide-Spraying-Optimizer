@echo off
echo Starting AgriSprayAI API Server...
echo ====================================
cd /d "C:\Dharshan Raj P A\College\Projects\TaRp\AI-Pesticide-Spraying-Optimizer"
call venv\Scripts\activate
echo API Server starting on http://localhost:8000
echo Keep this window open!
echo.
python code/api/server.py
pause
