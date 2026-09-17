@echo off
echo ===================================================
echo   Rock-Paper-Scissors Opponent Modeling Launcher
echo ===================================================
echo.
echo [1] Running simulation benchmarks...
python src\rps_agent.py
echo.
echo [2] Opening plot and starting interactive web app...
start "" "docs\winrate_plot.png"
start "" "http://localhost:5000"
python src\server.py
pause
