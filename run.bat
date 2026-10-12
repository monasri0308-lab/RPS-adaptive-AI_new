@echo off
echo =========================================================================
echo   Rock-Paper-Scissors Probabilistic AI Launcher
echo =========================================================================
echo.
echo [1/3] Running 1,000-Round Benchmark Suite across 8 Opponents...
python src\rps_agent.py
echo.
echo [2/3] Opening Benchmark Visualizations...
start "" "docs\winrate_plot.png"
start "" "docs\probability_plot.png"
echo.
echo [3/3] Launching Interactive Web Application (http://localhost:5000)...
start "" "http://localhost:5000"
python src\server.py
pause
