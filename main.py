"""
Main entry point for Rock-Paper-Scissors Opponent Modeling
"""

import os
import sys
import webbrowser

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from rps_agent import run_benchmark, play_interactive
from server import start_server

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ["-i", "--interactive"]:
        play_interactive()
    elif len(sys.argv) > 1 and sys.argv[1] in ["--web", "-w", "--server"]:
        start_server(port=5000, auto_open=True)
    else:
        print("\n--- [1/2] Running Opponent Modeling Benchmark ---")
        run_benchmark(rounds=1000, order=2, seed=42, plot_path="docs/winrate_plot.png")
        print("\n--- [2/2] Opening Benchmark Plot & Starting Web Server ---")
        try:
            webbrowser.open("http://localhost:5000")
        except Exception:
            pass
        start_server(port=5000, auto_open=True)
