"""
Main entry point for Rock-Paper-Scissors Opponent Modeling Suite
================================================================
Coordinates execution across benchmark simulations, the interactive web application,
and the live terminal console match.
"""

import argparse
import os
import sys
import webbrowser

# Establish absolute directories and configure import paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")
DOCS_DIR = os.path.join(BASE_DIR, "docs")

for path in [BASE_DIR, SRC_DIR]:
    if path not in sys.path:
        sys.path.insert(0, path)

try:
    from rps_agent import play_interactive, run_benchmark
    from server import start_server
except ImportError:
    from src.rps_agent import play_interactive, run_benchmark
    from src.server import start_server


def parse_args():
    """Parse command-line arguments for flexible execution."""
    parser = argparse.ArgumentParser(
        description="Rock-Paper-Scissors Probabilistic Opponent Modeling AI Suite"
    )
    parser.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="Run live interactive match in console against the adaptive agent",
    )
    parser.add_argument(
        "-w",
        "--web",
        "--server",
        action="store_true",
        dest="server_only",
        help="Start the interactive web server without running the initial benchmark",
    )
    parser.add_argument(
        "-p",
        "--port",
        type=int,
        default=5000,
        help="Port number for the web server (default: 5000)",
    )
    parser.add_argument(
        "-r",
        "--rounds",
        type=int,
        default=1000,
        help="Simulation rounds per opponent in the benchmark (default: 1000)",
    )
    parser.add_argument(
        "-o",
        "--order",
        type=int,
        default=2,
        help="Markov Chain context order (default: 2)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic reproducibility (default: 42)",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Disable automatic browser launching",
    )
    return parser.parse_args()


def main():
    """Main execution orchestrator."""
    args = parse_args()
    os.makedirs(DOCS_DIR, exist_ok=True)

    winrate_plot = os.path.join(DOCS_DIR, "winrate_plot.png")
    prob_plot = os.path.join(DOCS_DIR, "probability_plot.png")

    try:
        if args.interactive:
            play_interactive(order=args.order)
        elif args.server_only:
            start_server(port=args.port, auto_open=not args.no_browser)
        else:
            print("\n" + "=" * 75)
            print("  --- [1/2] Running Opponent Modeling Benchmark (1,000 Rounds) ---")
            print("=" * 75)
            run_benchmark(
                rounds=args.rounds,
                order=args.order,
                seed=args.seed,
                plot_path=winrate_plot,
                prob_plot_path=prob_plot,
            )

            print("\n" + "=" * 75)
            print("  --- [2/2] Opening Visualizations & Launching Web Server ---")
            print("=" * 75)

            if not args.no_browser:
                try:
                    if os.path.exists(winrate_plot):
                        webbrowser.open(os.path.abspath(winrate_plot))
                except Exception:
                    pass

            start_server(port=args.port, auto_open=not args.no_browser)

    except KeyboardInterrupt:
        print("\n\nExecution terminated by user. Goodbye!")
        sys.exit(0)


if __name__ == "__main__":
    main()
