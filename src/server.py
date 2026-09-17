"""
RPS Opponent Modeling Interactive Web Server
============================================
A lightweight HTTP and REST API server using Python's standard library
to provide a web UI for the Adaptive Markov Agent.
"""

import http.server
import json
import os
import socketserver
import urllib.parse
import webbrowser
from typing import Dict

import numpy as np

from rps_agent import (
    AdaptiveAgent,
    BiasedStreakOpponent,
    COUNTER_MOVES,
    MOVE_NAMES,
    MOVES,
    PureRandomOpponent,
    RotatingOpponent,
    evaluate_round,
    run_matchup,
)

PORT = 5000
WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Global live game state for web player
live_agent = AdaptiveAgent(order=2)
game_history = {
    "human_moves": [],
    "agent_moves": [],
    "predicted_moves": [],
    "outcomes": [],
    "human_wins": 0,
    "agent_wins": 0,
    "ties": 0,
}


class RPSRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path == "/" or path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            html_file = os.path.join(WEB_DIR, "index.html")
            with open(html_file, "rb") as f:
                self.wfile.write(f.read())
            return

        elif path == "/docs/winrate_plot.png":
            plot_file = os.path.join(PROJECT_ROOT, "docs", "winrate_plot.png")
            if os.path.exists(plot_file):
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.end_headers()
                with open(plot_file, "rb") as f:
                    self.wfile.write(f.read())
                return
            else:
                self.send_error(404, "Plot image not found")
                return

        elif path == "/api/state":
            self.send_json_response(self.get_current_state())
            return

        elif path == "/api/run_benchmark":
            rounds = 1000
            order = 2
            seed = 42

            opponents = [
                PureRandomOpponent(seed=seed),
                RotatingOpponent(cycle=["R", "P", "S"]),
                BiasedStreakOpponent(weights={"R": 0.50, "P": 0.30, "S": 0.20}, streak_prob=0.60, seed=seed),
            ]

            data = []
            for opp in opponents:
                a_agent = AdaptiveAgent(order=order, seed=seed)
                from rps_agent import RandomAgent
                r_agent = RandomAgent(seed=seed)

                a_res = run_matchup(a_agent, opp, rounds=rounds)
                r_res = run_matchup(r_agent, opp, rounds=rounds)

                data.append({
                    "opponent_name": opp.name,
                    "adaptive_win_rate": round(a_res["win_rate"] * 100, 1),
                    "adaptive_loss_rate": round(a_res["loss_rate"] * 100, 1),
                    "adaptive_tie_rate": round(a_res["tie_rate"] * 100, 1),
                    "random_win_rate": round(r_res["win_rate"] * 100, 1),
                    "delta_win_rate": round((a_res["win_rate"] - r_res["win_rate"]) * 100, 1),
                })

            self.send_json_response({"status": "success", "results": data})
            return

        super().do_GET()

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length > 0 else b"{}"

        try:
            payload = json.loads(body.decode("utf-8"))
        except Exception:
            payload = {}

        if path == "/api/play":
            human_move = payload.get("move", "").upper()
            if human_move not in MOVES:
                self.send_json_response({"error": f"Invalid move. Choose from {MOVES}"}, status=400)
                return

            # Predict next move before updating model with current move
            predicted_opponent_move = live_agent.model.predict_next_move()
            agent_move = COUNTER_MOVES[predicted_opponent_move]

            # Evaluate outcome
            outcome = evaluate_round(agent_move, human_move)

            # Update model with human move
            live_agent.observe(human_move)

            # Update history
            game_history["human_moves"].append(human_move)
            game_history["agent_moves"].append(agent_move)
            game_history["predicted_moves"].append(predicted_opponent_move)
            game_history["outcomes"].append(outcome)

            if outcome == "WIN":
                game_history["agent_wins"] += 1
            elif outcome == "LOSS":
                game_history["human_wins"] += 1
            else:
                game_history["ties"] += 1

            response_data = self.get_current_state()
            response_data["last_round"] = {
                "human_move": human_move,
                "human_move_name": MOVE_NAMES[human_move],
                "agent_move": agent_move,
                "agent_move_name": MOVE_NAMES[agent_move],
                "predicted_move": predicted_opponent_move,
                "predicted_move_name": MOVE_NAMES[predicted_opponent_move],
                "outcome": outcome,
            }
            self.send_json_response(response_data)
            return

        elif path == "/api/reset":
            live_agent.reset()
            game_history["human_moves"].clear()
            game_history["agent_moves"].clear()
            game_history["predicted_moves"].clear()
            game_history["outcomes"].clear()
            game_history["human_wins"] = 0
            game_history["agent_wins"] = 0
            game_history["ties"] = 0

            self.send_json_response(self.get_current_state())
            return

        self.send_error(404, "Endpoint not found")

    def get_current_state(self) -> Dict:
        total = game_history["human_wins"] + game_history["agent_wins"] + game_history["ties"]
        ai_win_rate = (game_history["agent_wins"] / total * 100) if total > 0 else 0.0
        human_win_rate = (game_history["human_wins"] / total * 100) if total > 0 else 0.0
        tie_rate = (game_history["ties"] / total * 100) if total > 0 else 0.0

        # Extract Markov transition counts for display
        order2_transitions = {}
        for context, counter in live_agent.model.n_grams.get(2, {}).items():
            context_key = f"{context[0]}->{context[1]}"
            order2_transitions[context_key] = dict(counter)

        order1_transitions = {}
        for context, counter in live_agent.model.n_grams.get(1, {}).items():
            order1_transitions[context[0]] = dict(counter)

        unigram_counts = dict(live_agent.model.unigram)

        return {
            "total_rounds": total,
            "human_wins": game_history["human_wins"],
            "agent_wins": game_history["agent_wins"],
            "ties": game_history["ties"],
            "ai_win_rate": round(ai_win_rate, 1),
            "human_win_rate": round(human_win_rate, 1),
            "tie_rate": round(tie_rate, 1),
            "history": [
                {
                    "round": i + 1,
                    "human": game_history["human_moves"][i],
                    "agent": game_history["agent_moves"][i],
                    "pred": game_history["predicted_moves"][i],
                    "outcome": game_history["outcomes"][i],
                }
                for i in range(len(game_history["human_moves"]))
            ][-15:],  # Return last 15 rounds
            "model_stats": {
                "order": live_agent.model.order,
                "unigrams": unigram_counts,
                "order1_transitions": order1_transitions,
                "order2_transitions": order2_transitions,
            },
        }

    def send_json_response(self, data: Dict, status: int = 200):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def start_server(port: int = PORT, auto_open: bool = True):
    os.makedirs(WEB_DIR, exist_ok=True)
    server_address = ("", port)
    
    # Allow address reuse
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(server_address, RPSRequestHandler) as httpd:
        url = f"http://localhost:{port}"
        print(f"\n" + "=" * 65)
        print(f"  RPS OPPONENT MODELING WEB APPLICATION RUNNING")
        print(f"  URL: {url}")
        print("=" * 65)
        print(f"  Press Ctrl+C to stop the server.\n")
        
        if auto_open:
            webbrowser.open(url)
            
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server...")


if __name__ == "__main__":
    start_server(PORT, auto_open=False)
