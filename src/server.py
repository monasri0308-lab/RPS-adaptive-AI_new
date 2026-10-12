"""
RPS Probabilistic Opponent Modeling Web Server
==============================================
A high-performance REST API and web application server providing:
1. Real-time interactive play with SHA-256 fair-play cryptographic commitment verification.
2. Bayesian posterior probability distribution and expected utility streaming.
3. Rule-based strategy detection and natural language explainability.
4. Interactive CSP constraint-satisfaction adversary lab.
5. Multi-agent 1000-round benchmark suite execution.
6. GDPR/CCPA data privacy consent and Right-to-be-Forgotten data wiping.
"""

import http.server
import json
import os
import socketserver
import sys
import urllib.parse
import webbrowser
from typing import Any, Dict

# Ensure src is in python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from bayesian_model import BayesianOpponent, BayesianPredictor
from csp_adversary import CSPAdversaryOpponent, CSPSequenceGenerator
from probabilistic_agent import ProbabilisticOpponentModeller
from reasoning_layer import (
    CopycatOpponent,
    CyclingOpponent,
    WinStayLoseShiftOpponent,
    evaluate_round,
)
from rps_agent import (
    COUNTER_MOVES,
    MOVE_NAMES,
    MOVES,
    BiasedStreakOpponent,
    PureRandomOpponent,
    RotatingOpponent,
    run_benchmark,
)

PORT = 5000
WEB_DIR = os.path.join(current_dir, "web")
PROJECT_ROOT = os.path.abspath(os.path.join(current_dir, ".."))

# Global live game state
live_agent = ProbabilisticOpponentModeller(order=2, prior_alpha=1.0)
game_history = {
    "human_moves": [],
    "agent_moves": [],
    "predicted_moves": [],
    "outcomes": [],
    "human_wins": 0,
    "agent_wins": 0,
    "ties": 0,
    "commitments": [],
    "salts": [],
    "verifications": [],
    "explainabilities": [],
}


class ThreadedHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class RPSRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS, HEAD")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def do_HEAD(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        if path in ["/", "/index.html"]:
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
        elif path.startswith("/docs/"):
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.end_headers()
        else:
            self.send_response(200)
            self.end_headers()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
            return

        if path in ["/", "/index.html"]:
            html_file = os.path.join(WEB_DIR, "index.html")
            if os.path.exists(html_file):
                with open(html_file, "rb") as f:
                    content = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)
                return
            else:
                self.send_error(404, "Index HTML not found")
                return

        elif path == "/docs/winrate_plot.png":
            plot_file = os.path.join(PROJECT_ROOT, "docs", "winrate_plot.png")
            if os.path.exists(plot_file):
                with open(plot_file, "rb") as f:
                    content = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)
                return
            else:
                self.send_error(404, "Win-rate plot not found")
                return

        elif path == "/docs/probability_plot.png":
            prob_file = os.path.join(PROJECT_ROOT, "docs", "probability_plot.png")
            if os.path.exists(prob_file):
                with open(prob_file, "rb") as f:
                    content = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)
                return
            else:
                self.send_error(404, "Probability plot not found")
                return

        elif path == "/api/state":
            self.send_json_response(self.get_current_state())
            return

        elif path == "/api/run_benchmark":
            results = run_benchmark(
                rounds=1000,
                order=2,
                seed=42,
                plot_path=os.path.join(PROJECT_ROOT, "docs", "winrate_plot.png"),
                prob_plot_path=os.path.join(PROJECT_ROOT, "docs", "probability_plot.png"),
            )

            data = []
            for r in results:
                data.append(
                    {
                        "opponent_name": r["opponent_name"],
                        "adaptive_win_rate": round(r["adaptive"]["win_rate"] * 100, 1),
                        "adaptive_loss_rate": round(r["adaptive"]["loss_rate"] * 100, 1),
                        "adaptive_tie_rate": round(r["adaptive"]["tie_rate"] * 100, 1),
                        "random_win_rate": round(r["random"]["win_rate"] * 100, 1),
                        "delta_win_rate": round(r["delta_win_rate"] * 100, 1),
                    }
                )

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
                self.send_json_response(
                    {"error": f"Invalid move. Choose from {MOVES}"}, status=400
                )
                return

            # 1. AI plans move and generates cryptographic commitment before receiving human move
            agent_move, commitment_hash = live_agent.plan_move()

            # 2. Reveal and observe human move
            obs = live_agent.observe(human_move)
            outcome = obs["outcome"]
            salt = obs["revealed_salt"]
            ver_passed = obs["verification_passed"]
            expl = obs["explainability"]

            # Update history
            game_history["human_moves"].append(human_move)
            game_history["agent_moves"].append(agent_move)
            game_history["predicted_moves"].append(
                expl.get("active_strategy", "BAYESIAN")
            )
            game_history["outcomes"].append(outcome)
            game_history["commitments"].append(commitment_hash)
            game_history["salts"].append(salt)
            game_history["verifications"].append(ver_passed)
            game_history["explainabilities"].append(expl)

            if outcome == "WIN":
                game_history["agent_wins"] += 1
            elif outcome == "LOSS":
                game_history["human_wins"] += 1
            else:
                game_history["ties"] += 1

            response_data = self.get_current_state()
            response_data["last_round"] = {
                "round": len(game_history["human_moves"]),
                "human_move": human_move,
                "human_move_name": MOVE_NAMES[human_move],
                "agent_move": agent_move,
                "agent_move_name": MOVE_NAMES[agent_move],
                "outcome": outcome,
                "commitment_hash": commitment_hash,
                "salt": salt,
                "verification_passed": ver_passed,
                "explainability": expl,
            }
            self.send_json_response(response_data)
            return

        elif path == "/api/reset":
            live_agent.reset()
            for key in game_history:
                if isinstance(game_history[key], list):
                    game_history[key].clear()
                elif isinstance(game_history[key], int):
                    game_history[key] = 0

            self.send_json_response(self.get_current_state())
            return

        elif path == "/api/csp/generate":
            length = int(payload.get("length", 150))
            max_p = float(payload.get("max_trans_prob", 0.45))
            generator = CSPSequenceGenerator(length=length, max_trans_prob=max_p)
            seq = generator.generate()
            audit = CSPSequenceGenerator.verify_constraints(seq, max_trans_prob=max_p)

            self.send_json_response(
                {
                    "status": "success",
                    "sequence_sample": seq[:60],
                    "total_length": len(seq),
                    "audit": audit,
                    "search_diagnostics": {
                        "nodes_explored": generator.nodes_explored,
                        "backtrack_count": generator.backtrack_count,
                        "constraint_checks": generator.constraint_checks,
                    },
                }
            )
            return

        elif path == "/api/privacy/consent":
            granted = bool(payload.get("consent", False))
            live_agent.consent_manager.set_consent(granted)
            self.send_json_response(
                {
                    "status": "success",
                    "consent_granted": granted,
                    "audit": live_agent.consent_manager.get_privacy_audit(),
                }
            )
            return

        elif path == "/api/privacy/purge":
            purge_info = live_agent.consent_manager.purge_all_data()
            live_agent.reset()
            for key in game_history:
                if isinstance(game_history[key], list):
                    game_history[key].clear()
                elif isinstance(game_history[key], int):
                    game_history[key] = 0

            self.send_json_response(
                {"status": "success", "purge_report": purge_info}
            )
            return

        self.send_error(404, "Endpoint not found")

    def get_current_state(self) -> Dict[str, Any]:
        total = (
            game_history["human_wins"]
            + game_history["agent_wins"]
            + game_history["ties"]
        )
        ai_win_rate = (game_history["agent_wins"] / total * 100) if total > 0 else 0.0
        human_win_rate = (
            (game_history["human_wins"] / total * 100) if total > 0 else 0.0
        )
        tie_rate = (game_history["ties"] / total * 100) if total > 0 else 0.0

        # Current Bayesian predictive distribution
        bayesian_dist = live_agent.bayesian_predictor.get_predictive_distribution()
        uncertainty = live_agent.bayesian_predictor.get_uncertainty_metrics(
            bayesian_dist
        )
        expected_utilities = BayesianPredictor.calculate_expected_utilities(
            bayesian_dist
        )

        # Current strategy detection
        detected_strategy = live_agent.rule_detector.detect_strategy()

        # Markov transition stats
        order2_transitions = {}
        for context, counter in live_agent.markov_model.n_grams.get(2, {}).items():
            context_key = f"{context[0]}->{context[1]}"
            order2_transitions[context_key] = dict(counter)

        order1_transitions = {}
        for context, counter in live_agent.markov_model.n_grams.get(1, {}).items():
            order1_transitions[context[0]] = dict(counter)

        return {
            "total_rounds": total,
            "human_wins": game_history["human_wins"],
            "agent_wins": game_history["agent_wins"],
            "ties": game_history["ties"],
            "ai_win_rate": round(ai_win_rate, 1),
            "human_win_rate": round(human_win_rate, 1),
            "tie_rate": round(tie_rate, 1),
            "bayesian_distribution": {
                k: round(v * 100, 1) for k, v in bayesian_dist.items()
            },
            "expected_utilities": expected_utilities,
            "uncertainty": uncertainty,
            "detected_strategy": detected_strategy,
            "privacy_audit": live_agent.consent_manager.get_privacy_audit(),
            "history": [
                {
                    "round": i + 1,
                    "human": game_history["human_moves"][i],
                    "agent": game_history["agent_moves"][i],
                    "outcome": game_history["outcomes"][i],
                    "commitment": game_history["commitments"][i][:12] + "...",
                    "verification": game_history["verifications"][i],
                }
                for i in range(len(game_history["human_moves"]))
            ][-15:],
            "model_stats": {
                "order": live_agent.markov_model.order,
                "unigrams": dict(live_agent.markov_model.unigram),
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
    socketserver.TCPServer.allow_reuse_address = True

    httpd = None
    active_port = port
    candidate_ports = [port, 5001, 5002, 8000, 8080]

    for p in candidate_ports:
        try:
            server_address = ("0.0.0.0", p)
            httpd = ThreadedHTTPServer(server_address, RPSRequestHandler)
            active_port = p
            break
        except OSError:
            continue

    if httpd is None:
        raise RuntimeError(f"Could not bind to any candidate port: {candidate_ports}")

    with httpd:
        url = f"http://localhost:{active_port}"
        alt_url = f"http://127.0.0.1:{active_port}"
        print("\n" + "=" * 75, flush=True)
        print("  RPS PROBABILISTIC OPPONENT MODELING WEB APPLICATION RUNNING", flush=True)
        print(f"  Localhost URL: {url}", flush=True)
        print(f"  IP Address   : {alt_url}", flush=True)
        print("=" * 75, flush=True)
        print("  Server is actively listening for incoming connections...\n", flush=True)

        if auto_open:
            try:
                webbrowser.open(url)
            except Exception:
                pass

        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server...", flush=True)


if __name__ == "__main__":
    start_server(PORT, auto_open=False)

