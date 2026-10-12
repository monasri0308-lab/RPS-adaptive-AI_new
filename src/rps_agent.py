"""
Rock-Paper-Scissors Opponent Modeling & Probabilistic AI Suite
=============================================================
A comprehensive decision-making AI suite implementing:
1. CSP Layer: Constraint-Satisfaction Adversary with backtracking search.
2. Knowledge & Reasoning Layer: Symbolic human heuristics (WSLS, Cycling, Copying).
3. Applied Bayesian Layer: Dirichlet-Multinomial Bayesian inference and Expected Utility Maximization.
4. Markov Pattern Modeling: Order-N Markov chain with hierarchical backoff.
5. Responsible AI Layer: Cryptographic fair-play commitments and transparent explainability.
"""

import argparse
import math
import os
import random
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

# Ensure local imports resolve in all execution contexts
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from bayesian_model import BayesianOpponent, BayesianPredictor
from csp_adversary import CSPAdversaryOpponent, CSPSequenceGenerator
from probabilistic_agent import ProbabilisticOpponentModeller
from reasoning_layer import (
    CopycatOpponent,
    CyclingOpponent,
    RuleBasedStrategyDetector,
    WinStayLoseShiftOpponent,
    evaluate_round,
)
from responsible_ai import (
    CryptographicCommitment,
    DataConsentManager,
    format_explainability_card,
)

# Valid Moves & Counter-Move Rules
MOVES: List[str] = ["R", "P", "S"]
MOVE_NAMES: Dict[str, str] = {"R": "Rock", "P": "Paper", "S": "Scissors"}
COUNTER_MOVES: Dict[str, str] = {"R": "P", "P": "S", "S": "R"}


class OpponentModel:
    """Frequency and pattern model using an Order-N Markov Chain with Hierarchical Backoff."""

    def __init__(self, order: int = 2, seed: Optional[int] = None):
        self.order = max(1, order)
        self.rng = random.Random(seed)
        self.history: List[str] = []
        self.n_grams: Dict[int, Dict[Tuple[str, ...], Counter]] = {
            k: defaultdict(Counter) for k in range(1, self.order + 1)
        }
        self.unigram: Counter = Counter()

    def update(self, opponent_move: str) -> None:
        """Record an observed opponent move and update n-gram count tables."""
        if opponent_move not in MOVES:
            raise ValueError(f"Invalid move '{opponent_move}'. Must be one of {MOVES}")

        for k in range(1, self.order + 1):
            if len(self.history) >= k:
                context = tuple(self.history[-k:])
                self.n_grams[k][context][opponent_move] += 1

        self.unigram[opponent_move] += 1
        self.history.append(opponent_move)

    def predict_next_move(self) -> str:
        """Predict the opponent's next move using hierarchical backoff strategy."""
        for k in range(self.order, 0, -1):
            if len(self.history) >= k:
                context = tuple(self.history[-k:])
                if context in self.n_grams[k] and self.n_grams[k][context]:
                    counts = self.n_grams[k][context]
                    max_count = max(counts.values())
                    candidates = [m for m, c in counts.items() if c == max_count]
                    return self.rng.choice(candidates)

        if self.unigram:
            max_count = max(self.unigram.values())
            candidates = [m for m, c in self.unigram.items() if c == max_count]
            return self.rng.choice(candidates)

        return self.rng.choice(MOVES)

    def get_counter_move(self) -> str:
        """Predict opponent's next move and return the winning counter-move."""
        predicted_move = self.predict_next_move()
        return COUNTER_MOVES[predicted_move]

    def reset(self) -> None:
        """Reset history and frequency counts."""
        self.history.clear()
        self.n_grams = {k: defaultdict(Counter) for k in range(1, self.order + 1)}
        self.unigram.clear()


# =====================================================================
# AGENT IMPLEMENTATIONS
# =====================================================================


class AdaptiveAgent:
    """Adaptive RPS Agent combining Markov pattern modeling, Bayesian updating, and symbolic reasoning."""

    def __init__(self, order: int = 2, seed: Optional[int] = None):
        self.order = order
        self.model = OpponentModel(order=order, seed=seed)
        self.probabilistic_modeller = ProbabilisticOpponentModeller(
            order=order, seed=seed
        )

    def get_move(self) -> str:
        return self.probabilistic_modeller.get_move()

    def observe(self, opponent_move: str) -> None:
        self.model.update(opponent_move)
        self.probabilistic_modeller.observe(opponent_move)

    def reset(self) -> None:
        self.model.reset()
        self.probabilistic_modeller.reset()


class RandomAgent:
    """Baseline agent playing uniformly at random."""

    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)

    def get_move(self) -> str:
        return self.rng.choice(MOVES)

    def observe(self, opponent_move: str) -> None:
        pass

    def reset(self) -> None:
        pass


# =====================================================================
# BASELINE OPPONENTS
# =====================================================================


class PureRandomOpponent:
    """Opponent playing pure uniform random moves (unexploitable Nash equilibrium)."""

    def __init__(self, seed: Optional[int] = None):
        self.name = "Pure Random"
        self.rng = random.Random(seed)

    def get_move(self) -> str:
        return self.rng.choice(MOVES)

    def observe(self, agent_move: str) -> None:
        pass

    def reset(self) -> None:
        pass


class RotatingOpponent:
    """Opponent playing a fixed rotating cycle (e.g. R -> P -> S -> R...)."""

    def __init__(self, cycle: Optional[List[str]] = None):
        self.name = "Rotating (R->P->S)"
        self.cycle = cycle or ["R", "P", "S"]
        self.index = 0

    def get_move(self) -> str:
        move = self.cycle[self.index % len(self.cycle)]
        self.index += 1
        return move

    def observe(self, agent_move: str) -> None:
        pass

    def reset(self) -> None:
        self.index = 0


class BiasedStreakOpponent:
    """Human-like opponent with marginal move bias and streak persistence."""

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        streak_prob: float = 0.60,
        seed: Optional[int] = None,
    ):
        self.name = "Human-Like Biased"
        self.weights = weights or {"R": 0.50, "P": 0.30, "S": 0.20}
        self.streak_prob = streak_prob
        self.rng = random.Random(seed)
        self.last_move: Optional[str] = None

        total = sum(self.weights.values())
        self.moves = list(self.weights.keys())
        self.probs = [self.weights[m] / total for m in self.moves]

    def get_move(self) -> str:
        if self.last_move is not None and self.rng.random() < self.streak_prob:
            move = self.last_move
        else:
            move = self.rng.choices(self.moves, weights=self.probs, k=1)[0]
        self.last_move = move
        return move

    def observe(self, agent_move: str) -> None:
        pass

    def reset(self) -> None:
        self.last_move = None


# =====================================================================
# SIMULATION & BENCHMARKING ENGINE
# =====================================================================


def run_matchup(agent, opponent, rounds: int) -> Dict[str, Any]:
    """Simulate a match of N rounds between an agent and an opponent."""
    agent.reset()
    opponent.reset()

    agent_moves = []
    opponent_moves = []
    outcomes = []
    win_flags = []
    probability_series = []

    for _ in range(rounds):
        a_move = agent.get_move()
        o_move = opponent.get_move()

        outcome = evaluate_round(a_move, o_move)
        agent.observe(o_move)
        opponent.observe(a_move)

        agent_moves.append(a_move)
        opponent_moves.append(o_move)
        outcomes.append(outcome)
        win_flags.append(1.0 if outcome == "WIN" else 0.0)

        # Track probability predictions if available
        if hasattr(agent, "probabilistic_modeller") and agent.probabilistic_modeller.probability_history:
            probability_series.append(dict(agent.probabilistic_modeller.probability_history[-1]))
        elif hasattr(agent, "probability_history") and agent.probability_history:
            probability_series.append(dict(agent.probability_history[-1]))

    win_flags_arr = np.array(win_flags)
    wins = outcomes.count("WIN")
    losses = outcomes.count("LOSS")
    ties = outcomes.count("TIE")

    return {
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "win_rate": wins / rounds,
        "loss_rate": losses / rounds,
        "tie_rate": ties / rounds,
        "win_flags": win_flags_arr,
        "outcomes": outcomes,
        "probabilities": probability_series,
    }


def calculate_rolling_win_rate(win_flags: np.ndarray, window: int = 50) -> np.ndarray:
    """Calculate rolling average win-rate over time."""
    n = len(win_flags)
    rolling = np.zeros(n)
    for i in range(n):
        start_idx = max(0, i - window + 1)
        rolling[i] = np.mean(win_flags[start_idx : i + 1])
    return rolling


def plot_simulation_results(
    results: List[Dict],
    rounds: int,
    output_path: str = "docs/winrate_plot.png",
    window: int = 50,
) -> None:
    """Generate and save publication-quality rolling win-rate comparison plots across all opponents."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    num_opponents = len(results)
    cols = 4 if num_opponents >= 4 else num_opponents
    rows = math.ceil(num_opponents / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, 4.2 * rows), sharey=True)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes_flat = axes.flatten()

    plt.style.use(
        "seaborn-v0_8-whitegrid"
        if "seaborn-v0_8-whitegrid" in plt.style.available
        else "default"
    )

    x_axis = np.arange(1, rounds + 1)

    for i, res in enumerate(results):
        ax = axes_flat[i]
        opp_name = res["opponent_name"]
        adaptive_rolling = calculate_rolling_win_rate(
            res["adaptive"]["win_flags"], window=window
        )
        random_rolling = calculate_rolling_win_rate(
            res["random"]["win_flags"], window=window
        )

        ax.plot(
            x_axis,
            adaptive_rolling * 100,
            label="Probabilistic Agent",
            color="#0284c7",
            linewidth=2.0,
        )
        ax.plot(
            x_axis,
            random_rolling * 100,
            label="Random Baseline",
            color="#ef4444",
            linestyle="--",
            linewidth=1.6,
            alpha=0.8,
        )
        ax.axhline(
            y=33.33,
            color="#64748b",
            linestyle=":",
            linewidth=1.3,
            label="Nash Eq (33.3%)",
        )

        ax.set_title(f"vs. {opp_name}", fontsize=11, fontweight="bold", pad=8)
        ax.set_xlabel("Rounds", fontsize=9.5)
        ax.set_ylim(-5, 105)
        ax.grid(True, linestyle="--", alpha=0.4)

        delta_str = f"Δ: {res['delta_win_rate'] * 100:+.1f}%"
        ax.text(
            0.04,
            0.92,
            f"Agent: {res['adaptive']['win_rate'] * 100:.1f}%\nBase: {res['random']['win_rate'] * 100:.1f}%\n{delta_str}",
            transform=ax.transAxes,
            fontsize=8.5,
            verticalalignment="top",
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="white",
                edgecolor="#cbd5e1",
                alpha=0.9,
            ),
        )

    # Turn off unused subplots if any
    for j in range(num_opponents, len(axes_flat)):
        axes_flat[j].axis("off")

    axes_flat[0].set_ylabel(f"Rolling Win Rate (%) [W={window}]", fontsize=10)
    axes_flat[0].legend(loc="lower right", frameon=True, fontsize=8)

    plt.suptitle(
        f"Probabilistic Opponent Modeller Benchmark: Win-Rate vs. 8 Opponent Classes ({rounds} Rounds)",
        fontsize=14,
        fontweight="bold",
        y=0.99,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\n[Plot Saved] Rolling win-rate curves saved to: {output_path}")


def plot_probability_distribution(
    key_matchups: Dict[str, List[Dict[str, float]]],
    output_path: str = "docs/probability_plot.png",
    rounds_to_display: int = 150,
) -> None:
    """Generate and save publication-quality probability distribution evolution plots."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    matchup_names = list(key_matchups.keys())
    n = len(matchup_names)

    fig, axes = plt.subplots(n, 1, figsize=(10, 3.2 * n), sharex=True)
    if n == 1:
        axes = [axes]

    colors = {"R": "#ef4444", "P": "#3b82f6", "S": "#10b981"}

    for ax, name in zip(axes, matchup_names):
        probs_list = key_matchups[name][:rounds_to_display]
        r_vals = [p.get("R", 0.33) for p in probs_list]
        p_vals = [p.get("P", 0.33) for p in probs_list]
        s_vals = [p.get("S", 0.33) for p in probs_list]
        x = np.arange(1, len(probs_list) + 1)

        ax.plot(x, r_vals, label="P(Rock)", color=colors["R"], linewidth=1.8)
        ax.plot(x, p_vals, label="P(Paper)", color=colors["P"], linewidth=1.8)
        ax.plot(x, s_vals, label="P(Scissors)", color=colors["S"], linewidth=1.8)
        ax.axhline(y=0.3333, color="#94a3b8", linestyle=":", linewidth=1.2, label="Uniform Prior (1/3)")

        ax.set_title(f"Predicted Probability Distribution Evolution: vs. {name}", fontsize=11, fontweight="bold")
        ax.set_ylabel("Probability", fontsize=10)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="upper right", frameon=True, fontsize=8.5, ncol=4)

    axes[-1].set_xlabel("Rounds (Initial Convergence Phase)", fontsize=10)
    plt.suptitle(
        "Bayesian Posterior Predictive Probabilities P(M_t) Under Uncertainty",
        fontsize=13,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Plot Saved] Probability evolution visualization saved to: {output_path}")


def run_benchmark(
    rounds: int = 1000,
    order: int = 2,
    seed: Optional[int] = 42,
    plot_path: str = "docs/winrate_plot.png",
    prob_plot_path: str = "docs/probability_plot.png",
) -> List[Dict]:
    """Execute complete 1000-round benchmark suite across Random, Rule-Based, Bayesian, and CSP opponents."""
    opponents = [
        PureRandomOpponent(seed=seed),
        RotatingOpponent(cycle=["R", "P", "S"]),
        BiasedStreakOpponent(weights={"R": 0.50, "P": 0.30, "S": 0.20}, streak_prob=0.60, seed=seed),
        WinStayLoseShiftOpponent(seed=seed),
        CyclingOpponent(direction="forward", seed=seed),
        CopycatOpponent(mode="direct", seed=seed),
        BayesianOpponent(prior_alpha=1.0, seed=seed),
        CSPAdversaryOpponent(length=rounds, max_trans_prob=0.45, seed=seed),
    ]

    benchmark_results = []
    key_prob_matchups = {}

    for opponent in opponents:
        adaptive_agent = AdaptiveAgent(order=order, seed=seed)
        random_agent = RandomAgent(seed=seed)

        adaptive_res = run_matchup(adaptive_agent, opponent, rounds=rounds)
        random_res = run_matchup(random_agent, opponent, rounds=rounds)

        delta = adaptive_res["win_rate"] - random_res["win_rate"]

        res_dict = {
            "opponent_name": opponent.name,
            "adaptive": adaptive_res,
            "random": random_res,
            "delta_win_rate": delta,
        }
        benchmark_results.append(res_dict)

        # Save probability trajectories for representative opponents
        if opponent.name in [
            "Human-Like Biased",
            "Win-Stay / Lose-Shift",
            "CSP Constraint Adversary",
        ]:
            key_prob_matchups[opponent.name] = adaptive_res["probabilities"]

    # Print Formatted Results Table
    header_fmt = "| {:<28} | {:>14} | {:>14} | {:>14} | {:>14} | {:>14} |"
    row_fmt = "| {:<28} | {:>13.1f}% | {:>13.1f}% | {:>13.1f}% | {:>13.1f}% | {:>+13.1f}% |"
    separator = "+------------------------------+----------------+----------------+----------------+----------------+----------------+"

    print("\n" + "=" * 110)
    print(f"  PROBABILISTIC OPPONENT MODELLER: 1000-ROUND BENCHMARK (Order: {order}, Seed: {seed})")
    print("=" * 110)
    print(separator)
    print(
        header_fmt.format(
            "Opponent Strategy",
            "Adaptive Win%",
            "Adaptive Loss%",
            "Adaptive Tie%",
            "Random Base Win%",
            "Delta Win%",
        )
    )
    print(separator)

    for r in benchmark_results:
        print(
            row_fmt.format(
                r["opponent_name"],
                r["adaptive"]["win_rate"] * 100,
                r["adaptive"]["loss_rate"] * 100,
                r["adaptive"]["tie_rate"] * 100,
                r["random"]["win_rate"] * 100,
                r["delta_win_rate"] * 100,
            )
        )
    print(separator)
    print(
        " Note: Delta Win% = (Adaptive Agent Win Rate) - (Random Baseline Agent Win Rate)\n"
    )

    # Generate Both Plots
    plot_simulation_results(benchmark_results, rounds=rounds, output_path=plot_path)
    if key_prob_matchups:
        plot_probability_distribution(key_prob_matchups, output_path=prob_plot_path)

    return benchmark_results


def play_interactive(order: int = 2) -> None:
    """Run an interactive live console session with fair-play cryptographic verification and explainability."""
    agent = ProbabilisticOpponentModeller(order=order)
    human_wins = 0
    agent_wins = 0
    ties = 0
    round_num = 1

    print("\n" + "=" * 75)
    print("  LIVE ARENA: HUMAN vs. PROBABILISTIC OPPONENT MODELLER")
    print(f"  (Bayesian Dirichlet Model + Symbolic Reasoning + SHA-256 Fair Play)")
    print("=" * 75)
    print("  Controls: [R]ock, [P]aper, [S]cissors, or [Q]uit")
    print("-" * 75)

    try:
        while True:
            # 1. Agent plans move and produces cryptographic commitment hash BEFORE human moves
            agent_move, commitment_hash = agent.plan_move()
            print(f"\n[Round {round_num}]")
            print(f"  [Fair-Play] AI SHA-256 Commitment: {commitment_hash[:16]}... (Move locked!)")

            user_input = input("  Your move (R/P/S or Q): ").strip().upper()
            if user_input in ["Q", "QUIT", "EXIT"]:
                break
            if user_input not in MOVES:
                print("  [!] Invalid input. Choose 'R', 'P', 'S', or 'Q' to quit.")
                continue

            # 2. Reveal commitment and update model
            obs = agent.observe(user_input)
            outcome = obs["outcome"]
            salt = obs["revealed_salt"]
            expl = obs["explainability"]

            if outcome == "WIN":
                agent_wins += 1
                result_str = "\033[91mAI WINS!\033[0m"
            elif outcome == "LOSS":
                human_wins += 1
                result_str = "\033[92mYOU WIN!\033[0m"
            else:
                ties += 1
                result_str = "\033[93mTIE!\033[0m"

            total = human_wins + agent_wins + ties
            probs = expl["predicted_probabilities"]

            print(f"  -> You played: {MOVE_NAMES[user_input]} ({user_input})")
            print(f"  -> AI revealed: {MOVE_NAMES[agent_move]} ({agent_move}) | Salt: {salt[:8]}...")
            print(f"  -> Verification: {'✓ PASSED (Zero-peeking confirmed)' if obs['verification_passed'] else '✗ FAILED'}")
            print(f"  -> Predicted Opponent Distribution: R:{probs['R']:.1%} | P:{probs['P']:.1%} | S:{probs['S']:.1%}")
            print(f"  -> AI Explainability: {expl['explanation']}")
            print(f"  -> Outcome: {result_str}")
            print(f"  [Scoreboard] You: {human_wins} | AI: {agent_wins} | Ties: {ties} (AI Win-Rate: {(agent_wins/total)*100:.1f}%)")

            round_num += 1

    except (KeyboardInterrupt, EOFError):
        pass

    print("\n" + "=" * 75)
    print("  MATCH SUMMARY")
    print("=" * 75)
    total_games = human_wins + agent_wins + ties
    if total_games > 0:
        print(f"  Total Rounds Played : {total_games}")
        print(f"  Your Wins           : {human_wins} ({(human_wins/total_games)*100:.1f}%)")
        print(f"  AI Wins             : {agent_wins} ({(agent_wins/total_games)*100:.1f}%)")
        print(f"  Ties                : {ties} ({(ties/total_games)*100:.1f}%)")
    print("=" * 75 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Rock-Paper-Scissors Probabilistic Opponent Modeling AI"
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=1000,
        help="Number of simulation rounds per opponent (default: 1000)",
    )
    parser.add_argument(
        "--order",
        type=int,
        default=2,
        help="Markov Chain order for pattern modeling (default: 2)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--plot-path",
        type=str,
        default="docs/winrate_plot.png",
        help="Destination path for rolling win-rate plot (default: docs/winrate_plot.png)",
    )
    parser.add_argument(
        "--prob-plot-path",
        type=str,
        default="docs/probability_plot.png",
        help="Destination path for probability evolution plot (default: docs/probability_plot.png)",
    )
    parser.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="Play live interactively in console against the adaptive agent",
    )

    args = parser.parse_args()

    if args.interactive:
        play_interactive(order=args.order)
    else:
        run_benchmark(
            rounds=args.rounds,
            order=args.order,
            seed=args.seed,
            plot_path=args.plot_path,
            prob_plot_path=args.prob_plot_path,
        )


if __name__ == "__main__":
    main()
