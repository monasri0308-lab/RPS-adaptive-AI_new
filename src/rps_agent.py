"""
Rock-Paper-Scissors Opponent Modeling Agent
============================================
An adaptive Rock-Paper-Scissors agent implementing an Order-N Markov chain
with hierarchical backoff to predict opponent actions and play optimal counter-moves.

Baseline opponents:
- Pure Random
- Fixed / Rotating Strategy (e.g., R -> P -> S cycle)
- Biased Streak Random (Human-like cognitive bias)
"""

import argparse
import os
import random
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

# Valid Moves & Counter-Move Rules
MOVES: List[str] = ["R", "P", "S"]
MOVE_NAMES: Dict[str, str] = {"R": "Rock", "P": "Paper", "S": "Scissors"}
COUNTER_MOVES: Dict[str, str] = {"R": "P", "P": "S", "S": "R"}


def evaluate_round(agent_move: str, opponent_move: str) -> str:
    """Evaluate outcome from the perspective of the agent.

    Returns:
        'WIN', 'LOSS', or 'TIE'
    """
    if agent_move == opponent_move:
        return "TIE"
    elif COUNTER_MOVES[opponent_move] == agent_move:
        return "WIN"
    else:
        return "LOSS"


class OpponentModel:
    """Frequency and pattern model using an Order-N Markov Chain with Hierarchical Backoff.

    Hierarchy:
        Order-N Markov Context -> Order-(N-1) -> ... -> Order-1 -> Unigram Frequency -> Uniform Random
    """

    def __init__(self, order: int = 2, seed: Optional[int] = None):
        self.order = max(1, order)
        self.rng = random.Random(seed)
        self.history: List[str] = []
        # n_grams[k] stores mapping: tuple of k moves -> Counter of next moves
        self.n_grams: Dict[int, Dict[Tuple[str, ...], Counter]] = {
            k: defaultdict(Counter) for k in range(1, self.order + 1)
        }
        self.unigram: Counter = Counter()

    def update(self, opponent_move: str) -> None:
        """Record an observed opponent move and update n-gram count tables."""
        if opponent_move not in MOVES:
            raise ValueError(f"Invalid move '{opponent_move}'. Must be one of {MOVES}")

        # Update n-gram transitions for all k in [1, order]
        for k in range(1, self.order + 1):
            if len(self.history) >= k:
                context = tuple(self.history[-k:])
                self.n_grams[k][context][opponent_move] += 1

        # Update unigram frequency
        self.unigram[opponent_move] += 1
        self.history.append(opponent_move)

    def predict_next_move(self) -> str:
        """Predict the opponent's next move using backoff strategy."""
        # 1. Try Order-k Markov predictions from Order-N down to Order-1
        for k in range(self.order, 0, -1):
            if len(self.history) >= k:
                context = tuple(self.history[-k:])
                if context in self.n_grams[k] and self.n_grams[k][context]:
                    counts = self.n_grams[k][context]
                    max_count = max(counts.values())
                    candidates = [m for m, c in counts.items() if c == max_count]
                    return self.rng.choice(candidates)

        # 2. Fall back to Unigram frequency
        if self.unigram:
            max_count = max(self.unigram.values())
            candidates = [m for m, c in self.unigram.items() if c == max_count]
            return self.rng.choice(candidates)

        # 3. Fall back to Uniform Random when no history is available
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
    """Adaptive RPS Agent powered by the Markov OpponentModel."""

    def __init__(self, order: int = 2, seed: Optional[int] = None):
        self.model = OpponentModel(order=order, seed=seed)

    def get_move(self) -> str:
        return self.model.get_counter_move()

    def observe(self, opponent_move: str) -> None:
        self.model.update(opponent_move)

    def reset(self) -> None:
        self.model.reset()


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
# BASELINE OPPONENT IMPLEMENTATIONS
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
    """Human-like opponent with marginal move bias (50% R, 30% P, 20% S) and streak bias.

    Simulates cognitive bias where humans non-randomly repeat their previous action.
    """

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

        # Normalize weights
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


def run_matchup(agent, opponent, rounds: int) -> Dict[str, np.ndarray]:
    """Simulate a match of N rounds between an agent and an opponent."""
    agent.reset()
    opponent.reset()

    agent_moves = []
    opponent_moves = []
    outcomes = []
    win_flags = []

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
    """Generate and save publication-quality rolling win-rate comparison plots."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    num_opponents = len(results)
    fig, axes = plt.subplots(1, num_opponents, figsize=(6 * num_opponents, 5), sharey=True)

    if num_opponents == 1:
        axes = [axes]

    # Clean modern styling
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    for ax, res in zip(axes, results):
        opp_name = res["opponent_name"]
        adaptive_rolling = calculate_rolling_win_rate(res["adaptive"]["win_flags"], window=window)
        random_rolling = calculate_rolling_win_rate(res["random"]["win_flags"], window=window)
        x_axis = np.arange(1, rounds + 1)

        # Plot curves
        ax.plot(
            x_axis,
            adaptive_rolling * 100,
            label=f"Adaptive Agent (Order-2)",
            color="#1f77b4",
            linewidth=2.2,
        )
        ax.plot(
            x_axis,
            random_rolling * 100,
            label="Random Baseline Agent",
            color="#d62728",
            linestyle="--",
            linewidth=1.8,
            alpha=0.8,
        )

        # Reference baseline at 33.3%
        ax.axhline(
            y=33.33,
            color="#7f7f7f",
            linestyle=":",
            linewidth=1.5,
            label="Nash Expected (33.3%)",
        )

        ax.set_title(f"vs. {opp_name}", fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("Rounds", fontsize=11)
        ax.set_ylim(-5, 105)
        ax.grid(True, linestyle="--", alpha=0.5)

        # Subtitle metrics box
        delta_str = f"Δ: {res['delta_win_rate'] * 100:+.1f}%"
        ax.text(
            0.05,
            0.92,
            f"Adaptive: {res['adaptive']['win_rate'] * 100:.1f}%\nBaseline: {res['random']['win_rate'] * 100:.1f}%\n{delta_str}",
            transform=ax.transAxes,
            fontsize=9.5,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#cccccc", alpha=0.9),
        )

    axes[0].set_ylabel(f"Rolling Win Rate (%) [Window={window}]", fontsize=11)
    axes[0].legend(loc="lower right", frameon=True, fontsize=9.5)

    plt.suptitle(
        f"Rock-Paper-Scissors Opponent Modeling Benchmark ({rounds} Rounds)",
        fontsize=15,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\n[Plot Saved] Visualization successfully exported to: {output_path}")


def run_benchmark(
    rounds: int = 1000,
    order: int = 2,
    seed: Optional[int] = 42,
    plot_path: str = "docs/winrate_plot.png",
) -> List[Dict]:
    """Execute complete benchmark suite across all baseline opponents and print results table."""
    opponents = [
        PureRandomOpponent(seed=seed),
        RotatingOpponent(cycle=["R", "P", "S"]),
        BiasedStreakOpponent(weights={"R": 0.50, "P": 0.30, "S": 0.20}, streak_prob=0.60, seed=seed),
    ]

    benchmark_results = []

    for opponent in opponents:
        adaptive_agent = AdaptiveAgent(order=order, seed=seed)
        random_agent = RandomAgent(seed=seed)

        adaptive_res = run_matchup(adaptive_agent, opponent, rounds=rounds)
        random_res = run_matchup(random_agent, opponent, rounds=rounds)

        delta = adaptive_res["win_rate"] - random_res["win_rate"]

        benchmark_results.append(
            {
                "opponent_name": opponent.name,
                "adaptive": adaptive_res,
                "random": random_res,
                "delta_win_rate": delta,
            }
        )

    # Print Formatted Results Table
    header_fmt = "| {:<22} | {:>14} | {:>14} | {:>14} | {:>14} | {:>14} |"
    row_fmt = "| {:<22} | {:>13.1f}% | {:>13.1f}% | {:>13.1f}% | {:>13.1f}% | {:>+13.1f}% |"
    separator = "+------------------------+----------------+----------------+----------------+----------------+----------------+"

    print("\n" + "=" * 100)
    print(f"  RPS OPPONENT MODELING BENCHMARK RESULTS (Rounds: {rounds}, Markov Order: {order}, Seed: {seed})")
    print("=" * 100)
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

    # Generate Plot
    plot_simulation_results(benchmark_results, rounds=rounds, output_path=plot_path)

    return benchmark_results


def play_interactive(order: int = 2) -> None:
    """Run an interactive live console session allowing a human to play against the adaptive agent."""
    agent = AdaptiveAgent(order=order)
    human_wins = 0
    agent_wins = 0
    ties = 0
    round_num = 1

    print("\n" + "=" * 65)
    print("  LIVE INTERACTIVE MATCH: HUMAN vs. ADAPTIVE MARKOV AGENT")
    print(f"  (Markov Order: {order} with Hierarchical Backoff)")
    print("=" * 65)
    print("  Controls: [R]ock, [P]aper, [S]cissors, or [Q]uit")
    print("-" * 65)

    try:
        while True:
            user_input = input(f"\n[Round {round_num}] Your move (R/P/S or Q): ").strip().upper()
            if user_input in ["Q", "QUIT", "EXIT"]:
                break
            if user_input not in MOVES:
                print("  [!] Invalid input. Please enter 'R', 'P', 'S', or 'Q' to quit.")
                continue

            # Agent predicts and picks counter-move
            predicted_opponent_move = agent.model.predict_next_move()
            agent_move = COUNTER_MOVES[predicted_opponent_move]

            # Evaluate outcome
            outcome = evaluate_round(agent_move, user_input)
            agent.observe(user_input)

            # Record stats
            if outcome == "WIN":
                agent_wins += 1
                result_str = "\033[91mAI WINS!\033[0m"
            elif outcome == "LOSS":
                human_wins += 1
                result_str = "\033[92mYOU WIN!\033[0m"
            else:
                ties += 1
                result_str = "\033[93mTIE!\033[0m"

            total_games = human_wins + agent_wins + ties
            ai_winrate = (agent_wins / total_games) * 100

            print(f"  -> You played: {MOVE_NAMES[user_input]} ({user_input})")
            print(f"  -> AI predicted: {MOVE_NAMES[predicted_opponent_move]} ({predicted_opponent_move}) | AI played: {MOVE_NAMES[agent_move]} ({agent_move})")
            print(f"  -> Result: {result_str}")
            print(f"  [Score] You: {human_wins} | AI: {agent_wins} | Ties: {ties} (AI Win-Rate: {ai_winrate:.1f}%)")

            round_num += 1

    except (KeyboardInterrupt, EOFError):
        pass

    print("\n" + "=" * 65)
    print("  MATCH SUMMARY")
    print("=" * 65)
    total_games = human_wins + agent_wins + ties
    if total_games > 0:
        print(f"  Total Rounds Played : {total_games}")
        print(f"  Your Wins           : {human_wins} ({(human_wins/total_games)*100:.1f}%)")
        print(f"  AI Wins             : {agent_wins} ({(agent_wins/total_games)*100:.1f}%)")
        print(f"  Ties                : {ties} ({(ties/total_games)*100:.1f}%)")
    else:
        print("  No rounds played.")
    print("=" * 65 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Rock-Paper-Scissors Adaptive Opponent Modeling Simulation"
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
        help="Destination path for the generated plot (default: docs/winrate_plot.png)",
    )
    parser.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="Play live interactively in the console against the adaptive agent",
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
        )


if __name__ == "__main__":
    main()

