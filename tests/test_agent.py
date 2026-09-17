"""
Unit and Integration Tests for RPS Opponent Modeling Agent
===========================================================
Tests verify:
- Counter-move logic and evaluation rules
- Markov chain update and backoff hierarchy on known sequences
- Win-rate against fixed/rotating strategy exceeds 60% (fully predictable)
- Win-rate against pure random opponent is statistically ~33%
"""

import os
import sys
import pytest

# Ensure src/ is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from rps_agent import (
    MOVES,
    COUNTER_MOVES,
    evaluate_round,
    OpponentModel,
    AdaptiveAgent,
    RandomAgent,
    PureRandomOpponent,
    RotatingOpponent,
    BiasedStreakOpponent,
    run_matchup,
)


def test_counter_move_logic():
    """Verify that rock counters scissors, paper counters rock, scissors counters paper."""
    assert COUNTER_MOVES["R"] == "P", "Paper must counter Rock"
    assert COUNTER_MOVES["P"] == "S", "Scissors must counter Paper"
    assert COUNTER_MOVES["S"] == "R", "Rock must counter Scissors"

    # Test round evaluations
    assert evaluate_round("P", "R") == "WIN"
    assert evaluate_round("S", "P") == "WIN"
    assert evaluate_round("R", "S") == "WIN"

    assert evaluate_round("R", "P") == "LOSS"
    assert evaluate_round("P", "S") == "LOSS"
    assert evaluate_round("S", "R") == "LOSS"

    assert evaluate_round("R", "R") == "TIE"
    assert evaluate_round("P", "P") == "TIE"
    assert evaluate_round("S", "S") == "TIE"


def test_markov_prediction_updates_order2():
    """Verify Order-2 Markov predictions update accurately on known deterministic sequences."""
    model = OpponentModel(order=2, seed=42)

    # Empty model falls back to random/unigram
    pred = model.predict_next_move()
    assert pred in MOVES

    # Feed sequence with pattern: R, P -> R, and R, P -> R
    # Context ("R", "P") always followed by "R"
    sequence = ["R", "P", "R", "P", "R", "P", "R"]
    for m in sequence:
        model.update(m)

    # Current last 2 moves in history are ["P", "R"]
    # The history contains ("P", "R") -> "P"
    assert model.predict_next_move() == "P"
    assert model.get_counter_move() == "S"  # Counter to P is S

    # Feed "P" so history ends in ("R", "P")
    model.update("P")
    # Context ("R", "P") was always followed by "R"
    assert model.predict_next_move() == "R"
    assert model.get_counter_move() == "P"  # Counter to R is P


def test_markov_backoff_hierarchy():
    """Verify backoff triggers from Order-2 down to Order-1, unigram, and random."""
    model = OpponentModel(order=2, seed=42)

    # Move 1: single move 'R'
    model.update("R")
    # Order-2 is empty (needs 2 moves for context). Order-1 context ('R',) has 0 transitions.
    # Unigram has {'R': 1}, so unigram predicts 'R'.
    assert model.predict_next_move() == "R"

    # Move 2: 'S'
    model.update("S")
    # Order-1 context ('S',) has no transitions yet.
    # Unigram has {'R': 1, 'S': 1}.
    pred = model.predict_next_move()
    assert pred in ["R", "S"]


def test_invalid_move_handling():
    """Verify OpponentModel raises ValueError on invalid move input."""
    model = OpponentModel(order=2)
    with pytest.raises(ValueError):
        model.update("X")


def test_winrate_against_fixed_strategy():
    """Verify win-rate against predictable rotating/fixed opponent exceeds 60%."""
    rounds = 500
    agent = AdaptiveAgent(order=2, seed=42)
    opponent = RotatingOpponent(cycle=["R", "P", "S"])

    results = run_matchup(agent, opponent, rounds=rounds)
    win_rate = results["win_rate"]

    # Rotating opponent is 100% predictable after learning initial 2-move transition
    assert win_rate > 0.60, f"Expected win-rate > 60%, got {win_rate * 100:.2f}%"
    assert win_rate >= 0.95, f"Expected near-perfect win-rate (>=95%), got {win_rate * 100:.2f}%"


def test_winrate_against_human_like_biased():
    """Verify adaptive agent significantly outperforms random baseline on biased human-like opponent."""
    rounds = 1000
    seed = 42
    agent = AdaptiveAgent(order=2, seed=seed)
    random_agent = RandomAgent(seed=seed)
    opponent = BiasedStreakOpponent(weights={"R": 0.50, "P": 0.30, "S": 0.20}, streak_prob=0.60, seed=seed)

    adaptive_res = run_matchup(agent, opponent, rounds=rounds)
    random_res = run_matchup(random_agent, opponent, rounds=rounds)

    assert adaptive_res["win_rate"] > 0.60, f"Expected win-rate > 60%, got {adaptive_res['win_rate'] * 100:.2f}%"
    assert adaptive_res["win_rate"] > random_res["win_rate"] + 0.25, (
        f"Expected at least +25% delta over random baseline, got {(adaptive_res['win_rate'] - random_res['win_rate']) * 100:.2f}%"
    )


def test_winrate_against_random_opponent():
    """Sanity check: Win-rate against pure random opponent should statistically approximate 33.3%."""
    rounds = 1500
    agent = AdaptiveAgent(order=2, seed=42)
    opponent = PureRandomOpponent(seed=42)

    results = run_matchup(agent, opponent, rounds=rounds)
    win_rate = results["win_rate"]

    # Against pure random (Nash equilibrium), expected win rate is 33.33%.
    # With 1500 rounds, 99% confidence interval is approximately [29.5%, 37.1%].
    assert 0.28 <= win_rate <= 0.39, (
        f"Expected win rate ~33.3% (within [28%, 39%]), got {win_rate * 100:.2f}%"
    )
