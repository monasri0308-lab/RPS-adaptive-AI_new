"""
Unit and Integration Tests for the Applied Bayesian Layer
=========================================================
Verifies:
- Dirichlet prior initialization and Bayesian update mechanics.
- Posterior predictive probability calculation.
- Uncertainty metrics (Dirichlet variance, Shannon entropy).
- Expected Utility calculations under payoff matrix.
- BayesianOpponent execution.
"""

import math
import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from bayesian_model import BayesianPredictor, BayesianOpponent, MOVES


def test_bayesian_dirichlet_updates():
    """Verify Dirichlet prior updates and posterior predictive distributions."""
    bp = BayesianPredictor(prior_alpha=1.0)

    # Initial uniform prior: alpha = [1, 1, 1], P(m) = 1/3 each
    probs = bp.get_unigram_distribution()
    for m in MOVES:
        assert math.isclose(probs[m], 1.0 / 3.0, abs_tol=1e-5)

    # Update with 7 Rocks
    for _ in range(7):
        bp.update("R")

    # Now alpha_R = 8, alpha_P = 1, alpha_S = 1. Total alpha = 10
    probs = bp.get_unigram_distribution()
    assert math.isclose(probs["R"], 0.8, abs_tol=1e-5)
    assert math.isclose(probs["P"], 0.1, abs_tol=1e-5)
    assert math.isclose(probs["S"], 0.1, abs_tol=1e-5)


def test_expected_utility_maximization():
    """Verify expected utilities and decision making."""
    # When P(R) = 0.8, P(P) = 0.1, P(S) = 0.1:
    # EU(Rock) = P(S) - P(P) = 0.1 - 0.1 = 0.0
    # EU(Paper) = P(Rock) - P(Scissors) = 0.8 - 0.1 = +0.7 (Countering Rock!)
    # EU(Scissors) = P(Paper) - P(Rock) = 0.1 - 0.8 = -0.7
    probs = {"R": 0.8, "P": 0.1, "S": 0.1}
    eu = BayesianPredictor.calculate_expected_utilities(probs)
    assert math.isclose(eu["R"], 0.0, abs_tol=1e-4)
    assert math.isclose(eu["P"], 0.7, abs_tol=1e-4)
    assert math.isclose(eu["S"], -0.7, abs_tol=1e-4)

    # Optimal action must be Paper
    best_action = max(eu, key=eu.get)
    assert best_action == "P"


def test_bayesian_entropy_and_uncertainty():
    """Verify Shannon entropy reaches theoretical maximum at uniform prior and drops with evidence."""
    bp = BayesianPredictor(prior_alpha=1.0)
    init_probs = bp.get_unigram_distribution()
    metrics = bp.get_uncertainty_metrics(init_probs)

    # Maximum entropy for 3 outcomes is log2(3) ~ 1.58496 bits
    assert math.isclose(metrics["entropy"], math.log2(3), abs_tol=1e-3)

    # After lots of skewed observations, entropy must strictly decrease
    for _ in range(25):
        bp.update("R")

    skewed_probs = bp.get_unigram_distribution()
    skewed_metrics = bp.get_uncertainty_metrics(skewed_probs)
    assert skewed_metrics["entropy"] < metrics["entropy"]


def test_bayesian_opponent():
    """Verify BayesianOpponent runs and counters predictable moves."""
    opp = BayesianOpponent(prior_alpha=1.0, seed=42)
    # Opponent plays against agent who repeatedly plays 'R'
    for _ in range(15):
        opp.observe("R")

    # Opponent predicts agent plays 'R', so opponent should play 'P' (counter to R)
    move = opp.get_move()
    assert move == "P"
