"""
Unit and Integration Tests for the Knowledge & Reasoning Layer
==============================================================
Verifies:
- Rule-based opponents (WSLS, Cycling, Copycat).
- RuleBasedStrategyDetector hypothesis evaluation, confidence calculation, and explanations.
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from reasoning_layer import (
    MOVES,
    COUNTER_MOVES,
    CYCLE_FORWARD,
    CYCLE_BACKWARD,
    WinStayLoseShiftOpponent,
    CyclingOpponent,
    CopycatOpponent,
    RuleBasedStrategyDetector,
    evaluate_round,
)


def test_wsls_opponent_behavior():
    """Verify Win-Stay / Lose-Shift logic."""
    opp = WinStayLoseShiftOpponent(shift_mode="counter_beater", seed=42)

    # Move 1: Opponent plays something
    m1 = opp.get_move()
    assert m1 in MOVES

    # Suppose agent plays move that loses to m1 -> Opponent wins!
    losing_agent_move = "S" if m1 == "R" else ("R" if m1 == "P" else "P")
    opp.observe(losing_agent_move)

    # Opponent won -> Win-Stay: must repeat m1
    m2 = opp.get_move()
    assert m2 == m1, f"Expected Win-Stay {m1}, got {m2}"

    # Now suppose agent beats opponent
    winning_agent_move = COUNTER_MOVES[m2]
    opp.observe(winning_agent_move)

    # Opponent lost -> Lose-Shift: must shift to counter winning_agent_move
    m3 = opp.get_move()
    expected_shift = COUNTER_MOVES[winning_agent_move]
    assert m3 == expected_shift, f"Expected Lose-Shift {expected_shift}, got {m3}"


def test_cycling_opponent():
    """Verify Forward and Backward Cycling."""
    f_opp = CyclingOpponent(direction="forward", seed=42)
    m1 = f_opp.get_move()
    m2 = f_opp.get_move()
    m3 = f_opp.get_move()
    assert m2 == CYCLE_FORWARD[m1]
    assert m3 == CYCLE_FORWARD[m2]

    b_opp = CyclingOpponent(direction="backward", seed=42)
    bm1 = b_opp.get_move()
    bm2 = b_opp.get_move()
    assert bm2 == CYCLE_BACKWARD[bm1]


def test_copycat_opponent():
    """Verify Copycat mirrors agent's previous move."""
    opp = CopycatOpponent(mode="direct", seed=42)
    _ = opp.get_move()
    opp.observe("P")
    assert opp.get_move() == "P"
    opp.observe("S")
    assert opp.get_move() == "S"


def test_strategy_detector_detects_forward_cycling():
    """Verify RuleBasedStrategyDetector successfully identifies forward cycling."""
    detector = RuleBasedStrategyDetector(window_size=12, confidence_threshold=0.70)
    cycle = ["R", "P", "S", "R", "P", "S", "R", "P", "S", "R"]

    for opp_m in cycle:
        detector.observe_round("R", opp_m)

    res = detector.detect_strategy()
    assert res["detected"] is True
    assert res["strategy"] == "FORWARD_CYCLING"
    assert res["confidence"] >= 0.70
    assert res["predicted_move"] == "P"  # Last was R -> next must be P
    assert "Forward Cycling" in res["explanation"]


def test_strategy_detector_detects_direct_mirroring():
    """Verify detector identifies Copycat mirroring agent moves."""
    detector = RuleBasedStrategyDetector(window_size=10, confidence_threshold=0.70)
    agent_moves = ["R", "P", "S", "R", "S", "P", "R", "P"]

    for i in range(1, len(agent_moves)):
        # Opponent mirrors previous agent move
        detector.observe_round(agent_moves[i], agent_moves[i - 1])

    res = detector.detect_strategy()
    assert res["detected"] is True
    assert res["strategy"] == "DIRECT_MIRRORING"
    assert res["predicted_move"] == agent_moves[-1]
