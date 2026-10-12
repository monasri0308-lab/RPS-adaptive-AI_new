"""
Unit and Integration Tests for the CSP Layer
============================================
Verifies:
- Iterative backtracking solver satisfies balance (|Count(m) - N/3| <= 1).
- No triple repeats constraint (zero RRR, PPP, SSS).
- Low order-1 predictability (max transition prob <= threshold, high transition entropy).
- CSPAdversaryOpponent plays consistent valid moves.
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from csp_adversary import CSPSequenceGenerator, CSPAdversaryOpponent, MOVES


def test_csp_generator_constraints_short():
    """Verify CSP solver satisfies all 3 constraints on N=150."""
    gen = CSPSequenceGenerator(length=150, max_trans_prob=0.45, seed=42)
    seq = gen.generate()
    assert len(seq) == 150

    audit = CSPSequenceGenerator.verify_constraints(seq, max_trans_prob=0.45)
    assert audit["balance_satisfied"] is True
    assert audit["no_triples_satisfied"] is True
    assert audit["triple_repeats_found"] == 0
    assert audit["all_constraints_satisfied"] is True


def test_csp_generator_constraints_benchmark_length():
    """Verify CSP solver satisfies all constraints on N=1000."""
    gen = CSPSequenceGenerator(length=1000, max_trans_prob=0.45, seed=123)
    seq = gen.generate()
    assert len(seq) == 1000

    audit = CSPSequenceGenerator.verify_constraints(seq, max_trans_prob=0.45)
    assert audit["balance_satisfied"] is True
    assert audit["no_triples_satisfied"] is True
    assert audit["triple_repeats_found"] == 0
    # Each move count must be either 333 or 334
    for m in MOVES:
        assert 333 <= audit["move_counts"][m] <= 334
    # Max transition probability must be low (< 0.45)
    assert audit["max_observed_trans_prob"] <= 0.45
    # Transition entropy must be close to maximum (~1.58 bits)
    for ent in audit["transition_entropies"].values():
        assert ent > 1.45


def test_csp_adversary_opponent():
    """Verify CSPAdversaryOpponent interface and move production."""
    opp = CSPAdversaryOpponent(length=100, max_trans_prob=0.45, seed=42)
    moves = [opp.get_move() for _ in range(100)]
    assert len(moves) == 100
    assert all(m in MOVES for m in moves)
    assert opp.audit["all_constraints_satisfied"] is True

    # Test reset
    opp.reset()
    first_move = opp.get_move()
    assert first_move == moves[0]
