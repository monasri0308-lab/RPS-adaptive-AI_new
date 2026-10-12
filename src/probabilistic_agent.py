"""
Probabilistic Opponent Modeller (Adaptive RPS AI)
=================================================
Unifies the complete multi-layer architecture:
1. CSP Layer: Stress-test against constraint-satisfaction adversaries.
2. Knowledge & Reasoning Layer: Rule-based human strategy detector (WSLS, Cycling, Copying).
3. Markov Pattern Modeling: Order-N Markov chain with hierarchical backoff.
4. Applied Bayesian Layer: Dirichlet-Multinomial Bayesian inference and Expected Utility Maximization.
5. Responsible AI Layer: Cryptographic fair-play commitments and transparent explainability cards.
"""

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


class ProbabilisticOpponentModeller:
    """Adaptive RPS Agent integrating Symbolic Reasoning, Bayesian Updating, and Decision Theory."""

    def __init__(
        self,
        order: int = 2,
        prior_alpha: float = 1.0,
        rule_threshold: float = 0.70,
        seed: Optional[int] = None,
    ):
        self.order = order
        self.prior_alpha = prior_alpha
        self.rule_threshold = rule_threshold
        self.rng = random.Random(seed)

        # 1. Knowledge & Reasoning Layer
        self.rule_detector = RuleBasedStrategyDetector(
            window_size=15, confidence_threshold=rule_threshold
        )

        # 2. Markov Pattern Model Layer
        self.markov_model = OpponentModel(order=order, seed=seed)

        # 3. Applied Bayesian Layer
        self.bayesian_predictor = BayesianPredictor(prior_alpha=prior_alpha)

        # 4. Responsible AI & Fair-Play Layer
        self.commitment = CryptographicCommitment()
        self.consent_manager = DataConsentManager()

        # State tracking
        self.round_counter = 0
        self.history: List[Tuple[str, str]] = []  # (agent_move, opp_move)
        self.probability_history: List[Dict[str, float]] = []
        self.last_commitment_hash: Optional[str] = None
        self.last_planned_move: Optional[str] = None
        self.last_explainability: Optional[Dict[str, Any]] = None

    def plan_move(self) -> Tuple[str, str]:
        """Pre-compute next move and return (agent_move, commitment_hash).

        Enforces fair-play commitment before human move is provided.
        """
        self.round_counter += 1

        # Step 1: Check symbolic reasoning layer
        reasoning = self.rule_detector.detect_strategy()

        if reasoning.get("detected", False) and reasoning.get("predicted_move"):
            predicted_opp_move = reasoning["predicted_move"]
            conf = reasoning["confidence"]

            other_p = max(0.01, (1.0 - conf) / 2.0)
            probs = {
                m: conf if m == predicted_opp_move else other_p for m in MOVES
            }
            # Normalize
            tot = sum(probs.values())
            probs = {m: probs[m] / tot for m in MOVES}

            eu = BayesianPredictor.calculate_expected_utilities(probs)
            agent_move = COUNTER_MOVES[predicted_opp_move]
            uncertainty = self.bayesian_predictor.get_uncertainty_metrics(probs)

        else:
            # Step 2: Check high-confidence Order-N Markov chain transition
            markov_matched = False
            for k in range(self.order, 0, -1):
                if len(self.markov_model.history) >= k:
                    ctx = tuple(self.markov_model.history[-k:])
                    if ctx in self.markov_model.n_grams[k]:
                        counts = self.markov_model.n_grams[k][ctx]
                        total_seen = sum(counts.values())
                        if total_seen >= 2:
                            max_c = max(counts.values())
                            ratio = max_c / total_seen
                            if ratio >= 0.75:
                                pred_m = max(counts, key=counts.get)
                                conf = min(0.95, ratio)
                                other_p = (1.0 - conf) / 2.0
                                probs = {
                                    m: conf if m == pred_m else other_p for m in MOVES
                                }
                                eu = BayesianPredictor.calculate_expected_utilities(probs)
                                agent_move = COUNTER_MOVES[pred_m]
                                uncertainty = self.bayesian_predictor.get_uncertainty_metrics(probs)
                                reasoning = {
                                    "detected": True,
                                    "strategy": f"ORDER_{k}_MARKOV_PATTERN",
                                    "confidence": round(ratio, 2),
                                    "predicted_move": pred_m,
                                    "explanation": f"Order-{k} context {ctx} historically transitioned to {pred_m} with {ratio:.0%} consistency ({max_c}/{total_seen} times).",
                                }
                                markov_matched = True
                                break

            # Step 3: Fall back to Bayesian Dirichlet predictive distribution & EU
            if not markov_matched:
                agent_move, probs, eu = self.bayesian_predictor.get_optimal_action()
                uncertainty = self.bayesian_predictor.get_uncertainty_metrics(probs)

                # If utilities are equal (e.g. uninformative prior at round 1), pick uniformly
                if all(abs(u - list(eu.values())[0]) < 1e-6 for u in eu.values()):
                    agent_move = self.rng.choice(MOVES)

        # Create cryptographic fair-play commitment
        commitment_hash = self.commitment.create_commitment(agent_move)

        self.last_planned_move = agent_move
        self.last_commitment_hash = commitment_hash
        self.probability_history.append(dict(probs))

        # Compile explainability
        self.last_explainability = format_explainability_card(
            round_num=self.round_counter,
            agent_move=agent_move,
            probs=probs,
            eu=eu,
            reasoning=reasoning,
            entropy=uncertainty["entropy"],
        )

        return agent_move, commitment_hash

    def get_move(self) -> str:
        """Standard interface: returns agent's planned move."""
        if self.last_planned_move is None:
            move, _ = self.plan_move()
            return move
        move = self.last_planned_move
        self.last_planned_move = None
        return move

    def observe(self, opponent_move: str) -> Dict[str, Any]:
        """Record opponent move, reveal fair-play commitment, and update models."""
        if opponent_move not in MOVES:
            raise ValueError(f"Invalid move '{opponent_move}'. Must be in {MOVES}")

        agent_move = (
            self.last_explainability["selected_action"]
            if self.last_explainability
            else "R"
        )

        # Reveal cryptographic commitment
        revealed_move, salt, comm_hash = self.commitment.reveal()
        verification_passed = CryptographicCommitment.verify(revealed_move, salt, comm_hash)

        # Evaluate round
        outcome = evaluate_round(agent_move, opponent_move)

        # Update knowledge & reasoning layer
        self.rule_detector.observe_round(agent_move, opponent_move)

        # Update Markov pattern model
        self.markov_model.update(opponent_move)

        # Update Bayesian layer
        self.bayesian_predictor.update(opponent_move)

        # Log telemetry if opted in
        if self.last_explainability:
            self.consent_manager.log_round(
                round_num=self.round_counter,
                human_move=opponent_move,
                agent_move=agent_move,
                outcome=outcome,
                predicted_probs=self.last_explainability["predicted_probabilities"],
                detected_strategy=self.last_explainability["active_strategy"],
            )

        self.history.append((agent_move, opponent_move))
        self.last_planned_move = None

        return {
            "agent_move": agent_move,
            "opponent_move": opponent_move,
            "outcome": outcome,
            "revealed_salt": salt,
            "commitment_hash": comm_hash,
            "verification_passed": verification_passed,
            "explainability": self.last_explainability,
        }

    def reset(self) -> None:
        """Reset internal states."""
        self.round_counter = 0
        self.rule_detector.reset()
        self.markov_model.reset()
        self.bayesian_predictor.reset()
        self.history.clear()
        self.probability_history.clear()
        self.last_planned_move = None
        self.last_commitment_hash = None
        self.last_explainability = None
