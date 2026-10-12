"""
Knowledge & Reasoning Layer: Symbolic Human Strategy Detector & Heuristic Opponents
===================================================================================
Encodes human cognitive biases and behavioral patterns as explicit symbolic rules:
1. Win-Stay / Lose-Shift (WSLS) (Wang et al., 2014)
2. Directional Cycling (Forward: R->P->S, Backward: R->S->P)
3. Copying / Mirroring (Tit-for-Tat direct copy or counter-copy)
4. Streak Alternation / Gambler's Fallacy (anti-repetition)

Provides:
- Realistic rule-driven opponents for benchmarking
- An online RuleBasedStrategyDetector evaluating candidate hypotheses from recent history
  with explainable reasoning output.
"""

import random
from typing import Dict, List, Optional, Tuple

MOVES: List[str] = ["R", "P", "S"]
COUNTER_MOVES: Dict[str, str] = {"R": "P", "P": "S", "S": "R"}
BEATEN_BY: Dict[str, str] = {"P": "R", "S": "P", "R": "S"}  # Reverse counter
CYCLE_FORWARD: Dict[str, str] = {"R": "P", "P": "S", "S": "R"}
CYCLE_BACKWARD: Dict[str, str] = {"R": "S", "S": "P", "P": "R"}


def evaluate_round(agent_move: str, opponent_move: str) -> str:
    """Evaluate outcome from agent perspective: 'WIN', 'LOSS', 'TIE'."""
    if agent_move == opponent_move:
        return "TIE"
    elif COUNTER_MOVES[opponent_move] == agent_move:
        return "WIN"
    else:
        return "LOSS"


# =====================================================================
# RULE-BASED OPPONENT IMPLEMENTATIONS
# =====================================================================


class WinStayLoseShiftOpponent:
    """Human-like opponent executing the classic Win-Stay / Lose-Shift heuristic.

    Empirical Psychology:
    - If opponent won previous round, they repeat the winning move (Win-Stay).
    - If opponent lost previous round, they shift to the move that counters what beat them
      or shift cyclically (Lose-Shift).
    - If tie, random or slight bias to shift.
    """

    def __init__(self, shift_mode: str = "counter_beater", seed: Optional[int] = None):
        self.name = "Win-Stay / Lose-Shift"
        self.shift_mode = shift_mode
        self.rng = random.Random(seed)
        self.last_move: Optional[str] = None
        self.last_agent_move: Optional[str] = None
        self.last_outcome_for_me: Optional[str] = None  # From opponent's view

    def get_move(self) -> str:
        if self.last_move is None:
            move = self.rng.choice(MOVES)
        elif self.last_outcome_for_me == "WIN":
            # Win-Stay: Repeat the winning move
            move = self.last_move
        elif self.last_outcome_for_me == "LOSS":
            # Lose-Shift
            if self.shift_mode == "counter_beater" and self.last_agent_move:
                # Play what counters the agent's winning move
                move = COUNTER_MOVES[self.last_agent_move]
            else:
                # Cyclic shift
                move = CYCLE_FORWARD[self.last_move]
        else:
            # TIE: Shift randomly to one of the other two
            other_moves = [m for m in MOVES if m != self.last_move]
            move = self.rng.choice(other_moves)

        self.last_move = move
        return move

    def observe(self, agent_move: str) -> None:
        self.last_agent_move = agent_move
        if self.last_move is not None:
            # Outcome for opponent: opponent wins if opponent beats agent
            if agent_move == self.last_move:
                self.last_outcome_for_me = "TIE"
            elif COUNTER_MOVES[agent_move] == self.last_move:
                self.last_outcome_for_me = "WIN"
            else:
                self.last_outcome_for_me = "LOSS"

    def reset(self) -> None:
        self.last_move = None
        self.last_agent_move = None
        self.last_outcome_for_me = None


class CyclingOpponent:
    """Opponent adhering strictly to cyclic transition patterns (Forward or Backward)."""

    def __init__(self, direction: str = "forward", seed: Optional[int] = None):
        self.direction = direction
        self.name = f"Cycling ({'Forward R->P->S' if direction == 'forward' else 'Backward R->S->P'})"
        self.cycle_map = CYCLE_FORWARD if direction == "forward" else CYCLE_BACKWARD
        self.rng = random.Random(seed)
        self.last_move: Optional[str] = None

    def get_move(self) -> str:
        if self.last_move is None:
            move = self.rng.choice(MOVES)
        else:
            move = self.cycle_map[self.last_move]
        self.last_move = move
        return move

    def observe(self, agent_move: str) -> None:
        pass

    def reset(self) -> None:
        self.last_move = None


class CopycatOpponent:
    """Tit-for-Tat / Mirroring opponent mimicking the agent's recent move."""

    def __init__(self, mode: str = "direct", seed: Optional[int] = None):
        """Args:
            mode: 'direct' (plays agent's last move) or 'counter' (counters agent's last move).
        """
        self.mode = mode
        self.name = f"Copycat ({'Direct Mirroring' if mode == 'direct' else 'Counter-Mirroring'})"
        self.rng = random.Random(seed)
        self.last_agent_move: Optional[str] = None

    def get_move(self) -> str:
        if self.last_agent_move is None:
            return self.rng.choice(MOVES)
        if self.mode == "direct":
            return self.last_agent_move
        else:
            return COUNTER_MOVES[self.last_agent_move]

    def observe(self, agent_move: str) -> None:
        self.last_agent_move = agent_move

    def reset(self) -> None:
        self.last_agent_move = None


# =====================================================================
# SYMBOLIC RULE-BASED STRATEGY DETECTOR
# =====================================================================


class RuleBasedStrategyDetector:
    """Infers opponent high-level strategies from round history using deductive rule testing.

    Evaluates hypotheses over a sliding history window:
    - Win-Stay / Lose-Shift (WSLS)
    - Forward Cycling (R -> P -> S -> R)
    - Backward Cycling (R -> S -> P -> R)
    - Direct Mirroring (Copying Agent's last move)
    - Counter Mirroring (Countering Agent's last move)
    """

    STRATEGIES = [
        "WIN_STAY_LOSE_SHIFT",
        "FORWARD_CYCLING",
        "BACKWARD_CYCLING",
        "DIRECT_MIRRORING",
        "COUNTER_MIRRORING",
    ]

    def __init__(self, window_size: int = 15, confidence_threshold: float = 0.70):
        self.window_size = window_size
        self.confidence_threshold = confidence_threshold

        # History buffers:
        # Each entry: (agent_move, opp_move, opp_outcome)
        self.history: List[Tuple[str, str, str]] = []

    def observe_round(self, agent_move: str, opponent_move: str) -> None:
        """Record completed round."""
        if agent_move == opponent_move:
            opp_outcome = "TIE"
        elif COUNTER_MOVES[agent_move] == opponent_move:
            opp_outcome = "WIN"
        else:
            opp_outcome = "LOSS"

        self.history.append((agent_move, opponent_move, opp_outcome))

    def reset(self) -> None:
        self.history.clear()

    def detect_strategy(self) -> Dict[str, any]:
        """Audit recent history window to identify active human heuristic strategy.

        Returns:
            Dict containing detected strategy name, confidence score, predicted move,
            and symbolic natural language explanation.
        """
        n = len(self.history)
        if n < 4:
            return {
                "detected": False,
                "strategy": "INSUFFICIENT_DATA",
                "confidence": 0.0,
                "predicted_move": None,
                "explanation": "Insufficient history (< 4 rounds) to deduce rule-based strategy.",
            }

        # Analyze window
        start_idx = max(1, n - self.window_size)
        window = self.history[start_idx:n]
        num_eval = len(window)

        scores: Dict[str, int] = {s: 0 for s in self.STRATEGIES}

        for i in range(start_idx, n):
            prev_agent, prev_opp, prev_opp_outcome = self.history[i - 1]
            curr_agent, curr_opp, curr_opp_outcome = self.history[i]

            # 1. Test WSLS (Wang et al.: Win-stay, Lose-shift to counter what beat them):
            if prev_opp_outcome == "WIN" and curr_opp == prev_opp:
                scores["WIN_STAY_LOSE_SHIFT"] += 1
            elif prev_opp_outcome == "LOSS" and curr_opp == COUNTER_MOVES[prev_agent]:
                scores["WIN_STAY_LOSE_SHIFT"] += 1
            elif prev_opp_outcome == "TIE" and curr_opp != prev_opp:
                scores["WIN_STAY_LOSE_SHIFT"] += 1

            # 2. Test Forward Cycling:
            if curr_opp == CYCLE_FORWARD[prev_opp]:
                scores["FORWARD_CYCLING"] += 1

            # 3. Test Backward Cycling:
            if curr_opp == CYCLE_BACKWARD[prev_opp]:
                scores["BACKWARD_CYCLING"] += 1

            # 4. Test Direct Mirroring:
            if curr_opp == prev_agent:
                scores["DIRECT_MIRRORING"] += 1

            # 5. Test Counter Mirroring:
            if curr_opp == COUNTER_MOVES[prev_agent]:
                scores["COUNTER_MIRRORING"] += 1

        # Calculate confidences
        confidences = {s: scores[s] / num_eval for s in self.STRATEGIES}
        best_strategy = max(confidences, key=confidences.get)
        best_conf = confidences[best_strategy]

        if best_conf >= self.confidence_threshold:
            # Predict next move using top rule
            last_agent, last_opp, last_opp_outcome = self.history[-1]
            predicted_move = None
            explanation = ""

            if best_strategy == "WIN_STAY_LOSE_SHIFT":
                if last_opp_outcome == "WIN":
                    predicted_move = last_opp
                    explanation = f"Opponent won last round with {last_opp}; WSLS rule prescribes Win-Stay -> {last_opp}."
                elif last_opp_outcome == "LOSS":
                    predicted_move = COUNTER_MOVES[last_agent]
                    explanation = f"Opponent lost to agent's {last_agent}; WSLS prescribes shift to counter {last_agent} -> {predicted_move}."
                else:
                    predicted_move = CYCLE_FORWARD[last_opp]
                    explanation = f"Opponent tied last round; WSLS prescribes cyclic shift from {last_opp} -> {predicted_move}."

            elif best_strategy == "FORWARD_CYCLING":
                predicted_move = CYCLE_FORWARD[last_opp]
                explanation = f"Opponent adheres to Forward Cycling (R->P->S). Last move {last_opp} -> {predicted_move}."

            elif best_strategy == "BACKWARD_CYCLING":
                predicted_move = CYCLE_BACKWARD[last_opp]
                explanation = f"Opponent adheres to Backward Cycling (R->S->P). Last move {last_opp} -> {predicted_move}."

            elif best_strategy == "DIRECT_MIRRORING":
                predicted_move = last_agent
                explanation = f"Opponent mimics Agent's previous move {last_agent} -> {predicted_move}."

            elif best_strategy == "COUNTER_MIRRORING":
                predicted_move = COUNTER_MOVES[last_agent]
                explanation = f"Opponent counters Agent's previous move {last_agent} -> {predicted_move}."

            return {
                "detected": True,
                "strategy": best_strategy,
                "confidence": round(best_conf, 3),
                "predicted_move": predicted_move,
                "explanation": explanation,
                "all_confidences": {k: round(v, 3) for k, v in confidences.items()},
            }

        return {
            "detected": False,
            "strategy": "NONE_CONFIDENT",
            "confidence": round(best_conf, 3),
            "predicted_move": None,
            "explanation": f"No single rule exceeded confidence threshold ({self.confidence_threshold:.0%}). Top candidate: {best_strategy} ({best_conf:.0%}).",
            "all_confidences": {k: round(v, 3) for k, v in confidences.items()},
        }
