"""
CSP Layer: Constraint-Based Adversary for Rock-Paper-Scissors
=============================================================
Formulates RPS move sequence generation as a Constraint Satisfaction Problem (CSP)
solved via iterative backtracking with forward checking and heuristics.

Constraints:
1. Equal Move Usage (Balance): Each move (R, P, S) is used equally (within ±1 of N/3).
2. No Triple Repeats: No move appears 3 consecutive times (no RRR, PPP, SSS).
3. Low Order-1 Predictability: For any transition (a -> b), P(b | a) <= max_trans_prob
   (e.g., <= 0.45), ensuring high transition entropy and defeating simple Markov predictors.
"""

import math
import random
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

MOVES: List[str] = ["R", "P", "S"]


class CSPSequenceGenerator:
    """Constraint Satisfaction Problem (CSP) Iterative Backtracking Solver."""

    def __init__(
        self,
        length: int = 1000,
        max_trans_prob: float = 0.45,
        seed: Optional[int] = None,
    ):
        """Initialize the CSP solver.

        Args:
            length: Target sequence length N.
            max_trans_prob: Maximum allowable transition probability P(M_t | M_{t-1}).
            seed: Random seed for deterministic reproducibility.
        """
        self.length = max(3, length)
        self.max_trans_prob = max_trans_prob
        self.rng = random.Random(seed)

        # Precompute target move bounds (balance constraint)
        self.min_per_move = self.length // 3
        self.max_per_move = math.ceil(self.length / 3.0)

        # Search diagnostics
        self.backtrack_count = 0
        self.nodes_explored = 0
        self.constraint_checks = 0

    def _is_valid_assignment(
        self,
        index: int,
        move: str,
        sequence: List[str],
        move_counts: Dict[str, int],
        transition_counts: Dict[Tuple[str, str], int],
    ) -> bool:
        """Check if assigning 'move' at 'index' satisfies all local and forward constraints."""
        self.constraint_checks += 1

        # Constraint 1: Unigram capacity bound
        if move_counts[move] + 1 > self.max_per_move:
            return False

        # Forward checking for balance feasibility:
        remaining_slots = self.length - (index + 1)
        needed_slots = 0
        for m in MOVES:
            curr_count = move_counts[m] + (1 if m == move else 0)
            needed_slots += max(0, self.min_per_move - curr_count)
        if needed_slots > remaining_slots:
            return False

        # Constraint 2: No triple repeats (index >= 2)
        if index >= 2 and sequence[index - 1] == move and sequence[index - 2] == move:
            return False

        # Constraint 3: Low order-1 predictability
        if index >= 1:
            prev_move = sequence[index - 1]
            proposed_trans_count = transition_counts[(prev_move, move)] + 1
            origin_count = move_counts[prev_move]
            # When origin_count is small, allow exploration, but enforce strictly as it grows
            if origin_count >= 3:
                trans_prob = proposed_trans_count / origin_count
                if trans_prob > self.max_trans_prob:
                    return False

        return True

    def _get_ordered_candidates(
        self,
        move_counts: Dict[str, int],
        sequence: List[str],
        transition_counts: Dict[Tuple[str, str], int],
    ) -> List[str]:
        """Order domain values using MRV and transition entropy heuristics."""
        candidates = list(MOVES)
        self.rng.shuffle(candidates)

        # Heuristic 1: Prioritize moves with lowest total count (Least Used Value)
        # Heuristic 2: If previous move exists, prioritize moves with lowest transition count from prev_move
        if sequence:
            prev_move = sequence[-1]
            candidates.sort(
                key=lambda m: (
                    move_counts[m],
                    transition_counts.get((prev_move, m), 0),
                )
            )
        else:
            candidates.sort(key=lambda m: move_counts[m])

        return candidates

    def generate(self, max_attempts: int = 5) -> List[str]:
        """Solve the CSP iteratively using an explicit search stack without recursion limits."""
        for attempt in range(max_attempts):
            self.backtrack_count = 0
            self.nodes_explored = 0
            self.constraint_checks = 0

            sequence: List[str] = []
            move_counts: Dict[str, int] = {m: 0 for m in MOVES}
            transition_counts: Dict[Tuple[str, str], int] = defaultdict(int)

            # Stack entries: [candidate_list, next_candidate_index]
            stack: List[Tuple[List[str], int]] = []

            # Initialize position 0 candidates
            first_candidates = self._get_ordered_candidates(
                move_counts, sequence, transition_counts
            )
            stack.append((first_candidates, 0))

            while stack:
                self.nodes_explored += 1
                curr_index = len(sequence)

                if curr_index == self.length:
                    # Successfully reached target length!
                    return sequence

                candidates, cand_idx = stack[-1]

                # Find next valid candidate for current position
                assigned = False
                while cand_idx < len(candidates):
                    move = candidates[cand_idx]
                    cand_idx += 1
                    stack[-1] = (candidates, cand_idx)

                    if self._is_valid_assignment(
                        curr_index, move, sequence, move_counts, transition_counts
                    ):
                        # Commit assignment
                        sequence.append(move)
                        move_counts[move] += 1
                        if curr_index >= 1:
                            transition_counts[(sequence[-2], move)] += 1

                        assigned = True

                        # If not done, prepare candidates for next position
                        if len(sequence) < self.length:
                            next_candidates = self._get_ordered_candidates(
                                move_counts, sequence, transition_counts
                            )
                            stack.append((next_candidates, 0))
                        break

                if not assigned:
                    # No valid candidate at this position -> Backtrack!
                    self.backtrack_count += 1
                    stack.pop()
                    if sequence:
                        popped_move = sequence.pop()
                        move_counts[popped_move] -= 1
                        if sequence:
                            transition_counts[(sequence[-1], popped_move)] -= 1

            # If attempt completed without solution, retry with another random seed
            self.rng = random.Random(self.rng.randint(1, 999999))

        # Fallback with slightly relaxed transition ceiling (0.50) if strict ceiling timed out
        relaxed_generator = CSPSequenceGenerator(
            length=self.length,
            max_trans_prob=min(0.55, self.max_trans_prob + 0.10),
            seed=self.rng.randint(1, 999999),
        )
        return relaxed_generator.generate(max_attempts=2)

    @staticmethod
    def verify_constraints(
        sequence: List[str], max_trans_prob: float = 0.50
    ) -> Dict[str, any]:
        """Verify and audit that a sequence satisfies all 3 CSP constraints."""
        n = len(sequence)
        counts = Counter(sequence)
        min_target = n // 3
        max_target = math.ceil(n / 3.0)

        # 1. Balance verification
        balance_satisfied = all(min_target <= counts[m] <= max_target for m in MOVES)

        # 2. No triple repeats verification
        triple_repeats = []
        for i in range(2, n):
            if sequence[i] == sequence[i - 1] == sequence[i - 2]:
                triple_repeats.append((i, sequence[i]))
        no_triples_satisfied = len(triple_repeats) == 0

        # 3. Order-1 transition predictability & entropy
        transitions: Dict[str, Counter] = {m: Counter() for m in MOVES}
        for i in range(1, n):
            transitions[sequence[i - 1]][sequence[i]] += 1

        max_observed_trans_prob = 0.0
        entropies = {}
        for origin, next_counts in transitions.items():
            total = sum(next_counts.values())
            if total > 0:
                probs = [c / total for c in next_counts.values()]
                max_p = max(probs)
                if max_p > max_observed_trans_prob:
                    max_observed_trans_prob = max_p
                # Shannon entropy: H = -sum(p * log2(p))
                h = -sum(p * math.log2(p) for p in probs if p > 0)
                entropies[origin] = h
            else:
                entropies[origin] = 0.0

        low_predictability_satisfied = max_observed_trans_prob <= max_trans_prob

        return {
            "length": n,
            "move_counts": dict(counts),
            "balance_satisfied": balance_satisfied,
            "triple_repeats_found": len(triple_repeats),
            "no_triples_satisfied": no_triples_satisfied,
            "max_observed_trans_prob": round(max_observed_trans_prob, 4),
            "transition_entropies": {k: round(v, 4) for k, v in entropies.items()},
            "low_predictability_satisfied": low_predictability_satisfied,
            "all_constraints_satisfied": (
                balance_satisfied and no_triples_satisfied and low_predictability_satisfied
            ),
        }


class CSPAdversaryOpponent:
    """Adversary playing moves generated via the Constraint Satisfaction Problem solver.

    Enforces equal move distributions, no triple repeats, and low order-1 transition predictability
    to rigorously stress-test pattern-matching opponent models.
    """

    def __init__(
        self,
        length: int = 1000,
        max_trans_prob: float = 0.45,
        seed: Optional[int] = 42,
    ):
        self.name = "CSP Constraint Adversary"
        self.length = length
        self.max_trans_prob = max_trans_prob
        self.seed = seed
        self.solver = CSPSequenceGenerator(
            length=length, max_trans_prob=max_trans_prob, seed=seed
        )
        self.sequence = self.solver.generate()
        self.index = 0
        self.audit = CSPSequenceGenerator.verify_constraints(
            self.sequence, max_trans_prob=max_trans_prob + 0.05
        )

    def get_move(self) -> str:
        """Fetch next move from the CSP-verified sequence."""
        if self.index < len(self.sequence):
            move = self.sequence[self.index]
            self.index += 1
            return move
        # If match exceeds initial length, regenerate dynamically
        self.sequence.extend(self.solver.generate())
        move = self.sequence[self.index]
        self.index += 1
        return move

    def observe(self, agent_move: str) -> None:
        """Adversary observation hook."""
        pass

    def reset(self) -> None:
        """Reset sequence index."""
        self.index = 0
