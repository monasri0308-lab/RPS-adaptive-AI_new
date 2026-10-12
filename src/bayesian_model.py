"""
Applied Bayesian Layer: Dirichlet-Multinomial Inference & Decision Theory
========================================================================
Implements Bayesian reasoning under uncertainty for Rock-Paper-Scissors:
1. Dirichlet-Multinomial Conjugate Model:
   - Prior: Dirichlet(alpha_R, alpha_P, alpha_S)
   - Online Bayesian updating: alpha_m <- alpha_m + 1
   - Posterior Predictive Distribution: P(M_{t+1} = m) = alpha_m / sum(alpha)
   - Uncertainty metrics: Dirichlet variance & Shannon entropy
2. Order-1 Bayesian Markov Model:
   - Conditional Dirichlet priors for each transition state s in {R, P, S}
3. Decision Theory:
   - Expected Utility Maximization over payoffs {Win: +1, Tie: 0, Loss: -1}
4. Bayesian Opponent Agent for competitive benchmarking.
"""

import math
import random
from typing import Dict, List, Optional, Tuple

MOVES: List[str] = ["R", "P", "S"]
COUNTER_MOVES: Dict[str, str] = {"R": "P", "P": "S", "S": "R"}


class BayesianPredictor:
    """Bayesian Dirichlet-Multinomial and Order-1 Markov Opponent Predictor."""

    def __init__(
        self,
        prior_alpha: float = 1.0,
        markov_prior_alpha: float = 1.0,
        decay_factor: float = 1.0,
    ):
        """Initialize Bayesian models.

        Args:
            prior_alpha: Initial symmetric Dirichlet prior weight (1.0 = Laplace uniform).
            markov_prior_alpha: Prior weight for conditional transition models.
            decay_factor: Optional recency discounting (1.0 = standard stationary Bayesian update).
        """
        self.prior_alpha = prior_alpha
        self.markov_prior_alpha = markov_prior_alpha
        self.decay_factor = decay_factor

        # Global Unigram Dirichlet parameters: {move: alpha}
        self.unigram_alphas: Dict[str, float] = {m: prior_alpha for m in MOVES}

        # Order-1 Transition Dirichlet parameters: {prev_move: {next_move: alpha}}
        self.markov_alphas: Dict[str, Dict[str, float]] = {
            m: {next_m: markov_prior_alpha for next_m in MOVES} for m in MOVES
        }

        self.history: List[str] = []

    def update(self, opponent_move: str) -> None:
        """Perform exact Bayesian posterior update upon observing opponent_move."""
        if opponent_move not in MOVES:
            raise ValueError(f"Invalid move '{opponent_move}'. Must be in {MOVES}")

        # Optional decay for non-stationary adaptation
        if self.decay_factor < 1.0:
            for m in MOVES:
                self.unigram_alphas[m] = self.prior_alpha + (
                    self.unigram_alphas[m] - self.prior_alpha
                ) * self.decay_factor
                for next_m in MOVES:
                    self.markov_alphas[m][next_m] = self.markov_prior_alpha + (
                        self.markov_alphas[m][next_m] - self.markov_prior_alpha
                    ) * self.decay_factor

        # Unigram Dirichlet update: alpha_m <- alpha_m + 1
        self.unigram_alphas[opponent_move] += 1.0

        # Conditional Order-1 Markov Dirichlet update
        if self.history:
            prev_move = self.history[-1]
            self.markov_alphas[prev_move][opponent_move] += 1.0

        self.history.append(opponent_move)

    def get_unigram_distribution(self) -> Dict[str, float]:
        """Compute the unigram posterior predictive distribution E[theta_m | data]."""
        total_alpha = sum(self.unigram_alphas.values())
        return {m: self.unigram_alphas[m] / total_alpha for m in MOVES}

    def get_markov_distribution(self, prev_move: str) -> Dict[str, float]:
        """Compute the conditional posterior predictive distribution P(M_{t+1} | M_t = prev_move)."""
        alphas = self.markov_alphas[prev_move]
        total_alpha = sum(alphas.values())
        return {m: alphas[m] / total_alpha for m in MOVES}

    def get_predictive_distribution(self) -> Dict[str, float]:
        """Get the current posterior predictive distribution for the opponent's next move.

        Blends Order-1 Markov predictive distribution with unigram distribution based on
        effective sample size in the current transition state.
        """
        if not self.history:
            return self.get_unigram_distribution()

        prev_move = self.history[-1]
        markov_dist = self.get_markov_distribution(prev_move)
        unigram_dist = self.get_unigram_distribution()

        # Weight transition model according to observed transitions from prev_move
        observed_transitions = sum(self.markov_alphas[prev_move].values()) - (
            3 * self.markov_prior_alpha
        )

        if observed_transitions <= 0:
            return unigram_dist

        # Credibility weight lambda in [0, 1]
        credibility = min(1.0, observed_transitions / 10.0)
        blended = {}
        for m in MOVES:
            blended[m] = (credibility * markov_dist[m]) + ((1.0 - credibility) * unigram_dist[m])

        # Normalize
        total = sum(blended.values())
        return {m: blended[m] / total for m in MOVES}

    def get_uncertainty_metrics(self, probs: Dict[str, float]) -> Dict[str, float]:
        """Compute entropy and Dirichlet variances to quantify epistemic and aleatoric uncertainty."""
        # Shannon Entropy H(p) = -sum(p * log2(p))
        entropy = -sum(p * math.log2(p) for p in probs.values() if p > 0)

        # Total observations
        total_alpha = sum(self.unigram_alphas.values())
        variances = {}
        for m in MOVES:
            a = self.unigram_alphas[m]
            # Dirichlet marginal variance: Var(theta_m) = alpha_m * (alpha_0 - alpha_m) / (alpha_0^2 * (alpha_0 + 1))
            variances[m] = (a * (total_alpha - a)) / (total_alpha**2 * (total_alpha + 1))

        return {
            "entropy": round(entropy, 4),
            "max_entropy": round(math.log2(3), 4),  # ~1.5850 bits
            "normalized_entropy": round(entropy / math.log2(3), 4),  # 0 to 1 scale
            "variances": {k: round(v, 6) for k, v in variances.items()},
        }

    @staticmethod
    def calculate_expected_utilities(probs: Dict[str, float]) -> Dict[str, float]:
        """Compute Expected Utility for each candidate action under payoff {Win: +1, Tie: 0, Loss: -1}.

        EU(Rock) = P(Scissors) - P(Paper)
        EU(Paper) = P(Rock) - P(Scissors)
        EU(Scissors) = P(Paper) - P(Rock)
        """
        p_r = probs["R"]
        p_p = probs["P"]
        p_s = probs["S"]

        return {
            "R": round(p_s - p_p, 4),
            "P": round(p_r - p_s, 4),
            "S": round(p_p - p_r, 4),
        }

    def get_optimal_action(self) -> Tuple[str, Dict[str, float], Dict[str, float]]:
        """Compute optimal move via Expected Utility Maximization.

        Returns:
            Tuple of (optimal_move, probability_distribution, expected_utilities)
        """
        probs = self.get_predictive_distribution()
        eu = self.calculate_expected_utilities(probs)
        best_move = max(eu, key=eu.get)
        return best_move, probs, eu

    def reset(self) -> None:
        """Reset Bayesian prior counts and history."""
        self.unigram_alphas = {m: self.prior_alpha for m in MOVES}
        self.markov_alphas = {
            m: {next_m: self.markov_prior_alpha for next_m in MOVES} for m in MOVES
        }
        self.history.clear()


class BayesianOpponent:
    """Bayesian Opponent modeling the player/agent via Dirichlet Markov inference."""

    def __init__(self, prior_alpha: float = 1.0, seed: Optional[int] = None):
        self.name = "Bayesian Dirichlet Opponent"
        self.predictor = BayesianPredictor(prior_alpha=prior_alpha)
        self.rng = random.Random(seed)

    def get_move(self) -> str:
        best_move, probs, eu = self.predictor.get_optimal_action()
        # If all expected utilities are 0 (e.g. at round 1), pick uniformly
        if all(u == 0.0 for u in eu.values()):
            return self.rng.choice(MOVES)
        return best_move

    def observe(self, agent_move: str) -> None:
        self.predictor.update(agent_move)

    def reset(self) -> None:
        self.predictor.reset()
