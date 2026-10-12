"""
RPS Opponent Modeling Package
"""

from .bayesian_model import BayesianOpponent, BayesianPredictor
from .csp_adversary import CSPAdversaryOpponent, CSPSequenceGenerator
from .probabilistic_agent import ProbabilisticOpponentModeller
from .reasoning_layer import (
    CopycatOpponent,
    CyclingOpponent,
    RuleBasedStrategyDetector,
    WinStayLoseShiftOpponent,
)
from .rps_agent import (
    AdaptiveAgent,
    BiasedStreakOpponent,
    OpponentModel,
    PureRandomOpponent,
    RandomAgent,
    RotatingOpponent,
)

__all__ = [
    "AdaptiveAgent",
    "BayesianOpponent",
    "BayesianPredictor",
    "BiasedStreakOpponent",
    "CSPAdversaryOpponent",
    "CSPSequenceGenerator",
    "CopycatOpponent",
    "CyclingOpponent",
    "OpponentModel",
    "ProbabilisticOpponentModeller",
    "PureRandomOpponent",
    "RandomAgent",
    "RotatingOpponent",
    "RuleBasedStrategyDetector",
    "WinStayLoseShiftOpponent",
]
