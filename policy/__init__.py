"""Public policy API; heuristic matches do not require PyTorch."""

from policy.environment import AmazonsEnv, Phase
from policy.match import MatchRunner, TrainingExample, evaluate_policies
from policy.mcts import MCTS, MCTSConfig
from policy.opponents import OpponentSchedule, TRAINING_MODES
from policy.strategies import (
    MCTSPolicy, NearOpponentPolicy, POLICY_NAMES, Policy, RandomPolicy, create_policy,
)

__all__ = [
    "AmazonsEnv", "Phase", "Policy", "RandomPolicy", "NearOpponentPolicy",
    "MCTSPolicy", "create_policy", "POLICY_NAMES", "MCTS", "MCTSConfig",
    "MatchRunner", "TrainingExample", "evaluate_policies", "OpponentSchedule",
    "TRAINING_MODES",
]
