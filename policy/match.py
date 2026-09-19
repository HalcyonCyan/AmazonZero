"""Shared match generation and alternating-color evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import FrozenSet, List, Tuple

import numpy as np

from policy.environment import AmazonsEnv
from policy.mcts import select_from_policy
from policy.strategies import Policy
from policy.value import perspective_value


@dataclass
class TrainingExample:
    state: np.ndarray
    policy: np.ndarray
    legal_mask: np.ndarray
    player: int
    value: float = 0.0


class MatchRunner:
    """Run any pair of policies; collect only explicitly selected players.

    Training players receive search noise and early temperature sampling.
    Other policies are sampled at temperature 1, retaining heuristic randomness.
    Evaluation players use greedy actions without search noise.
    """

    def __init__(self, board_size: int, exploration_turns: int = 8):
        if exploration_turns < 0:
            raise ValueError("exploration_turns must be nonnegative")
        self.board_size = board_size
        self.exploration_turns = exploration_turns

    def play_game(
        self,
        player1: Policy,
        player2: Policy,
        *,
        collect_players: FrozenSet[int] = frozenset(),
        training: bool = False,
        greedy_players: FrozenSet[int] = frozenset(),
    ) -> Tuple[List[TrainingExample], int]:
        if not set(collect_players).union(greedy_players).issubset({1, 2}):
            raise ValueError("Players must be 1 or 2")
        env = AmazonsEnv(self.board_size)
        policies = {1: player1, 2: player2}
        # One shared self-play policy must observe each action exactly once.
        unique_policies = list({id(policy): policy for policy in policies.values()}.values())
        for policy in unique_policies:
            policy.reset()
        history: List[TrainingExample] = []

        while not env.is_terminal():
            player = env.current_player
            learn = training and player in collect_players
            probabilities = np.asarray(
                policies[player].probabilities(env, explore=learn), dtype=np.float32
            )
            mask = env.legal_mask()
            if (
                probabilities.shape != (env.action_size,)
                or not np.isfinite(probabilities).all()
                or (probabilities < 0).any()
                or not np.isclose(probabilities.sum(), 1.0)
                or (probabilities[~mask] != 0).any()
            ):
                raise ValueError("Policy must return a normalized, legal probability vector")
            if player in collect_players:
                history.append(TrainingExample(
                    state=env.encode(), policy=probabilities.copy(), legal_mask=mask, player=player
                ))
            temperature = 1.0
            if player in greedy_players or (learn and env.completed_turns >= self.exploration_turns):
                temperature = 0.0
            action = select_from_policy(probabilities, temperature)
            env.step(action)
            for policy in unique_policies:
                policy.observe_action(action)

        winner = env.winner
        assert winner is not None
        for example in history:
            example.value = perspective_value(1.0, winner, example.player)
        return history, winner


def evaluate_policies(
    board_size: int, learner: Policy, opponent: Policy, games: int
) -> float:
    """Learner win rate with alternating colors, greedy learner, no root noise."""
    if games < 1:
        raise ValueError("evaluation games must be positive")
    runner = MatchRunner(board_size)
    wins = 0
    for index in range(games):
        learner_player = 1 + index % 2
        players = (learner, opponent) if learner_player == 1 else (opponent, learner)
        greedy = {learner_player}
        if learner is opponent:
            greedy = {1, 2}
        _, winner = runner.play_game(*players, greedy_players=frozenset(greedy))
        wins += winner == learner_player
    return wins / games
