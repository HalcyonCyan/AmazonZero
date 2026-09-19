"""Interchangeable opponents sharing the same staged-action interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from policy.environment import AmazonsEnv, Phase
from policy.mcts import MCTS, MCTSConfig, Evaluator, SearchNode, root_visit_policy, select_from_policy


POLICY_NAMES = ("random", "near_opponent", "self_play")


class Policy(ABC):
    """Return a distribution for one stage without changing the environment.

    Call reset() at game start and observe_action() after EVERY applied action,
    including the other player's actions, to keep stateful policies synchronized.
    """

    @abstractmethod
    def probabilities(self, env: AmazonsEnv, *, explore: bool = False) -> np.ndarray:
        ...

    def select_action(
        self, env: AmazonsEnv, *, temperature: float = 1.0, explore: bool = False
    ) -> int:
        return select_from_policy(self.probabilities(env, explore=explore), temperature)

    def reset(self) -> None:
        pass

    def observe_action(self, action: int) -> None:
        pass


class RandomPolicy(Policy):
    """Uniform over legal actions at EACH stage, not over complete triples."""

    def probabilities(self, env: AmazonsEnv, *, explore: bool = False) -> np.ndarray:
        actions = env.legal_actions()
        if not actions:
            raise ValueError("Cannot choose an action in a terminal state")
        probabilities = np.zeros(env.action_size, dtype=np.float32)
        probabilities[actions] = 1.0 / len(actions)
        return probabilities


class NearOpponentPolicy(RandomPolicy):
    """Random piece; moves and arrows prefer cells near opposing pieces.

    Distance is Chebyshev distance (eight-neighbor geometry), ignoring obstacles
    only for ranking; all candidates still come from the legal-action mask.
    Within radius: sample uniformly. If none: sample among nearest legal cells.
    random_fraction mixes in uniform exploration over all legal cells.
    """

    def __init__(self, radius: int = 1, random_fraction: float = 0.2):
        if isinstance(radius, bool) or not isinstance(radius, int) or radius < 1:
            raise ValueError("radius must be a positive integer")
        if not np.isfinite(random_fraction) or not 0 <= random_fraction <= 1:
            raise ValueError("random_fraction must be between 0 and 1")
        self.radius = radius
        self.random_fraction = random_fraction

    def probabilities(self, env: AmazonsEnv, *, explore: bool = False) -> np.ndarray:
        uniform = super().probabilities(env)
        if env.phase == Phase.SELECT_PIECE:
            return uniform
        opponent = str(3 - env.current_player)
        positions = [
            divmod(i, env.board_size)
            for i, tile in enumerate(env.game.board.game_tiles)
            if tile.to_string() == opponent
        ]
        if not positions:
            return uniform
        actions = env.legal_actions()
        distances = np.asarray([
            min(max(abs(row - r), abs(column - c)) for r, c in positions)
            for row, column in (divmod(action, env.board_size) for action in actions)
        ])
        preferred = distances <= self.radius
        if not preferred.any():
            preferred = distances == distances.min()
        candidates = np.asarray(actions)[preferred]
        probabilities = uniform * self.random_fraction
        probabilities[candidates] += (1.0 - self.random_fraction) / len(candidates)
        return probabilities


class MCTSPolicy(Policy):
    """Current network plus search, usable for either side or both sides."""

    def __init__(self, search: MCTS):
        self.search = search
        self.root: Optional[SearchNode] = None

    def probabilities(self, env: AmazonsEnv, *, explore: bool = False) -> np.ndarray:
        self.root = self.search.search(env, self.root, exploration_noise=explore)
        return root_visit_policy(self.root, env.action_size)

    def reset(self) -> None:
        self.root = None

    def observe_action(self, action: int) -> None:
        self.root = self.root.children.get(action) if self.root is not None else None


def create_policy(
    name: str,
    *,
    evaluator: Optional[Evaluator] = None,
    mcts_config: Optional[MCTSConfig] = None,
    radius: int = 1,
    random_fraction: float = 0.2,
) -> Policy:
    """Build a named policy. Only self_play requires a network evaluator."""
    if name == "random":
        return RandomPolicy()
    if name == "near_opponent":
        return NearOpponentPolicy(radius, random_fraction)
    if name == "self_play":
        if evaluator is None:
            raise ValueError("self_play requires an evaluator")
        return MCTSPolicy(MCTS(evaluator, mcts_config or MCTSConfig()))
    raise ValueError(f"Unknown policy {name!r}; choose from {POLICY_NAMES}")
