"""Single-agent adapters for comparing RL targets on CartPole and Amazons."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from policy.environment import AmazonsEnv
from policy.strategies import create_policy


@dataclass(frozen=True)
class Observation:
    state: np.ndarray
    mask: np.ndarray


@dataclass(frozen=True)
class StepResult:
    observation: Observation
    reward: float
    terminated: bool
    truncated: bool = False


class DiscreteEnvironment(Protocol):
    observation_size: int
    action_size: int

    def reset(self, *, seed: int, learner_player: int | None = None) -> Observation: ...
    def step(self, action: int) -> StepResult: ...
    def close(self) -> None: ...


class CartPoleAdapter:
    """Gymnasium is optional when running only the Amazons experiments."""

    observation_size = 4
    action_size = 2

    def __init__(self) -> None:
        try:
            import gymnasium as gym
        except ImportError as error:
            raise RuntimeError("Install requirements-rl.txt to run CartPole") from error
        self.env = gym.make("CartPole-v1")

    def reset(self, *, seed: int, learner_player: int | None = None) -> Observation:
        state, _ = self.env.reset(seed=seed)
        return Observation(np.asarray(state, dtype=np.float32), np.ones(2, dtype=np.bool_))

    def step(self, action: int) -> StepResult:
        state, reward, terminated, truncated, _ = self.env.step(action)
        mask = np.zeros(2, dtype=np.bool_) if terminated else np.ones(2, dtype=np.bool_)
        return StepResult(Observation(np.asarray(state, dtype=np.float32), mask),
                          float(reward), bool(terminated), bool(truncated))

    def close(self) -> None:
        self.env.close()


class AmazonsAdapter:
    """One step is one learner stage; opponent turns run inside the adapter.

    Nonterminal observations always belong to the learner. This makes ordinary
    single-agent Bellman/GAE formulas valid without negamax sign conversions.
    Discounting is per learner stage, including the automatic opponent response.
    """

    def __init__(self, board_size: int = 3, opponent: str = "random") -> None:
        if opponent not in ("random", "near_opponent"):
            raise ValueError("The single-agent adapter requires a fixed heuristic opponent")
        self.board_size = board_size
        self.observation_size = 8 * board_size**2
        self.action_size = board_size**2
        self.opponent = create_policy(opponent)
        self.env = AmazonsEnv(board_size)
        self.rng = np.random.default_rng(0)
        self.learner_player = 1

    def _observation(self) -> Observation:
        return Observation(self.env.encode().reshape(-1), self.env.legal_mask())

    def _opponent_turn(self) -> None:
        while not self.env.is_terminal() and self.env.current_player != self.learner_player:
            action = int(self.rng.choice(self.action_size, p=self.opponent.probabilities(self.env)))
            self.env.step(action)
            self.opponent.observe_action(action)

    def reset(self, *, seed: int, learner_player: int | None = None) -> Observation:
        if learner_player not in (None, 1, 2):
            raise ValueError("learner_player must be 1 or 2")
        self.rng = np.random.default_rng(seed)
        self.learner_player = int(self.rng.integers(1, 3)) if learner_player is None else learner_player
        self.env = AmazonsEnv(self.board_size)
        self.opponent.reset()
        self._opponent_turn()
        if self.env.is_terminal():
            raise RuntimeError("Opponent ended the game before the learner could act")
        return self._observation()

    def step(self, action: int) -> StepResult:
        if self.env.is_terminal():
            raise RuntimeError("Reset the adapter before stepping a finished game")
        self.env.step(action)
        self.opponent.observe_action(action)
        self._opponent_turn()
        terminated = self.env.is_terminal()
        reward = float(1 if self.env.winner == self.learner_player else -1) if terminated else 0.0
        return StepResult(self._observation(), reward, terminated)

    def close(self) -> None:
        pass


def make_environment(name: str, board_size: int = 3, opponent: str = "random") -> DiscreteEnvironment:
    if name == "cartpole":
        return CartPoleAdapter()
    if name == "amazons":
        return AmazonsAdapter(board_size, opponent)
    raise ValueError(f"Unknown environment {name!r}")
