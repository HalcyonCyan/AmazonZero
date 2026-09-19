"""PUCT search over staged Amazons actions; independent of PyTorch."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Protocol, Tuple

import numpy as np

from policy.environment import AmazonsEnv, Phase
from policy.value import perspective_value


class Evaluator(Protocol):
    def evaluate(self, env: AmazonsEnv) -> Tuple[np.ndarray, float]:
        """Return legal priors and a value from the current player's view."""
        ...


@dataclass(frozen=True)
class MCTSConfig:
    simulations: int = 40
    c_puct: float = 1.8
    dirichlet_alpha: float = 0.3
    dirichlet_fraction: float = 0.25

    def __post_init__(self) -> None:
        if self.simulations < 1:
            raise ValueError("MCTS simulations must be at least one")
        if not math.isfinite(self.c_puct) or self.c_puct < 0:
            raise ValueError("c_puct must be finite and nonnegative")
        if not math.isfinite(self.dirichlet_alpha) or self.dirichlet_alpha <= 0:
            raise ValueError("dirichlet_alpha must be finite and positive")
        if not math.isfinite(self.dirichlet_fraction) or not 0 <= self.dirichlet_fraction <= 1:
            raise ValueError("dirichlet_fraction must be between 0 and 1")


class SearchNode:
    """One staged decision node in the PUCT search tree."""

    def __init__(self, prior: float, to_play: int, phase: Phase):
        self.prior = float(prior)
        self.to_play = to_play
        self.phase = phase
        self.visit_count = 0
        self.value_sum = 0.0
        self.children: Dict[int, "SearchNode"] = {}

    @property
    def mean_value(self) -> float:
        return self.value_sum / self.visit_count if self.visit_count else 0.0

    def expand(self, env: AmazonsEnv, probabilities: np.ndarray) -> None:
        if self.children:
            return
        actions = env.legal_actions()
        child_player = (
            3 - env.current_player
            if env.phase == Phase.SELECT_ARROW
            else env.current_player
        )
        child_phase = Phase((int(env.phase) + 1) % 3)
        for action in actions:
            self.children[action] = SearchNode(
                prior=float(probabilities[action]),
                to_play=child_player,
                phase=child_phase,
            )


class MCTS:
    """Neural PUCT search with correct value handling for staged turns."""

    def __init__(self, evaluator: Evaluator, config: MCTSConfig):
        if config.simulations < 1:
            raise ValueError("MCTS simulations must be at least one")
        self.evaluator = evaluator
        self.config = config

    def search(
        self,
        env: AmazonsEnv,
        root: Optional[SearchNode] = None,
        exploration_noise: bool = False,
    ) -> SearchNode:
        if env.is_terminal():
            raise ValueError("Cannot search a terminal state")
        if (
            root is None
            or root.to_play != env.current_player
            or root.phase != env.phase
        ):
            root = SearchNode(0.0, env.current_player, env.phase)

        if not root.children:
            probabilities, _ = self.evaluator.evaluate(env)
            root.expand(env, probabilities)
        if exploration_noise:
            self._add_root_noise(root)

        for _ in range(self.config.simulations):
            simulation_env = env.clone()
            node = root
            path = [node]

            while node.children:
                action, node = self._select_child(node)
                simulation_env.step(action)
                path.append(node)
                if simulation_env.is_terminal():
                    break

            leaf_player = simulation_env.current_player
            if simulation_env.is_terminal():
                # The current player has no complete legal play.
                leaf_value = -1.0
            else:
                probabilities, leaf_value = self.evaluator.evaluate(simulation_env)
                node.expand(simulation_env, probabilities)

            for path_node in path:
                path_node.value_sum += perspective_value(
                    leaf_value,
                    source_player=leaf_player,
                    target_player=path_node.to_play,
                )
                path_node.visit_count += 1

        return root

    def _select_child(self, parent: SearchNode) -> Tuple[int, SearchNode]:
        best_score = -float("inf")
        best_pair: Optional[Tuple[int, SearchNode]] = None
        parent_scale = math.sqrt(parent.visit_count + 1)

        for action, child in parent.children.items():
            q_value = perspective_value(
                child.mean_value,
                source_player=child.to_play,
                target_player=parent.to_play,
            )
            exploration = (
                self.config.c_puct
                * child.prior
                * parent_scale
                / (1 + child.visit_count)
            )
            score = q_value + exploration
            if score > best_score:
                best_score = score
                best_pair = (action, child)

        if best_pair is None:
            raise RuntimeError("Expanded search node has no children")
        return best_pair

    def _add_root_noise(self, root: SearchNode) -> None:
        actions = list(root.children)
        if not actions or self.config.dirichlet_fraction <= 0:
            return
        noise = np.random.dirichlet(
            [self.config.dirichlet_alpha] * len(actions)
        )
        fraction = self.config.dirichlet_fraction
        for action, random_prior in zip(actions, noise):
            child = root.children[action]
            child.prior = (
                (1.0 - fraction) * child.prior
                + fraction * float(random_prior)
            )


def root_visit_policy(root: SearchNode, action_size: int) -> np.ndarray:
    policy = np.zeros(action_size, dtype=np.float32)
    for action, child in root.children.items():
        policy[action] = child.visit_count
    total = float(policy.sum())
    if total == 0:
        actions = list(root.children)
        policy[actions] = 1.0 / len(actions)
    else:
        policy /= total
    return policy


def select_from_policy(policy: np.ndarray, temperature: float) -> int:
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError("temperature must be finite and nonnegative")
    policy = np.asarray(policy, dtype=np.float64)
    if policy.ndim != 1 or not np.isfinite(policy).all() or (policy < 0).any() or not (policy > 0).any():
        raise ValueError("Policy must be a finite nonnegative vector with positive mass")
    if temperature <= 1e-8:
        return int(np.argmax(policy))
    # Subtract before scaling to avoid underflow for small temperatures.
    positive = policy > 0
    logits = np.log(policy[positive])
    adjusted = np.zeros_like(policy)
    adjusted[positive] = np.exp((logits - logits.max()) / temperature)
    adjusted /= adjusted.sum()
    return int(np.random.choice(len(policy), p=adjusted))


