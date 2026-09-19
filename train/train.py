
from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from policy.environment import AmazonsEnv, Phase, run_environment_smoke_tests
from model.model import (
    ModelTrainer,
    NetworkEvaluator,
    PolicyValueNet,
    perspective_value,
    resolve_device,
)


@dataclass(frozen=True)
class MCTSConfig:
    simulations: int = 40
    c_puct: float = 1.8
    dirichlet_alpha: float = 0.3
    dirichlet_fraction: float = 0.25


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

    def __init__(self, evaluator: NetworkEvaluator, config: MCTSConfig):
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
    if temperature <= 1e-8:
        return int(np.argmax(policy))
    adjusted = np.power(policy, 1.0 / temperature)
    adjusted_sum = float(adjusted.sum())
    if adjusted_sum <= 0:
        raise RuntimeError("Policy has no positive probability")
    adjusted /= adjusted_sum
    return int(np.random.choice(len(policy), p=adjusted))


@dataclass
class TrainingExample:
    state: np.ndarray
    policy: np.ndarray
    legal_mask: np.ndarray
    player: int
    value: float = 0.0


class ReplayBuffer:
    """Fixed-size replay buffer of (state, MCTS policy, result) examples."""

    def __init__(self, capacity: int):
        if capacity < 1:
            raise ValueError("Replay capacity must be positive")
        self.capacity = capacity
        self._data: List[TrainingExample] = []
        self._next_index = 0

    def __len__(self) -> int:
        return len(self._data)

    def extend(self, examples: Sequence[TrainingExample]) -> None:
        for example in examples:
            if len(self._data) < self.capacity:
                self._data.append(example)
            else:
                self._data[self._next_index] = example
            self._next_index = (self._next_index + 1) % self.capacity

    def sample_arrays(
        self, batch_size: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if not self._data:
            raise RuntimeError("Cannot sample an empty replay buffer")
        indices = random.sample(range(len(self._data)), min(batch_size, len(self._data)))
        batch = [self._data[index] for index in indices]
        return (
            np.stack([item.state for item in batch]),
            np.stack([item.policy for item in batch]),
            np.stack([item.legal_mask for item in batch]),
            np.asarray([item.value for item in batch], dtype=np.float32),
        )


class SelfPlayRunner:
    def __init__(
        self,
        board_size: int,
        search: MCTS,
        exploration_turns: int = 8,
    ):
        self.board_size = board_size
        self.search = search
        self.exploration_turns = exploration_turns

    def play_game(self) -> Tuple[List[TrainingExample], int]:
        env = AmazonsEnv(self.board_size)
        history: List[TrainingExample] = []
        root: Optional[SearchNode] = None

        while not env.is_terminal():
            root = self.search.search(env, root, exploration_noise=True)
            policy = root_visit_policy(root, env.action_size)
            history.append(
                TrainingExample(
                    state=env.encode(),
                    policy=policy,
                    legal_mask=env.legal_mask(),
                    player=env.current_player,
                )
            )

            temperature = 1.0 if env.completed_turns < self.exploration_turns else 0.0
            action = select_from_policy(policy, temperature)
            next_root = root.children.get(action)
            env.step(action)
            # Reuse the searched child subtree at the next staged decision.
            root = next_root

        assert env.winner is not None
        winner = env.winner
        for example in history:
            example.value = perspective_value(
                1.0, source_player=winner, target_player=example.player
            )
        return history, winner


def evaluate_against_random(
    board_size: int,
    search: MCTS,
    games: int,
) -> float:
    """Evaluate deterministically, alternating the model's color."""
    model_wins = 0
    for game_index in range(games):
        env = AmazonsEnv(board_size)
        model_player = 1 if game_index % 2 == 0 else 2
        root: Optional[SearchNode] = None

        while not env.is_terminal():
            if env.current_player == model_player:
                root = search.search(env, root, exploration_noise=False)
                policy = root_visit_policy(root, env.action_size)
                action = select_from_policy(policy, temperature=0.0)
                next_root = root.children.get(action)
            else:
                action = random.choice(env.legal_actions())
                next_root = None
            env.step(action)
            root = next_root

        model_wins += int(env.winner == model_player)
    return model_wins / games


@dataclass(frozen=True)
class TrainingConfig:
    board_size: int = 4
    iterations: int = 5
    games_per_iteration: int = 8
    train_batches: int = 40
    batch_size: int = 64
    replay_capacity: int = 50_000
    evaluation_games: int = 4
    checkpoint: str = "checkpoints/amazons_latest.pt"


def run_training(
    training_config: TrainingConfig,
    mcts_config: MCTSConfig,
    trainer: ModelTrainer,
    resume: bool = False,
) -> None:
    checkpoint_path = Path(training_config.checkpoint)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    start_iteration = 1
    if resume:
        checkpoint = trainer.load_checkpoint(checkpoint_path)
        start_iteration = int(checkpoint["iteration"]) + 1
        print(f"Resumed checkpoint at iteration {start_iteration - 1}")

    evaluator = NetworkEvaluator(trainer.model, trainer.device, trainer.use_amp)
    search = MCTS(evaluator, mcts_config)
    self_play = SelfPlayRunner(training_config.board_size, search)
    replay = ReplayBuffer(training_config.replay_capacity)

    for iteration in range(
        start_iteration, training_config.iterations + 1
    ):
        winners = {1: 0, 2: 0}
        generated_states = 0
        for _ in range(training_config.games_per_iteration):
            examples, winner = self_play.play_game()
            replay.extend(examples)
            generated_states += len(examples)
            winners[winner] += 1

        metrics: Dict[str, List[float]] = {
            "loss": [],
            "policy_loss": [],
            "value_loss": [],
            "gradient_norm": [],
        }
        for _ in range(training_config.train_batches):
            arrays = replay.sample_arrays(training_config.batch_size)
            batch_metrics = trainer.train_batch(*arrays)
            for name, value in batch_metrics.items():
                metrics[name].append(value)

        random_win_rate = evaluate_against_random(
            training_config.board_size,
            search,
            training_config.evaluation_games,
        )
        means = {name: float(np.mean(values)) for name, values in metrics.items()}
        print(
            f"iteration={iteration:03d} generated={generated_states:5d} "
            f"buffer={len(replay):6d} winners={winners} "
            f"loss={means['loss']:.4f} "
            f"policy={means['policy_loss']:.4f} "
            f"value={means['value_loss']:.4f} "
            f"vs_random={random_win_rate:.1%}"
        )
        trainer.save_checkpoint(
            checkpoint_path,
            iteration,
            extra={
                "random_win_rate": random_win_rate,
                "metrics": means,
                "mcts_config": vars(mcts_config),
                "training_config": vars(training_config),
            },
        )


def run_full_smoke_test(
    model: PolicyValueNet,
    device: torch.device,
) -> None:
    run_environment_smoke_tests()
    env = AmazonsEnv(model.board_size)
    evaluator = NetworkEvaluator(model.to(device), device, use_amp=False)
    probabilities, value = evaluator.evaluate(env)
    assert probabilities.shape == (env.action_size,)
    assert np.isclose(probabilities.sum(), 1.0)
    assert np.all(probabilities[~env.legal_mask()] == 0)
    assert -1.0 <= value <= 1.0
    search = MCTS(evaluator, MCTSConfig(simulations=2))
    root = search.search(env)
    assert root.visit_count == 2
    print("Network, legal mask, value and two MCTS simulations: passed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--board-size", type=int, default=4, choices=[3, 4, 5, 10])
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--games-per-iteration", type=int, default=8)
    parser.add_argument("--simulations", type=int, default=40)
    parser.add_argument("--train-batches", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--replay-capacity", type=int, default=50_000)
    parser.add_argument("--evaluation-games", type=int, default=4)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--residual-blocks", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--checkpoint", default="checkpoints/amazons_latest.pt")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        print(f"Device: {device} ({torch.cuda.get_device_name(device)})")
    else:
        print(f"Device: {device}")

    model = PolicyValueNet(
        board_size=args.board_size,
        channels=args.channels,
        residual_blocks=args.residual_blocks,
    )
    if args.smoke_test:
        run_full_smoke_test(model, device)
        return

    trainer = ModelTrainer(
        model,
        device,
        learning_rate=args.learning_rate,
        use_amp=not args.no_amp,
    )
    training_config = TrainingConfig(
        board_size=args.board_size,
        iterations=args.iterations,
        games_per_iteration=args.games_per_iteration,
        train_batches=args.train_batches,
        batch_size=args.batch_size,
        replay_capacity=args.replay_capacity,
        evaluation_games=args.evaluation_games,
        checkpoint=args.checkpoint,
    )
    mcts_config = MCTSConfig(simulations=args.simulations)
    run_training(training_config, mcts_config, trainer, resume=args.resume)


if __name__ == "__main__":
    main()
