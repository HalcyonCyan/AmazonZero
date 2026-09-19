"""Train against configurable heuristic opponents or the current network."""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

from policy.environment import AmazonsEnv, run_environment_smoke_tests
# Re-export the original search symbols for callers of train.train.
from policy.mcts import MCTS, MCTSConfig, SearchNode, root_visit_policy, select_from_policy
from policy.match import MatchRunner, TrainingExample, evaluate_policies
from policy.opponents import OpponentSchedule, TRAINING_MODES
from policy.strategies import MCTSPolicy, NearOpponentPolicy, RandomPolicy, create_policy
from model.model import (
    ModelTrainer,
    NetworkEvaluator,
    PolicyValueNet,
    resolve_device,
)


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
    """Compatibility wrapper around the shared match runner."""

    def __init__(self, board_size: int, search: MCTS, exploration_turns: int = 8):
        self.board_size = board_size
        self.search = search
        self.exploration_turns = exploration_turns

    def play_game(self) -> Tuple[List[TrainingExample], int]:
        policy = MCTSPolicy(self.search)
        return MatchRunner(self.board_size, self.exploration_turns).play_game(
            policy, policy, collect_players=frozenset({1, 2}), training=True
        )


def evaluate_against_random(board_size: int, search: MCTS, games: int) -> float:
    """Compatibility wrapper for the original random-opponent evaluation."""
    return evaluate_policies(board_size, MCTSPolicy(search), RandomPolicy(), games)


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
    opponent: str = "self_play"
    curriculum_stage_iterations: int = 5
    near_radius: int = 1
    near_random_fraction: float = 0.2
    exploration_turns: int = 8

    def __post_init__(self) -> None:
        if self.board_size not in (3, 4, 5, 10):
            raise ValueError("board_size must be 3, 4, 5 or 10")
        for name in (
            "iterations", "games_per_iteration", "train_batches", "batch_size",
            "replay_capacity", "evaluation_games",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        OpponentSchedule(self.opponent, self.curriculum_stage_iterations)
        NearOpponentPolicy(self.near_radius, self.near_random_fraction)
        if self.exploration_turns < 0:
            raise ValueError("exploration_turns must be nonnegative")


def run_training(
    training_config: TrainingConfig,
    mcts_config: MCTSConfig,
    trainer: ModelTrainer,
    resume: bool = False,
) -> None:
    if trainer.model.board_size != training_config.board_size:
        raise ValueError("Training board size must match the model board size")
    checkpoint_path = Path(training_config.checkpoint)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    start_iteration = 1
    if resume:
        checkpoint = trainer.load_checkpoint(checkpoint_path)
        start_iteration = int(checkpoint["iteration"]) + 1
        print(f"Resumed checkpoint at iteration {start_iteration - 1}")

    evaluator = NetworkEvaluator(trainer.model, trainer.device, trainer.use_amp)
    search = MCTS(evaluator, mcts_config)
    learner = MCTSPolicy(search)
    runner = MatchRunner(training_config.board_size, training_config.exploration_turns)
    schedule = OpponentSchedule(
        training_config.opponent, training_config.curriculum_stage_iterations
    )
    near_opponent = NearOpponentPolicy(
        training_config.near_radius, training_config.near_random_fraction
    )
    replay = ReplayBuffer(training_config.replay_capacity)

    for iteration in range(
        start_iteration, training_config.iterations + 1
    ):
        winners = {1: 0, 2: 0}
        generated_states = 0
        opponent_counts: Dict[str, int] = {}
        for game_index in range(training_config.games_per_iteration):
            opponent_name = schedule.choose(iteration)
            opponent_counts[opponent_name] = opponent_counts.get(opponent_name, 0) + 1
            if opponent_name == "self_play":
                examples, winner = runner.play_game(
                    learner, learner, collect_players=frozenset({1, 2}), training=True
                )
            else:
                opponent = create_policy(
                    opponent_name,
                    radius=training_config.near_radius,
                    random_fraction=training_config.near_random_fraction,
                )
                # Alternate across iteration boundaries even with one game per iteration.
                learner_player = 1 + (
                    (iteration - 1) * training_config.games_per_iteration + game_index
                ) % 2
                players = (learner, opponent) if learner_player == 1 else (opponent, learner)
                examples, winner = runner.play_game(
                    *players, collect_players=frozenset({learner_player}), training=True
                )
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
        near_win_rate = evaluate_policies(
            training_config.board_size, learner, near_opponent, training_config.evaluation_games
        )
        means = {name: float(np.mean(values)) for name, values in metrics.items()}
        print(
            f"iteration={iteration:03d} generated={generated_states:5d} "
            f"buffer={len(replay):6d} winners={winners} "
            f"loss={means['loss']:.4f} "
            f"policy={means['policy_loss']:.4f} "
            f"value={means['value_loss']:.4f} "
            f"opponents={opponent_counts} "
            f"vs_random={random_win_rate:.1%} vs_near={near_win_rate:.1%}"
        )
        trainer.save_checkpoint(
            checkpoint_path,
            iteration,
            extra={
                "random_win_rate": random_win_rate,
                "near_opponent_win_rate": near_win_rate,
                "opponent_counts": opponent_counts,
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
    parser.add_argument("--opponent", choices=TRAINING_MODES, default="self_play",
                        help="Training opponent or scheduling mode (default: self_play)")
    parser.add_argument("--curriculum-stage-iterations", type=int, default=5,
                        help="Iterations each for random and near_opponent, then self_play")
    parser.add_argument("--near-radius", type=int, default=1)
    parser.add_argument("--near-random-fraction", type=float, default=0.2,
                        help="Uniform exploration fraction for near_opponent, from 0 to 1")
    parser.add_argument("--exploration-turns", type=int, default=8)
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
        opponent=args.opponent,
        curriculum_stage_iterations=args.curriculum_stage_iterations,
        near_radius=args.near_radius,
        near_random_fraction=args.near_random_fraction,
        exploration_turns=args.exploration_turns,
    )
    mcts_config = MCTSConfig(simulations=args.simulations)
    run_training(training_config, mcts_config, trainer, resume=args.resume)


if __name__ == "__main__":
    main()
