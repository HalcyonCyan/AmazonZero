"""Small CPU integration runs covering opponent modes and checkpoint resume."""

from contextlib import redirect_stdout
from dataclasses import replace
from io import StringIO
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np
import torch

from model.model import ModelTrainer, NetworkEvaluator, PolicyValueNet
from policy.environment import AmazonsEnv
from policy import MCTSConfig, TRAINING_MODES
from train.train import TrainingConfig, run_training


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls) -> None:
        torch.set_num_threads(cls.previous_threads)

    def make_trainer(self) -> ModelTrainer:
        random.seed(3)
        np.random.seed(3)
        torch.manual_seed(3)
        return ModelTrainer(
            PolicyValueNet(3, channels=4, residual_blocks=1), torch.device("cpu"), use_amp=False
        )

    def config(self, path: Path, mode: str, iterations: int = 1) -> TrainingConfig:
        return TrainingConfig(
            board_size=3, opponent=mode, iterations=iterations,
            games_per_iteration=2, train_batches=1, batch_size=8,
            replay_capacity=128, evaluation_games=2, checkpoint=str(path),
            curriculum_stage_iterations=1,
        )

    def test_all_modes_update_weights_and_save_usable_checkpoints(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            for mode in TRAINING_MODES:
                with self.subTest(mode=mode):
                    trainer = self.make_trainer()
                    before = trainer.model.stem[0].weight.detach().clone()
                    path = Path(directory) / f"{mode}.pt"
                    config = self.config(path, mode, 3 if mode == "curriculum" else 1)
                    with redirect_stdout(StringIO()):
                        run_training(config, MCTSConfig(simulations=2), trainer)
                    self.assertFalse(torch.equal(before, trainer.model.stem[0].weight))
                    checkpoint = trainer.load_checkpoint(path)
                    self.assertEqual(checkpoint["iteration"], config.iterations)
                    extra = checkpoint["extra"]
                    self.assertEqual(extra["training_config"]["opponent"], mode)
                    self.assertEqual(sum(extra["opponent_counts"].values()), 2)
                    if mode == "curriculum":
                        self.assertEqual(extra["opponent_counts"], {"self_play": 2})
                    for name in ("random_win_rate", "near_opponent_win_rate"):
                        self.assertTrue(0 <= extra[name] <= 1)
                    self.assertTrue(all(np.isfinite(value) for value in extra["metrics"].values()))
                    self.assertTrue(checkpoint["optimizer_state"]["state"])

    def test_resume_continues_curriculum_at_absolute_iteration(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "resume.pt"
            config = self.config(path, "curriculum")
            with redirect_stdout(StringIO()):
                run_training(config, MCTSConfig(simulations=2), self.make_trainer())
            trainer = self.make_trainer()
            checkpoint = trainer.load_checkpoint(path)
            self.assertEqual(checkpoint["extra"]["opponent_counts"], {"random": 2})
            with redirect_stdout(StringIO()):
                run_training(replace(config, iterations=2), MCTSConfig(simulations=2), trainer, resume=True)
            checkpoint = trainer.load_checkpoint(path)
            self.assertEqual(checkpoint["iteration"], 2)
            self.assertEqual(checkpoint["extra"]["opponent_counts"], {"near_opponent": 2})

    def test_rejects_invalid_training_before_writing_checkpoint(self) -> None:
        for kwargs in ({"evaluation_games": 0}, {"games_per_iteration": 0}, {"opponent": "invalid"}):
            with self.assertRaises(ValueError):
                TrainingConfig(**kwargs)
        with self.assertRaisesRegex(ValueError, "board size"):
            run_training(TrainingConfig(board_size=4), MCTSConfig(simulations=2), self.make_trainer())

    def test_half_precision_outputs_keep_mask_and_losses_finite(self) -> None:
        class HalfOutputNet(PolicyValueNet):
            def forward(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                logits, values = super().forward(states)
                return logits.half(), values.half()

        env = AmazonsEnv(3)
        model = HalfOutputNet(3, channels=4, residual_blocks=1)
        trainer = ModelTrainer(model, torch.device("cpu"), use_amp=False)
        evaluator = NetworkEvaluator(model, trainer.device, use_amp=False)
        states = np.repeat(env.encode()[None], 4, axis=0)
        masks = np.repeat(env.legal_mask()[None], 4, axis=0)
        policies, values = evaluator.evaluate_batch(states, masks)
        self.assertTrue(np.isfinite(policies).all() and np.isfinite(values).all())
        self.assertTrue(np.all(policies[~masks] == 0))
        np.testing.assert_allclose(policies.sum(axis=1), 1.0, atol=1e-6)
        before = model.stem[0].weight.detach().clone()
        metrics = trainer.train_batch(states, policies, masks, np.ones(4, dtype=np.float32))
        self.assertTrue(all(np.isfinite(value) for value in metrics.values()))
        self.assertFalse(torch.equal(before, model.stem[0].weight))


if __name__ == "__main__":
    unittest.main()
