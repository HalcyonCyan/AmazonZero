"""Independent six-by-six rules and larger/deeper checkpoint initialization."""

from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from model.inference import FrozenNetworkEvaluator
from model.model import ModelTrainer, NetworkEvaluator, PolicyValueNet
from policy.environment import AmazonsEnv
from policy.mcts import MCTSConfig
from train.train import TrainingConfig, run_training


def reference_plays(env: AmazonsEnv) -> dict[int, dict[int, list[int]]]:
    """Enumerate rays on a numeric board, physically moving before arrow search."""
    size = env.board_size
    board = [tile.to_string() for tile in env.game.board.game_tiles]
    directions = [(r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if (r, c) != (0, 0)]

    def rays(tiles: list[str], origin: int) -> list[int]:
        available = []
        for dr, dc in directions:
            row, column = divmod(origin, size)
            row, column = row + dr, column + dc
            while 0 <= row < size and 0 <= column < size:
                index = row * size + column
                if tiles[index] != "-":
                    break
                available.append(index)
                row, column = row + dr, column + dc
        return available

    plays = {}
    for origin, tile in enumerate(board):
        if tile != str(env.current_player):
            continue
        moves = {}
        for destination in rays(board, origin):
            moved = board.copy()
            moved[origin], moved[destination] = "-", tile
            arrows = rays(moved, destination)
            if arrows:
                moves[destination] = arrows
        if moves:
            plays[origin] = moves
    return plays


class ExpansionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls) -> None:
        torch.set_num_threads(cls.previous_threads)

    def test_six_by_six_rules_match_independent_rays_through_a_full_game(self) -> None:
        env = AmazonsEnv(6)
        self.assertEqual(env.encode().shape, (8, 6, 6))
        self.assertEqual(env.legal_actions(), [0, 2, 5])
        pieces = {player: [index for index, tile in enumerate(env.game.board.game_tiles)
                           if tile.to_string() == str(player)] for player in (1, 2)}
        self.assertEqual(pieces[2], sorted(35 - index for index in pieces[1]))
        self.assertEqual(env.initial_complete_move_count(), 460)
        rng = np.random.default_rng(17)
        turns = 0
        while True:
            expected = reference_plays(env)
            actual = env._available_plays()
            self.assertEqual(set(actual), set(expected))
            for origin, moves in expected.items():
                self.assertEqual(set(actual[origin]), set(moves))
                for destination, arrows in moves.items():
                    self.assertEqual(set(actual[origin][destination]), set(arrows))
            if env.is_terminal():
                self.assertFalse(expected)
                break
            origin = int(rng.choice(list(expected)))
            destination = int(rng.choice(list(expected[origin])))
            arrow = int(rng.choice(expected[origin][destination]))
            before = env.encode().copy()
            clone = env.clone()
            player = env.current_player
            self.assertFalse(clone.step(origin))
            self.assertEqual(clone.current_player, player)
            self.assertFalse(clone.step(destination))
            self.assertEqual(clone.current_player, player)
            self.assertTrue(clone.step(arrow))
            self.assertEqual(clone.current_player, 3 - player)
            np.testing.assert_array_equal(env.encode(), before)
            self.assertEqual(int(clone.encode()[2].sum()), turns + 1)
            env = clone
            turns += 1
            self.assertLessEqual(turns, 30)
        self.assertIn(env.winner, (1, 2))

    def test_transfer_preserves_features_heads_and_fresh_optimizer_then_learns(self) -> None:
        torch.manual_seed(7)
        source = ModelTrainer(PolicyValueNet(4, 32, 2), torch.device("cpu"), use_amp=False)
        target = ModelTrainer(PolicyValueNet(6, 32, 6), torch.device("cpu"), use_amp=False)
        heads = {key: value.clone() for key, value in target.model.state_dict().items()
                 if key.startswith(("policy_head.", "value_head."))}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.pt"
            source.save_checkpoint(path, 20)
            original = path.read_bytes()
            metadata = target.initialize_from_checkpoint(path)
            self.assertEqual(metadata["copied_residual_blocks"], 2)
            self.assertEqual(metadata["identity_residual_blocks"], 4)
            self.assertFalse(target.optimizer.state)
            for key, value in source.model.state_dict().items():
                if key.startswith(("stem.", "trunk.")):
                    self.assertTrue(torch.equal(target.model.state_dict()[key], value), key)
            for key, value in heads.items():
                self.assertTrue(torch.equal(target.model.state_dict()[key], value), key)
            env = AmazonsEnv(6)
            states = np.repeat(env.encode()[None], 4, axis=0)
            masks = np.repeat(env.legal_mask()[None], 4, axis=0)
            source.model.eval()
            target.model.eval()
            with torch.inference_mode():
                inputs = torch.from_numpy(states)
                expected = source.model.trunk(source.model.stem(inputs))
                actual = target.model.trunk(target.model.stem(inputs))
                torch.testing.assert_close(actual, expected)
                logits, values = target.model(inputs)
            self.assertEqual(logits.shape, (4, 36))
            self.assertEqual(values.shape, (4,))
            policies = masks.astype(np.float32) / masks.sum(axis=1, keepdims=True)
            metrics = target.train_batch(states, policies.astype(np.float32), masks, np.ones(4, np.float32))
            self.assertTrue(all(np.isfinite(value) for value in metrics.values()))
            self.assertTrue(torch.any(target.model.trunk[-1].bn2.weight != 0))
            self.assertEqual(path.read_bytes(), original)
            with self.assertRaisesRegex(ValueError, "before any optimizer"):
                target.initialize_from_checkpoint(path)
            incompatible = ModelTrainer(PolicyValueNet(6, 8, 6), torch.device("cpu"), use_amp=False)
            with self.assertRaisesRegex(ValueError, "channel"):
                incompatible.initialize_from_checkpoint(path)
            with self.assertRaisesRegex(ValueError, "configuration"):
                target.load_checkpoint(path)

    def test_new_six_by_six_training_starts_at_one_and_records_initialization(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source_path = Path(directory) / "source.pt"
            target_path = Path(directory) / "six.pt"
            source = ModelTrainer(PolicyValueNet(4, 4, 2), torch.device("cpu"), use_amp=False)
            source.save_checkpoint(source_path, 20)
            original = source_path.read_bytes()
            trainer = ModelTrainer(PolicyValueNet(6, 4, 6), torch.device("cpu"), use_amp=False)
            config = TrainingConfig(board_size=6, iterations=1, games_per_iteration=1,
                                    train_batches=1, batch_size=8, evaluation_games=2,
                                    checkpoint=str(target_path), fast_inference=True)
            with redirect_stdout(StringIO()):
                run_training(config, MCTSConfig(simulations=2), trainer, initialize_from=source_path)
            checkpoint = trainer.load_checkpoint(target_path)
            self.assertEqual(checkpoint["iteration"], 1)
            self.assertEqual(checkpoint["model_config"], {"board_size": 6, "channels": 4, "residual_blocks": 6})
            self.assertEqual(checkpoint["extra"]["initialization"]["source_iteration"], 20)
            self.assertEqual(source_path.read_bytes(), original)
            with self.assertRaisesRegex(ValueError, "combined"):
                run_training(config, MCTSConfig(simulations=2), trainer, resume=True, initialize_from=source_path)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required for six-block graph replay")
    def test_six_by_six_deep_cuda_graph(self) -> None:
        device = torch.device("cuda")
        model = PolicyValueNet(6, 32, 6).to(device)
        reference = NetworkEvaluator(model, device)
        fast = FrozenNetworkEvaluator(model, device)
        env = AmazonsEnv(6)
        for _ in range(6):
            expected_p, expected_v = reference.evaluate(env)
            actual_p, actual_v = fast.evaluate(env)
            np.testing.assert_allclose(actual_p, expected_p, atol=5e-3)
            self.assertAlmostEqual(actual_v, expected_v, delta=5e-3)
            self.assertTrue((actual_p[~env.legal_mask()] == 0).all())
            self.assertAlmostEqual(float(actual_p.sum()), 1.0, places=6)
            env.step(env.legal_actions()[0])


if __name__ == "__main__":
    unittest.main()
