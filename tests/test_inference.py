"""Check inference snapshots, CUDA replay masks and safe training integration."""

import copy
import unittest

import numpy as np
import torch
from torch import nn

from model.inference import FrozenNetworkEvaluator
from model.model import NetworkEvaluator, PolicyValueNet
from policy.environment import AmazonsEnv


class InferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls) -> None:
        torch.set_num_threads(cls.previous_threads)

    def make_model(self) -> PolicyValueNet:
        torch.manual_seed(11)
        model = PolicyValueNet(4, channels=8, residual_blocks=2)
        # Nontrivial running stats exercise BN folding beyond identity stats.
        with torch.no_grad():
            for module in model.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.running_mean.normal_(0, 0.1)
                    module.running_var.uniform_(0.5, 1.5)
        return model

    def test_fusion_preserves_predictions_and_source_model(self) -> None:
        model = self.make_model()
        before = copy.deepcopy(model.state_dict())
        fast = FrozenNetworkEvaluator(model, torch.device("cpu"), use_amp=False)
        self.assertTrue(model.training)
        self.assertFalse(fast.model.training)
        self.assertFalse(any(isinstance(module, nn.BatchNorm2d) for module in fast.model.modules()))
        self.assertTrue(any(isinstance(module, nn.BatchNorm2d) for module in model.modules()))
        reference = NetworkEvaluator(copy.deepcopy(model), torch.device("cpu"), use_amp=False)
        env = AmazonsEnv(4)
        for _ in range(12):
            if env.is_terminal():
                break
            expected_p, expected_v = reference.evaluate(env)
            actual_p, actual_v = fast.evaluate(env)
            np.testing.assert_allclose(actual_p, expected_p, atol=1e-6)
            self.assertAlmostEqual(actual_v, expected_v, places=6)
            env.step(env.legal_actions()[0])
        for name, tensor in model.state_dict().items():
            self.assertTrue(torch.equal(tensor, before[name]), name)

    def test_snapshot_is_independent_and_rebuilding_updates_it(self) -> None:
        model = self.make_model()
        env = AmazonsEnv(4)
        old = FrozenNetworkEvaluator(model, torch.device("cpu"), use_amp=False)
        expected_p, expected_v = old.evaluate(env)
        with torch.no_grad():
            model.policy_head[-1].bias.add_(torch.arange(16, dtype=torch.float32))
            model.value_head[-2].bias.add_(1.0)
        unchanged_p, unchanged_v = old.evaluate(env)
        np.testing.assert_array_equal(unchanged_p, expected_p)
        self.assertEqual(unchanged_v, expected_v)
        fresh = FrozenNetworkEvaluator(model, torch.device("cpu"), use_amp=False)
        updated_p, updated_v = fresh.evaluate(env)
        self.assertFalse(np.allclose(updated_p, expected_p))
        self.assertNotAlmostEqual(updated_v, expected_v)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for real graph replay")
    def test_cuda_graph_replays_changing_states_and_masks_and_falls_back(self) -> None:
        device = torch.device("cuda")
        model = self.make_model().to(device)
        for amp in (False, True):
            with self.subTest(amp=amp):
                reference = NetworkEvaluator(model, device, use_amp=amp)
                fast = FrozenNetworkEvaluator(model, device, use_amp=amp)
                self.assertIsNotNone(fast.graph)
                env = AmazonsEnv(4)
                retained = []
                for _ in range(12):
                    if env.is_terminal():
                        break
                    expected_p, expected_v = reference.evaluate(env)
                    actual_p, actual_v = fast.evaluate(env)
                    tolerance = 5e-3 if amp else 1e-5
                    np.testing.assert_allclose(actual_p, expected_p, atol=tolerance)
                    self.assertAlmostEqual(actual_v, expected_v, delta=tolerance)
                    self.assertTrue((actual_p[~env.legal_mask()] == 0).all())
                    self.assertAlmostEqual(float(actual_p.sum()), 1.0, places=6)
                    retained.append((actual_p, actual_p.copy()))
                    env.step(env.legal_actions()[0])
                for actual, original in retained:
                    np.testing.assert_array_equal(actual, original)
                # Graph captured batch=1; a batch of two must use eager fallback.
                opening = AmazonsEnv(4)
                states = np.repeat(opening.encode()[None], 2, axis=0)
                masks = np.repeat(opening.legal_mask()[None], 2, axis=0)
                actual_p, actual_v = fast.evaluate_batch(states, masks)
                expected_p, expected_v = reference.evaluate_batch(states, masks)
                np.testing.assert_allclose(actual_p, expected_p, atol=tolerance)
                np.testing.assert_allclose(actual_v, expected_v, atol=tolerance)
                with self.assertRaises(ValueError):
                    fast.evaluate_batch(states[:1], np.zeros_like(masks[:1]))


if __name__ == "__main__":
    unittest.main()
