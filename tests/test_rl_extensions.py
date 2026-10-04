"""Numerical target checks and small end-to-end runs for the RL extensions."""

from contextlib import redirect_stdout
from dataclasses import replace
from io import StringIO
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np
import torch

from policy.rl_policy import LearnedRLPolicy
from train.rl_algorithms import (
    ALGORITHMS, DQNAgent, PolicyGradientAgent, RLConfig, Transition,
    create_agent, dqn_targets, generalized_advantages, masked_distribution,
    monte_carlo_returns, ppo_surrogate,
)
from train.rl_environments import AmazonsAdapter, CartPoleAdapter
from train.rl_extensions import evaluate, load_agent, run_experiment


def transition(reward: float, *, terminated: bool = False, truncated: bool = False,
               value: float = 0.0, next_value: float = 0.0) -> Transition:
    return Transition(np.zeros(4, dtype=np.float32), np.ones(2, dtype=np.bool_), 0,
        reward, np.ones(4, dtype=np.float32), np.ones(2, dtype=np.bool_),
        terminated, truncated, value=value, next_value=next_value)


class TargetTests(unittest.TestCase):
    def test_dqn_masks_illegal_actions_and_does_not_bootstrap_terminal(self) -> None:
        next_q = torch.tensor([[2.0, 999.0], [999.0, 999.0]], requires_grad=True)
        targets = dqn_targets(torch.tensor([1.0, -1.0]), torch.tensor([False, True]),
            next_q, torch.tensor([[True, False], [False, False]]), 0.9)
        torch.testing.assert_close(targets, torch.tensor([2.8, -1.0]))
        self.assertFalse(targets.requires_grad)
        self.assertTrue(torch.isfinite(targets).all())
        with self.assertRaises(ValueError):
            dqn_targets(torch.ones(1), torch.tensor([False]), next_q[:1],
                        torch.zeros(1, 2, dtype=torch.bool), 0.9)

    def test_gae_bootstraps_truncation_without_leaking_across_reset(self) -> None:
        batch = [transition(1, truncated=True, value=2, next_value=3),
                 transition(2, terminated=True, value=0.5, next_value=999)]
        advantages, returns = generalized_advantages(batch, 0.9, 0.95)
        np.testing.assert_allclose(advantages, [1.7, 1.5], rtol=1e-6)
        np.testing.assert_allclose(returns, [3.7, 2.0], rtol=1e-6)
        rollout_advantages, _ = generalized_advantages([transition(1, value=2, next_value=5)], 0.9, 0.95)
        np.testing.assert_allclose(rollout_advantages, [3.5])

    def test_mc_returns_stop_at_episode_boundaries_and_reject_partial_episodes(self) -> None:
        batch = [transition(1), transition(2, terminated=True), transition(3, truncated=True)]
        np.testing.assert_allclose(monte_carlo_returns(batch, 0.9), [2.8, 2, 3])
        with self.assertRaises(ValueError):
            monte_carlo_returns([transition(1)], 0.9)

    def test_ppo_clipping_handles_both_advantage_signs(self) -> None:
        ratios = torch.tensor([1.5, 0.5, 1.5, 0.5])
        advantages = torch.tensor([1.0, 1.0, -1.0, -1.0])
        torch.testing.assert_close(ppo_surrogate(ratios, advantages, 0.2), torch.tensor([1.2, 0.5, -1.5, -0.8]))

    def test_policy_mask_zeroes_illegal_mass_and_keeps_entropy_finite(self) -> None:
        logits = torch.tensor([[0.0, 1_000.0]], requires_grad=True)
        distribution = masked_distribution(logits, torch.tensor([[True, False]]))
        torch.testing.assert_close(distribution.probs, torch.tensor([[1.0, 0.0]]))
        distribution.entropy().sum().backward()
        self.assertTrue(torch.isfinite(logits.grad).all())


class AdapterTests(unittest.TestCase):
    def test_amazons_returns_only_learner_states_and_correct_terminal_rewards(self) -> None:
        for opponent in ("random", "near_opponent"):
            for seed in range(12):
                env = AmazonsAdapter(3, opponent)
                learner = 1 + seed % 2
                observation = env.reset(seed=seed, learner_player=learner)
                rng = np.random.default_rng(seed + 100)
                for _ in range(30):
                    self.assertEqual(env.env.current_player, learner)
                    self.assertTrue(observation.mask.any())
                    result = env.step(int(rng.choice(np.flatnonzero(observation.mask))))
                    observation = result.observation
                    if result.terminated:
                        self.assertFalse(observation.mask.any())
                        self.assertEqual(result.reward, 1 if env.env.winner == learner else -1)
                        break
                    self.assertEqual(result.reward, 0)
                else:
                    self.fail("3x3 Amazons must terminate within 30 learner stages")

    def test_cartpole_keeps_bootstrap_observation_at_time_limit(self) -> None:
        try:
            env = CartPoleAdapter()
        except RuntimeError:
            self.skipTest("Gymnasium is optional")
        try:
            env.env._max_episode_steps = 1
            env.reset(seed=7)
            result = env.step(0)
            self.assertTrue(result.truncated)
            self.assertFalse(result.terminated)
            self.assertTrue(result.observation.mask.all())
        finally:
            env.close()


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls) -> None:
        torch.set_num_threads(cls.previous_threads)

    def test_all_algorithms_train_save_reload_and_use_existing_policy_interface(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            for algorithm in ALGORITHMS:
                with self.subTest(algorithm=algorithm):
                    config = RLConfig(algorithm=algorithm, environment="amazons", seed=7,
                        total_steps=48, rollout_steps=12, batch_size=4, learning_starts=4,
                        evaluation_games=2, evaluation_interval=24, hidden_size=8)
                    torch.manual_seed(config.seed)
                    initial_agent = create_agent(72, 9, config)
                    before = [parameter.detach().clone() for parameter in initial_agent.model.parameters()]
                    output = Path(directory) / algorithm
                    with redirect_stdout(StringIO()):
                        result = run_experiment(config, output)
                    self.assertGreater(result["optimizer_steps"], 0)
                    self.assertGreaterEqual(result["actual_steps"], config.total_steps)
                    if algorithm not in ("reinforce", "reinforce_baseline"):
                        self.assertEqual(result["actual_steps"], config.total_steps)
                    self.assertTrue(all(np.isfinite(value) for value in result["last_update"].values()))
                    agent, loaded_config = load_agent(output / "model.pt")
                    self.assertTrue(any(not torch.equal(old, new) for old, new in zip(before, agent.model.parameters())))
                    self.assertEqual(config, loaded_config)
                    self.assertEqual(evaluate(agent, config), result["final"])
                    policy = LearnedRLPolicy(output / "model.pt")
                    env = AmazonsAdapter(3)
                    env.reset(seed=7)
                    probabilities = policy.probabilities(env.env)
                    self.assertAlmostEqual(float(probabilities.sum()), 1, places=6)
                    self.assertTrue((probabilities[~env.env.legal_mask()] == 0).all())
                    with self.assertRaises(FileExistsError):
                        run_experiment(config, output)

    def test_dqn_updates_only_online_gradients_and_soft_updates_target(self) -> None:
        config = RLConfig(algorithm="dqn", hidden_size=8, batch_size=4, target_tau=0.1)
        agent = DQNAgent(4, 2, config)
        before = [parameter.detach().clone() for parameter in agent.target.parameters()]
        for _ in range(4):
            agent.memory.add(transition(1, terminated=True))
        agent.update()
        self.assertTrue(all(parameter.grad is None for parameter in agent.target.parameters()))
        for old, target, online in zip(before, agent.target.parameters(), agent.model.parameters()):
            torch.testing.assert_close(target, 0.9 * old + 0.1 * online)

    def test_reinforce_does_not_train_critic_and_baseline_does(self) -> None:
        for algorithm in ("reinforce", "reinforce_baseline"):
            torch.manual_seed(7)
            config = RLConfig(algorithm=algorithm, hidden_size=8, entropy_coefficient=0)
            agent = PolicyGradientAgent(4, 2, config)
            before = [parameter.detach().clone() for parameter in agent.model.critic.parameters()]
            batch = [transition(1), transition(1, terminated=True)]
            agent.update(batch)
            changed = any(not torch.equal(old, new) for old, new in zip(before, agent.model.critic.parameters()))
            self.assertEqual(changed, algorithm == "reinforce_baseline")

    def test_evaluation_does_not_change_training_random_streams(self) -> None:
        config = RLConfig(environment="amazons", evaluation_games=2, hidden_size=8)
        agent = PolicyGradientAgent(72, 9, config)
        random.seed(3)
        np.random.seed(3)
        torch.manual_seed(3)
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        torch_state = torch.get_rng_state().clone()
        evaluate(agent, config)
        self.assertEqual(random.getstate(), python_state)
        np.testing.assert_array_equal(np.random.get_state()[1], numpy_state[1])
        torch.testing.assert_close(torch.get_rng_state(), torch_state)

    def test_invalid_config_is_rejected(self) -> None:
        for kwargs in ({"gamma": float("nan")}, {"rollout_steps": 0}, {"target_tau": 0},
                       {"evaluation_games": 3}, {"seed": -1}, {"replay_capacity": 2}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                replace(RLConfig(), **kwargs)


if __name__ == "__main__":
    unittest.main()
