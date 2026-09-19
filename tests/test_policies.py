"""Regression checks for legal policies, opponent games and training schedules."""

import random
import unittest

import numpy as np

from policy import (
    AmazonsEnv, MatchRunner, MCTS, MCTSConfig, MCTSPolicy, NearOpponentPolicy,
    OpponentSchedule, RandomPolicy, create_policy, evaluate_policies,
)
from policy.mcts import select_from_policy
from policy.value import perspective_value
from rule.game_backend import Flame


class UniformEvaluator:
    def evaluate(self, env: AmazonsEnv) -> tuple[np.ndarray, float]:
        return RandomPolicy().probabilities(env), 0.0


class RecordingPolicy(RandomPolicy):
    def __init__(self) -> None:
        self.events: list[tuple[int, bool]] = []
        self.observed: list[int] = []

    def reset(self) -> None:
        self.events.clear()
        self.observed.clear()

    def probabilities(self, env: AmazonsEnv, *, explore: bool = False) -> np.ndarray:
        self.events.append((env.current_player, explore))
        return super().probabilities(env)

    def observe_action(self, action: int) -> None:
        self.observed.append(action)


class PolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        random.seed(7)
        np.random.seed(7)

    def test_heuristics_remain_legal_through_full_games(self) -> None:
        for size in (3, 4, 5, 10):
            env = AmazonsEnv(size)
            policies = (RandomPolicy(), NearOpponentPolicy())
            steps = 0
            while not env.is_terminal():
                policy = policies[env.current_player - 1]
                before = env.encode().copy()
                probabilities = policy.probabilities(env)
                np.testing.assert_array_equal(before, env.encode())
                self.assertTrue(np.isclose(probabilities.sum(), 1.0))
                self.assertTrue((probabilities[~env.legal_mask()] == 0).all())
                env.step(select_from_policy(probabilities, 1.0))
                steps += 1
                self.assertLessEqual(steps, 3 * size * size)
            self.assertIn(env.winner, (1, 2))
            with self.assertRaises(ValueError):
                policies[0].select_action(env)

    def test_near_opponent_move_and_arrow(self) -> None:
        env = AmazonsEnv(3)
        policy = NearOpponentPolicy(random_fraction=0)
        env.step(0)
        self.assertEqual(np.flatnonzero(policy.probabilities(env)).tolist(), [4])
        env.step(4)
        self.assertEqual(np.flatnonzero(policy.probabilities(env)).tolist(), [5, 7])

    def test_nearest_fallback_with_obstructed_board(self) -> None:
        env = AmazonsEnv(5)
        for cell in range(10, 15):
            env.game.board.game_tiles[cell] = Flame(0, cell)
        env.step(0)
        probabilities = NearOpponentPolicy(random_fraction=0).probabilities(env)
        self.assertEqual(np.flatnonzero(probabilities).tolist(), [5, 6])

    def test_uniform_exploration_and_radius(self) -> None:
        env = AmazonsEnv(3)
        env.step(0)
        uniform = RandomPolicy().probabilities(env)
        np.testing.assert_allclose(NearOpponentPolicy(random_fraction=1).probabilities(env), uniform)
        np.testing.assert_allclose(NearOpponentPolicy(radius=2, random_fraction=0).probabilities(env), uniform)
        mixed = NearOpponentPolicy(random_fraction=0.2).probabilities(env)
        self.assertTrue((mixed[env.legal_mask()] > 0).all())
        self.assertGreater(mixed[4], mixed[1])

    def test_match_collects_only_learner_and_labels_its_perspective(self) -> None:
        for learner_player in (1, 2):
            players = (RecordingPolicy(), RecordingPolicy())
            examples, winner = MatchRunner(3).play_game(
                *players, collect_players=frozenset({learner_player}), training=True
            )
            self.assertTrue(examples)
            self.assertTrue(all(item.player == learner_player for item in examples))
            self.assertTrue(all(item.value == (1 if winner == learner_player else -1) for item in examples))
            self.assertTrue(all(explore for _, explore in players[learner_player - 1].events))
            self.assertFalse(any(explore for _, explore in players[2 - learner_player].events))
            self.assertEqual(players[0].observed, players[1].observed)

    def test_shared_self_play_observes_actions_once_and_resets(self) -> None:
        policy = RecordingPolicy()
        for _ in range(2):
            examples, winner = MatchRunner(3).play_game(
                policy, policy, collect_players=frozenset({1, 2}), training=True
            )
            self.assertEqual(len(policy.observed), len(examples))
            self.assertEqual({item.player for item in examples}, {1, 2})
            for item in examples:
                self.assertEqual(item.value, 1 if item.player == winner else -1)

    def test_mcts_tree_follows_both_players_actions(self) -> None:
        policy = MCTSPolicy(MCTS(UniformEvaluator(), MCTSConfig(simulations=4)))
        env = AmazonsEnv(3)
        while not env.is_terminal():
            if env.current_player == 1:
                probabilities = policy.probabilities(env)
            else:
                probabilities = RandomPolicy().probabilities(env)
            action = select_from_policy(probabilities, 1.0)
            expected = policy.root.children.get(action) if policy.root else None
            env.step(action)
            policy.observe_action(action)
            self.assertIs(policy.root, expected)
            if policy.root:
                self.assertEqual(policy.root.to_play, env.current_player)
                self.assertEqual(policy.root.phase, env.phase)
        policy.reset()
        self.assertIsNone(policy.root)

    def test_evaluation_alternates_colors_without_noise(self) -> None:
        class ColorRecorder(RecordingPolicy):
            def reset(self) -> None:
                pass

        learner = ColorRecorder()
        win_rate = evaluate_policies(3, learner, RandomPolicy(), 2)
        self.assertTrue(0 <= win_rate <= 1)
        self.assertEqual({player for player, _ in learner.events}, {1, 2})
        self.assertFalse(any(explore for _, explore in learner.events))
        with self.assertRaises(ValueError):
            evaluate_policies(3, learner, RandomPolicy(), 0)

    def test_factory_and_configuration_errors(self) -> None:
        self.assertIsInstance(create_policy("random"), RandomPolicy)
        self.assertIsInstance(create_policy("near_opponent"), NearOpponentPolicy)
        self.assertIsInstance(create_policy("self_play", evaluator=UniformEvaluator()), MCTSPolicy)
        for name in ("unknown", "self_play"):
            with self.assertRaises(ValueError):
                create_policy(name)
        for kwargs in ({"radius": 0}, {"radius": 1.5}, {"random_fraction": -0.1}, {"random_fraction": float("nan")}):
            with self.assertRaises(ValueError):
                NearOpponentPolicy(**kwargs)

    def test_schedule_fixed_mixed_and_absolute_curriculum(self) -> None:
        schedule = OpponentSchedule("curriculum", stage_iterations=2)
        self.assertEqual([schedule.choose(i) for i in range(1, 8)], [
            "random", "random", "near_opponent", "near_opponent", "self_play", "self_play", "self_play"
        ])
        self.assertEqual(OpponentSchedule("near_opponent").choose(100), "near_opponent")
        self.assertEqual({OpponentSchedule("mixed").choose(1) for _ in range(100)}, {
            "random", "near_opponent", "self_play"
        })
        with self.assertRaises(ValueError):
            OpponentSchedule("curriculum", 0)

    def test_small_temperature_and_perspective(self) -> None:
        self.assertEqual(select_from_policy(np.array([0, 0.4, 0.6]), 0.0001), 2)
        self.assertEqual(perspective_value(0.5, 1, 1), 0.5)
        self.assertEqual(perspective_value(0.5, 1, 2), -0.5)

    def test_invalid_custom_policy_is_rejected(self) -> None:
        class InvalidPolicy(RandomPolicy):
            def probabilities(self, env: AmazonsEnv, *, explore: bool = False) -> np.ndarray:
                return np.ones(env.action_size, dtype=np.float32) / env.action_size

        with self.assertRaisesRegex(ValueError, "legal probability"):
            MatchRunner(3).play_game(InvalidPolicy(), RandomPolicy())


if __name__ == "__main__":
    unittest.main()
