import unittest

import numpy as np
import tensorflow as tf

from agent.q_math import (
    BellmanInputs,
    double_dqn_targets,
    dqn_targets,
    gather_action_values,
    huber,
    select_target_rule,
    weighted_mean,
)
from utils.device import use_cpu_only

GAMMA = 0.5


def setUpModule() -> None:
    use_cpu_only()


def bellman_inputs(rewards: list[float], dones: list[bool], online: list[list[float]], target: list[list[float]]) -> BellmanInputs:
    return BellmanInputs(
        rewards=tf.constant(rewards, tf.float32),
        dones=tf.constant(dones, tf.bool),
        next_q_online=tf.constant(online, tf.float32),
        next_q_target=tf.constant(target, tf.float32),
    )


class TargetRuleTests(unittest.TestCase):
    def test_dqn_targets_terminal_uses_only_reward(self) -> None:
        inputs = bellman_inputs([2.0, -1.0], [True, True], [[9.0, 9.0], [9.0, 9.0]], [[7.0, 8.0], [5.0, 6.0]])
        np.testing.assert_allclose(dqn_targets(inputs, GAMMA).numpy(), [2.0, -1.0])

    def test_dqn_targets_bootstraps_with_target_max(self) -> None:
        inputs = bellman_inputs([1.0, 0.0], [False, True], [[0.0, 0.0], [0.0, 0.0]], [[3.0, 4.0], [10.0, 1.0]])
        np.testing.assert_allclose(dqn_targets(inputs, GAMMA).numpy(), [1.0 + GAMMA * 4.0, 0.0])

    def test_double_dqn_uses_online_argmax_and_target_value(self) -> None:
        inputs = bellman_inputs([1.0], [False], [[1.0, 5.0, 0.0]], [[10.0, 2.0, 0.0]])
        double = double_dqn_targets(inputs, GAMMA).numpy()
        np.testing.assert_allclose(double, [1.0 + GAMMA * 2.0])
        self.assertNotAlmostEqual(float(double[0]), float(dqn_targets(inputs, GAMMA).numpy()[0]))

    def test_select_target_rule_returns_known_rules(self) -> None:
        self.assertIs(select_target_rule("dqn"), dqn_targets)
        self.assertIs(select_target_rule("double"), double_dqn_targets)

    def test_select_target_rule_rejects_unknown_name(self) -> None:
        with self.assertRaises(ValueError):
            select_target_rule("desconocida")


class LossMathTests(unittest.TestCase):
    def test_huber_quadratic_and_linear_regions(self) -> None:
        errors = tf.constant([0.5, -3.0, 1.0], tf.float32)
        np.testing.assert_allclose(huber(errors, 1.0).numpy(), [0.125, 2.5, 0.5])

    def test_weighted_mean(self) -> None:
        values = tf.constant([1.0, 3.0], tf.float32)
        self.assertAlmostEqual(float(weighted_mean(values, tf.constant([1.0, 3.0]))), 2.5)

    def test_weighted_mean_with_zero_weights_is_zero(self) -> None:
        values = tf.constant([1.0, 3.0], tf.float32)
        self.assertEqual(float(weighted_mean(values, tf.zeros(2))), 0.0)

    def test_gather_action_values_picks_each_row_action(self) -> None:
        q_values = tf.constant([[1.0, 2.0], [3.0, 4.0]], tf.float32)
        picked = gather_action_values(q_values, tf.constant([1, 0], tf.int32))
        np.testing.assert_allclose(picked.numpy(), [2.0, 3.0])


if __name__ == "__main__":
    unittest.main()
