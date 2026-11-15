import unittest

import numpy as np

from agent.network import build_optimizer, build_q_network, copy_weights
from utils.device import use_cpu_only

STATE_SIZE = 11
ACTION_SIZE = 4
HIDDEN_UNITS = (16, 8)


def setUpModule() -> None:
    use_cpu_only()


class NetworkTests(unittest.TestCase):
    def test_build_q_network_output_shape(self) -> None:
        network = build_q_network(STATE_SIZE, ACTION_SIZE, HIDDEN_UNITS)
        for rows in (1, 5, 64):
            output = network(np.zeros((rows, STATE_SIZE), dtype=np.float32))
            self.assertEqual(tuple(output.shape), (rows, ACTION_SIZE))

    def test_build_q_network_has_one_dense_per_hidden_unit_plus_output(self) -> None:
        network = build_q_network(STATE_SIZE, ACTION_SIZE, HIDDEN_UNITS)
        self.assertEqual(len(network.trainable_variables), 2 * (len(HIDDEN_UNITS) + 1))

    def test_copy_weights_makes_outputs_equal(self) -> None:
        source = build_q_network(STATE_SIZE, ACTION_SIZE, HIDDEN_UNITS)
        target = build_q_network(STATE_SIZE, ACTION_SIZE, HIDDEN_UNITS)
        states = np.random.default_rng(0).random((4, STATE_SIZE), dtype=np.float32)
        copy_weights(source, target)
        np.testing.assert_allclose(source(states).numpy(), target(states).numpy())

    def test_copy_weights_rejects_different_architectures(self) -> None:
        source = build_q_network(STATE_SIZE, ACTION_SIZE, HIDDEN_UNITS)
        target = build_q_network(STATE_SIZE, ACTION_SIZE, (16,))
        with self.assertRaises(ValueError):
            copy_weights(source, target)

    def test_build_optimizer_uses_learning_rate_and_clipnorm(self) -> None:
        optimizer = build_optimizer(0.01, 5.0)
        self.assertAlmostEqual(float(optimizer.learning_rate), 0.01)
        self.assertEqual(optimizer.clipnorm, 5.0)


if __name__ == "__main__":
    unittest.main()
