import unittest

import numpy as np

from agent import Batch, ReplayMemory, build_q_network, double_dqn_targets, epsilon_at, q_values, set_seed, sync_target, train_step, training_targets
from config import EPSILON_END, EPSILON_START, STATE_SIZE


def tiny_batch(size: int = 4) -> Batch:
    rng = np.random.default_rng(0)
    states = rng.random((size, STATE_SIZE)).astype(np.float32)
    return Batch(states, np.zeros(size, dtype=int), np.ones(size, dtype=np.float32), states, np.zeros(size, dtype=np.float32))


class ReplayMemoryTest(unittest.TestCase):
    def test_capacity_discards_oldest(self) -> None:
        memory = ReplayMemory(2)
        for value in range(3):
            memory.add(np.full(STATE_SIZE, value), value, 0.0, np.zeros(STATE_SIZE), False)
        self.assertEqual(len(memory), 2)
        self.assertEqual(memory.buffer[0][1], 1)

    def test_sample_returns_batch_arrays(self) -> None:
        memory = ReplayMemory(10)
        for value in range(5):
            memory.add(np.zeros(STATE_SIZE), value % 3, 1.0, np.ones(STATE_SIZE), value == 4)
        batch = memory.sample(3, np.random.default_rng(0))
        self.assertEqual(batch.states.shape, (3, STATE_SIZE))
        self.assertEqual(batch.actions.shape, (3,))


class DoubleDqnTest(unittest.TestCase):
    def test_online_chooses_and_target_evaluates(self) -> None:
        batch = Batch(None, None, np.array([1.0, 1.0]), None, np.array([0.0, 1.0]))
        q_online_next = np.array([[0.0, 9.0, 0.0], [9.0, 0.0, 0.0]])
        q_target_next = np.array([[5.0, 2.0, 7.0], [3.0, 3.0, 3.0]])
        np.testing.assert_allclose(double_dqn_targets(q_online_next, q_target_next, batch, 0.5), [2.0, 1.0])

    def test_training_targets_only_change_taken_action(self) -> None:
        full = training_targets(np.zeros((2, 3)), np.array([2, 0]), np.array([4.0, -1.0]))
        np.testing.assert_array_equal(full, [[0, 0, 4], [-1, 0, 0]])

    def test_epsilon_decays_linearly_to_floor(self) -> None:
        self.assertEqual(epsilon_at(0, 100), EPSILON_START)
        self.assertAlmostEqual(epsilon_at(100, 100), EPSILON_END)


class TrainStepTest(unittest.TestCase):
    def test_train_step_reduces_loss_on_fixed_batch(self) -> None:
        set_seed(0)
        online, target = build_q_network(), build_q_network()
        sync_target(online, target)
        batch = tiny_batch()
        first_loss = train_step(online, target, batch, 0.0)
        for _ in range(30):
            last_loss = train_step(online, target, batch, 0.0)
        self.assertLess(last_loss, first_loss)

    def test_sync_target_copies_weights(self) -> None:
        online, target = build_q_network(), build_q_network()
        sync_target(online, target)
        states = tiny_batch().states
        np.testing.assert_allclose(q_values(online, states), q_values(target, states))


if __name__ == "__main__":
    unittest.main()
