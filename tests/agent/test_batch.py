import unittest

import numpy as np

from agent.batch import batch_from_sample


class BatchFromSampleTests(unittest.TestCase):
    def test_converts_dtypes_and_adds_unit_weights(self) -> None:
        sample = (
            np.zeros((3, 2), dtype=np.float64),
            np.array([0, 1, 2], dtype=np.int64),
            np.array([1.0, 0.0, -1.0], dtype=np.float64),
            np.ones((3, 2), dtype=np.float64),
            np.array([0, 0, 1]),
        )
        batch = batch_from_sample(sample)
        self.assertEqual(batch.states.dtype, np.float32)
        self.assertEqual(batch.actions.dtype, np.int32)
        self.assertEqual(batch.rewards.dtype, np.float32)
        self.assertEqual(batch.next_states.dtype, np.float32)
        self.assertEqual(batch.dones.dtype, np.bool_)
        np.testing.assert_array_equal(batch.dones, [False, False, True])
        np.testing.assert_array_equal(batch.weights, np.ones(3, dtype=np.float32))

    def test_keeps_arrays_that_already_have_the_right_dtype(self) -> None:
        states = np.zeros((2, 2), dtype=np.float32)
        sample = (states, np.zeros(2, np.int32), np.zeros(2, np.float32), states, np.zeros(2, np.bool_))
        self.assertIs(batch_from_sample(sample).states, states)


if __name__ == "__main__":
    unittest.main()
