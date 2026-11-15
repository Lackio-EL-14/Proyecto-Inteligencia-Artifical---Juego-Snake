import unittest

import numpy as np
import tensorflow as tf

from utils.device import use_cpu_only
from utils.seeding import seed_everything


def setUpModule() -> None:
    use_cpu_only()


def draw_numbers(seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = seed_everything(seed)
    return rng.random(5), tf.random.uniform([5]).numpy()


class SeedEverythingTests(unittest.TestCase):
    def test_same_seed_produces_same_numbers(self) -> None:
        first_numpy, first_tensorflow = draw_numbers(123)
        second_numpy, second_tensorflow = draw_numbers(123)
        np.testing.assert_array_equal(first_numpy, second_numpy)
        np.testing.assert_array_equal(first_tensorflow, second_tensorflow)

    def test_different_seeds_produce_different_numbers(self) -> None:
        self.assertFalse(np.array_equal(draw_numbers(1)[0], draw_numbers(2)[0]))


if __name__ == "__main__":
    unittest.main()
