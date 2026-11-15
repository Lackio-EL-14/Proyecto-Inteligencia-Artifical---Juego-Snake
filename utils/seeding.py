import numpy as np
import tensorflow as tf


def seed_everything(seed: int) -> np.random.Generator:
    tf.random.set_seed(seed)
    return np.random.default_rng(seed)
