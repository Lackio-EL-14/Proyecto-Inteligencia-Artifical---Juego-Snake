from typing import NamedTuple

import numpy as np


class TransitionBatch(NamedTuple):
    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_states: np.ndarray
    dones: np.ndarray
    weights: np.ndarray


def batch_from_sample(sample: tuple[np.ndarray, ...]) -> TransitionBatch:
    states, actions, rewards, next_states, dones = sample
    return TransitionBatch(
        states=np.asarray(states).astype(np.float32, copy=False),
        actions=np.asarray(actions).astype(np.int32, copy=False),
        rewards=np.asarray(rewards).astype(np.float32, copy=False),
        next_states=np.asarray(next_states).astype(np.float32, copy=False),
        dones=np.asarray(dones).astype(np.bool_, copy=False),
        weights=np.ones(len(actions), dtype=np.float32),
    )
