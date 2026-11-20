import os
from collections import deque
from typing import NamedTuple

from config import USE_GPU

if not USE_GPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam

from config import ACTION_COUNT, EPSILON_DECAY_FRACTION, EPSILON_END, EPSILON_START, HIDDEN_UNITS, LEARNING_RATE, STATE_SIZE


class Batch(NamedTuple):
    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_states: np.ndarray
    dones: np.ndarray


def build_q_network(trainable: bool = True) -> Sequential:
    model = Sequential([Input(shape=(STATE_SIZE,))])
    for units in HIDDEN_UNITS:
        model.add(Dense(units, activation="relu"))
    model.add(Dense(ACTION_COUNT, activation="linear"))
    if trainable:
        model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss="huber")
    return model


def load_network(weights_path: str) -> Sequential:
    network = build_q_network(trainable=False)
    network.load_weights(weights_path)
    return network


def sync_target(online: Sequential, target: Sequential) -> None:
    target.set_weights(online.get_weights())


def q_values(network: Sequential, states: np.ndarray) -> np.ndarray:
    return network(np.asarray(states, dtype=np.float32).reshape(-1, STATE_SIZE), training=False).numpy()


def greedy_action(network: Sequential, state: np.ndarray) -> int:
    return int(np.argmax(q_values(network, state)[0]))


def epsilon_greedy(network: Sequential, state: np.ndarray, epsilon: float, rng: np.random.Generator) -> int:
    if rng.random() < epsilon:
        return int(rng.integers(ACTION_COUNT))
    return greedy_action(network, state)


def epsilon_at(episode: int, total_episodes: int) -> float:
    progress = min(1.0, episode / (EPSILON_DECAY_FRACTION * total_episodes))
    return EPSILON_START + progress * (EPSILON_END - EPSILON_START)


class ReplayMemory:
    def __init__(self, capacity: int) -> None:
        self.buffer: deque = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self.buffer)

    def add(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int, rng: np.random.Generator) -> Batch:
        indices = rng.choice(len(self.buffer), size=batch_size, replace=False)
        columns = zip(*(self.buffer[index] for index in indices))
        states, actions, rewards, next_states, dones = (np.asarray(column) for column in columns)
        return Batch(states.astype(np.float32), actions, rewards.astype(np.float32), next_states.astype(np.float32), dones.astype(np.float32))


def double_dqn_targets(q_online_next: np.ndarray, q_target_next: np.ndarray, batch: Batch, gamma: float) -> np.ndarray:
    best_actions = np.argmax(q_online_next, axis=1)
    best_values = q_target_next[np.arange(len(best_actions)), best_actions]
    return batch.rewards + gamma * (1.0 - batch.dones) * best_values


def training_targets(q_current: np.ndarray, actions: np.ndarray, targets: np.ndarray) -> np.ndarray:
    full_targets = q_current.copy()
    full_targets[np.arange(len(actions)), actions] = targets
    return full_targets


def train_step(online: Sequential, target: Sequential, batch: Batch, gamma: float) -> float:
    targets = double_dqn_targets(q_values(online, batch.next_states), q_values(target, batch.next_states), batch, gamma)
    full_targets = training_targets(q_values(online, batch.states), batch.actions, targets)
    return float(online.train_on_batch(batch.states, full_targets))


def set_seed(seed: int) -> None:
    tf.keras.utils.set_random_seed(seed)
