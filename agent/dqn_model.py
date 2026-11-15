from pathlib import Path
from typing import NamedTuple

import numpy as np
import tensorflow as tf

from agent.agent_spec import AgentSpec
from agent.batch import TransitionBatch
from agent.network import build_optimizer, build_q_network, copy_weights
from agent.q_math import BellmanInputs, gather_action_values, huber, select_target_rule, weighted_mean


class TrainResult(NamedTuple):
    loss: tf.Tensor
    td_errors: tf.Tensor


def batch_signature(state_size: int) -> TransitionBatch:
    return TransitionBatch(
        states=tf.TensorSpec([None, state_size], tf.float32),
        actions=tf.TensorSpec([None], tf.int32),
        rewards=tf.TensorSpec([None], tf.float32),
        next_states=tf.TensorSpec([None, state_size], tf.float32),
        dones=tf.TensorSpec([None], tf.bool),
        weights=tf.TensorSpec([None], tf.float32),
    )


def as_state_batch(states: np.ndarray, state_size: int) -> np.ndarray:
    return np.asarray(states, dtype=np.float32).reshape(-1, state_size)


class DQNAgent:
    def __init__(self, spec: AgentSpec, rng: np.random.Generator) -> None:
        self._spec = spec
        self._rng = rng
        self._online = build_q_network(spec.state_size, spec.action_size, spec.hidden_units)
        self._target = build_q_network(spec.state_size, spec.action_size, spec.hidden_units)
        copy_weights(self._online, self._target)
        self._optimizer = build_optimizer(spec.learning_rate, spec.gradient_clipnorm)
        self._optimizer.build(self._online.trainable_variables)
        self._target_rule = select_target_rule(spec.target_rule)
        self._compile_functions()

    def _compile_functions(self) -> None:
        size = self._spec.state_size
        self._forward = tf.function(self._forward_impl, input_signature=[tf.TensorSpec([None, size], tf.float32)])
        self._greedy = tf.function(self._greedy_impl, input_signature=[tf.TensorSpec([1, size], tf.float32)])
        self._train = tf.function(
            self._train_impl, input_signature=[batch_signature(size)], jit_compile=self._spec.use_xla
        )

    @property
    def model(self) -> tf.keras.Model:
        return self._online

    def act(self, state: np.ndarray, epsilon: float) -> int:
        if self._rng.random() < epsilon:
            return int(self._rng.integers(self._spec.action_size))
        return self.greedy_action(state)

    def greedy_action(self, state: np.ndarray) -> int:
        return int(self._greedy(as_state_batch(state, self._spec.state_size)))

    def q_values(self, states: np.ndarray) -> np.ndarray:
        return self._forward(as_state_batch(states, self._spec.state_size)).numpy()

    def train_step(self, batch: TransitionBatch) -> TrainResult:
        loss, td_errors = self._train(batch)
        return TrainResult(loss=loss, td_errors=td_errors)

    def sync_target(self) -> None:
        copy_weights(self._online, self._target)

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._online.save_weights(path)

    def load(self, path: str) -> None:
        self._online.load_weights(path)
        self.sync_target()

    def _forward_impl(self, states: tf.Tensor) -> tf.Tensor:
        return self._online(states, training=False)

    def _greedy_impl(self, state: tf.Tensor) -> tf.Tensor:
        return tf.argmax(self._online(state, training=False), axis=1, output_type=tf.int32)[0]

    def _train_impl(self, batch: TransitionBatch) -> tuple[tf.Tensor, tf.Tensor]:
        targets = self._compute_targets(batch)
        return self._apply_gradients(batch, targets)

    def _compute_targets(self, batch: TransitionBatch) -> tf.Tensor:
        inputs = BellmanInputs(
            rewards=batch.rewards,
            dones=batch.dones,
            next_q_online=self._online(batch.next_states, training=False),
            next_q_target=self._target(batch.next_states, training=False),
        )
        return tf.stop_gradient(self._target_rule(inputs, self._spec.gamma))

    def _apply_gradients(self, batch: TransitionBatch, targets: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        variables = self._online.trainable_variables
        with tf.GradientTape() as tape:
            q_taken = gather_action_values(self._online(batch.states, training=True), batch.actions)
            td_errors = targets - q_taken
            loss = weighted_mean(huber(td_errors, self._spec.huber_delta), batch.weights)
        gradients = tape.gradient(loss, variables)
        self._optimizer.apply_gradients(zip(gradients, variables))
        return loss, td_errors
