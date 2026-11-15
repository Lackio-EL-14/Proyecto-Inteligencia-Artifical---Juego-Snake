from collections.abc import Callable
from dataclasses import dataclass

import tensorflow as tf

HUBER_QUADRATIC_FACTOR = 0.5


@dataclass(frozen=True)
class BellmanInputs:
    rewards: tf.Tensor
    dones: tf.Tensor
    next_q_online: tf.Tensor
    next_q_target: tf.Tensor


TargetRule = Callable[[BellmanInputs, float], tf.Tensor]


def gather_action_values(q_values: tf.Tensor, actions: tf.Tensor) -> tf.Tensor:
    return tf.gather(q_values, actions, axis=1, batch_dims=1)


def continuation_mask(dones: tf.Tensor) -> tf.Tensor:
    return 1.0 - tf.cast(dones, tf.float32)


def bootstrap(inputs: BellmanInputs, next_values: tf.Tensor, gamma: float) -> tf.Tensor:
    return inputs.rewards + gamma * continuation_mask(inputs.dones) * next_values


def dqn_targets(inputs: BellmanInputs, gamma: float) -> tf.Tensor:
    return bootstrap(inputs, tf.reduce_max(inputs.next_q_target, axis=1), gamma)


def double_dqn_targets(inputs: BellmanInputs, gamma: float) -> tf.Tensor:
    best_actions = tf.argmax(inputs.next_q_online, axis=1, output_type=tf.int32)
    return bootstrap(inputs, gather_action_values(inputs.next_q_target, best_actions), gamma)


def huber(td_errors: tf.Tensor, delta: float) -> tf.Tensor:
    absolute_errors = tf.abs(td_errors)
    quadratic = HUBER_QUADRATIC_FACTOR * tf.square(td_errors)
    linear = delta * (absolute_errors - HUBER_QUADRATIC_FACTOR * delta)
    return tf.where(absolute_errors <= delta, quadratic, linear)


def weighted_mean(values: tf.Tensor, weights: tf.Tensor) -> tf.Tensor:
    return tf.math.divide_no_nan(tf.reduce_sum(values * weights), tf.reduce_sum(weights))


TARGET_RULES: dict[str, TargetRule] = {"dqn": dqn_targets, "double": double_dqn_targets}


def select_target_rule(name: str) -> TargetRule:
    if name not in TARGET_RULES:
        raise ValueError(f"Regla de objetivo desconocida: '{name}'. Opciones: {sorted(TARGET_RULES)}")
    return TARGET_RULES[name]
