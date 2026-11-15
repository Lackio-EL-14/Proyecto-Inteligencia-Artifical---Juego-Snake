import argparse
import csv
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tensorflow as tf

from agent.agent_spec import AgentSpec
from agent.batch import batch_from_sample
from agent.dqn_model import DQNAgent
from core.experience import ExperienceReplay
from core.snake_env import SnakeEnv
from training.episode_runner import StepHook, run_training_episode
from training.factories import build_agent_spec, build_learner_spec, build_memory
from training.learner import Learner
from utils.config import (
    ACTION_SIZE,
    BATCH_SIZE,
    EPSILON_MIN,
    GAMMA,
    GRADIENT_CLIPNORM,
    HUBER_DELTA,
    LEARNING_RATE,
    LEVEL_PATHS,
    MAX_MEMORY,
    MAX_STEPS_PER_EPISODE,
    STATE_SIZE,
    TARGET_RULE,
    WARMUP_STEPS,
)
from utils.device import use_cpu_only

WARMUP_CALLS = 50
DEFAULT_MEASURED_CALLS = 500
BENCHMARK_SEED = 42
RANDOM_TERMINAL_FRACTION = 0.1
COMPARISON_HIDDEN_UNITS = (64, 64)
LEGACY_VERSION = "anterior"
NEW_VERSION = "nueva"
NEW_XLA_VERSION = "nueva_xla"
LEARNER_VERSION = "nueva_learner"
MILLISECONDS_PER_SECOND = 1000.0
CSV_PATH = "metrics/benchmark_agent.csv"
CSV_FIELDS = ("version", "medicion", "dispositivo", "ms_por_llamada", "llamadas_por_segundo", "llamadas_medidas")
ACTION_MEASUREMENT = "elegir_accion"
TRAIN_MEASUREMENT = "paso_entrenamiento_lote"
END_TO_END_MEASUREMENT = "paso_entorno_punta_a_punta"

LegacyBatch = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
Policy = Callable[[np.ndarray], int]
Learn = Callable[[LegacyBatch], object]


@dataclass(frozen=True)
class Measurement:
    version: str
    name: str
    device: str
    ms_per_call: float
    calls: int


@dataclass(frozen=True)
class Operations:
    choose_action: Callable[[], object]
    train_batch: Callable[[], object]
    env_step: Callable[[], object]


def milliseconds_per_call(elapsed_seconds: float, calls: int) -> float:
    return elapsed_seconds * MILLISECONDS_PER_SECOND / calls


def calls_per_second(ms_per_call: float) -> float:
    return MILLISECONDS_PER_SECOND / ms_per_call


def time_calls(call: Callable[[], object], warmup: int, measured: int) -> float:
    for _ in range(warmup):
        call()
    start = time.perf_counter()
    for _ in range(measured):
        call()
    return milliseconds_per_call(time.perf_counter() - start, measured)


def visible_device_label() -> str:
    return "GPU" if tf.config.list_logical_devices("GPU") else "CPU"


def new_version_label(use_xla: bool) -> str:
    return NEW_XLA_VERSION if use_xla else NEW_VERSION


def new_agent_spec(use_xla: bool) -> AgentSpec:
    return AgentSpec(
        state_size=STATE_SIZE,
        action_size=ACTION_SIZE,
        hidden_units=COMPARISON_HIDDEN_UNITS,
        learning_rate=LEARNING_RATE,
        gradient_clipnorm=GRADIENT_CLIPNORM,
        gamma=GAMMA,
        huber_delta=HUBER_DELTA,
        target_rule=TARGET_RULE,
        use_xla=use_xla,
    )


def build_legacy_model(state_size: int, action_size: int, learning_rate: float) -> tf.keras.Model:
    hidden_layers = [tf.keras.layers.Dense(units, activation="relu") for units in COMPARISON_HIDDEN_UNITS]
    model = tf.keras.Sequential(
        [tf.keras.Input(shape=(state_size,)), *hidden_layers, tf.keras.layers.Dense(action_size, activation="linear")]
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss="mse")
    return model


def legacy_greedy_action(model: tf.keras.Model, state: np.ndarray) -> int:
    q_values = model.predict(np.expand_dims(state, axis=0), verbose=0)
    return int(np.argmax(q_values[0]))


def legacy_act(model: tf.keras.Model, state: np.ndarray, epsilon: float, rng: np.random.Generator) -> int:
    if rng.random() <= epsilon:
        return int(rng.integers(ACTION_SIZE))
    return legacy_greedy_action(model, state)


def legacy_fill_targets(targets: np.ndarray, q_next: np.ndarray, batch: LegacyBatch, gamma: float) -> None:
    _, actions, rewards, _, dones = batch
    for i in range(len(actions)):
        if dones[i]:
            targets[i, actions[i]] = rewards[i]
        else:
            targets[i, actions[i]] = rewards[i] + gamma * np.amax(q_next[i])


def legacy_train_step(model: tf.keras.Model, batch: LegacyBatch, gamma: float) -> None:
    states, _, _, next_states, _ = batch
    targets = model.predict(states, verbose=0)
    q_next = model.predict(next_states, verbose=0)
    legacy_fill_targets(targets, q_next, batch, gamma)
    model.train_on_batch(states, targets)


def random_batch(rng: np.random.Generator, batch_size: int) -> LegacyBatch:
    states = rng.random((batch_size, STATE_SIZE), dtype=np.float32)
    actions = rng.integers(ACTION_SIZE, size=batch_size).astype(np.int32)
    rewards = rng.standard_normal(batch_size).astype(np.float32)
    next_states = rng.random((batch_size, STATE_SIZE), dtype=np.float32)
    dones = rng.random(batch_size) < RANDOM_TERMINAL_FRACTION
    return states, actions, rewards, next_states, dones


class TrainingLoop:
    def __init__(self, env: SnakeEnv, memory: ExperienceReplay, policy: Policy, learn: Learn) -> None:
        self.env = env
        self.memory = memory
        self.policy = policy
        self.learn = learn
        self.state, _ = env.reset(seed=BENCHMARK_SEED)

    def step(self) -> None:
        action = self.policy(self.state)
        next_state, reward, terminated, truncated, _ = self.env.step(action)
        self.memory.remember(self.state, action, reward, next_state, terminated)
        if len(self.memory) > BATCH_SIZE:
            self.learn(self.memory.sample_batch(BATCH_SIZE))
        self.state = self.next_state_after(next_state, terminated or truncated)

    def next_state_after(self, next_state: np.ndarray, episode_done: bool) -> np.ndarray:
        if episode_done:
            return self.env.reset()[0]
        return next_state


def build_training_loop(policy: Policy, learn: Learn) -> TrainingLoop:
    env = SnakeEnv(LEVEL_PATHS[0], max_steps=MAX_STEPS_PER_EPISODE)
    memory = ExperienceReplay(max_memory=MAX_MEMORY, state_size=STATE_SIZE, seed=BENCHMARK_SEED)
    return TrainingLoop(env, memory, policy, learn)


def legacy_operations(rng: np.random.Generator) -> Operations:
    model = build_legacy_model(STATE_SIZE, ACTION_SIZE, LEARNING_RATE)
    state = rng.random(STATE_SIZE, dtype=np.float32)
    batch = random_batch(rng, BATCH_SIZE)
    loop = build_training_loop(
        lambda current: legacy_act(model, current, EPSILON_MIN, rng),
        lambda sample: legacy_train_step(model, sample, GAMMA),
    )
    return Operations(
        choose_action=lambda: legacy_greedy_action(model, state),
        train_batch=lambda: legacy_train_step(model, batch, GAMMA),
        env_step=loop.step,
    )


def new_operations(rng: np.random.Generator, use_xla: bool) -> Operations:
    agent = DQNAgent(new_agent_spec(use_xla), rng)
    state = rng.random(STATE_SIZE, dtype=np.float32)
    batch = batch_from_sample(random_batch(rng, BATCH_SIZE))
    loop = build_training_loop(
        lambda current: agent.act(current, EPSILON_MIN),
        lambda sample: agent.train_step(batch_from_sample(sample)),
    )
    return Operations(
        choose_action=lambda: agent.greedy_action(state),
        train_batch=lambda: agent.train_step(batch),
        env_step=loop.step,
    )


def measure_operations(version: str, device: str, operations: Operations, calls: int) -> list[Measurement]:
    action_ms = time_calls(operations.choose_action, WARMUP_CALLS, calls)
    train_ms = time_calls(operations.train_batch, WARMUP_CALLS, calls)
    end_to_end_ms = time_calls(operations.env_step, WARMUP_CALLS + BATCH_SIZE, calls)
    return [
        Measurement(version, ACTION_MEASUREMENT, device, action_ms, calls),
        Measurement(version, TRAIN_MEASUREMENT, device, train_ms, calls),
        Measurement(version, END_TO_END_MEASUREMENT, device, end_to_end_ms, calls),
    ]


def run_until_steps(env: SnakeEnv, agent: DQNAgent, step_hook: StepHook, minimum_steps: int) -> int:
    steps = 0
    while steps < minimum_steps:
        steps += run_training_episode(env, agent, EPSILON_MIN, step_hook).steps
    return steps


def measure_learner_end_to_end(rng: np.random.Generator, device: str, calls: int) -> Measurement:
    agent = DQNAgent(build_agent_spec(), rng)
    memory = build_memory(BENCHMARK_SEED)
    step_hook = StepHook(memory, Learner(agent, memory, build_learner_spec()))
    env = SnakeEnv(LEVEL_PATHS[0], max_steps=MAX_STEPS_PER_EPISODE)
    env.reset(seed=BENCHMARK_SEED)
    run_until_steps(env, agent, step_hook, WARMUP_STEPS + WARMUP_CALLS)
    start = time.perf_counter()
    steps = run_until_steps(env, agent, step_hook, calls)
    ms_per_step = milliseconds_per_call(time.perf_counter() - start, steps)
    return Measurement(LEARNER_VERSION, END_TO_END_MEASUREMENT, device, ms_per_step, steps)


def measurement_row(measurement: Measurement) -> dict[str, str]:
    values = (
        measurement.version,
        measurement.name,
        measurement.device,
        f"{measurement.ms_per_call:.4f}",
        f"{calls_per_second(measurement.ms_per_call):.1f}",
        str(measurement.calls),
    )
    return dict(zip(CSV_FIELDS, values))


def print_measurements(measurements: list[Measurement]) -> None:
    print(" | ".join(CSV_FIELDS))
    for measurement in measurements:
        print(" | ".join(measurement_row(measurement).values()))


def write_csv(path: str, measurements: list[Measurement]) -> None:
    csv_path = Path(path)
    is_new_file = not csv_path.exists()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        if is_new_file:
            writer.writeheader()
        writer.writerows(measurement_row(measurement) for measurement in measurements)


def parse_arguments(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mide el costo por llamada del agente DQN.")
    parser.add_argument("--cpu", action="store_true", help="oculta la GPU a TensorFlow")
    parser.add_argument("--xla", action="store_true", help="compila el paso de entrenamiento nuevo con XLA")
    parser.add_argument("--skip-legacy", action="store_true", help="mide solo la versión nueva")
    parser.add_argument("--steps", type=int, default=DEFAULT_MEASURED_CALLS, help="llamadas medidas por prueba")
    return parser.parse_args(argv)


def collect_measurements(arguments: argparse.Namespace, rng: np.random.Generator, device: str) -> list[Measurement]:
    measurements = []
    if not arguments.skip_legacy:
        measurements += measure_operations(LEGACY_VERSION, device, legacy_operations(rng), arguments.steps)
    new_version = new_version_label(arguments.xla)
    measurements += measure_operations(new_version, device, new_operations(rng, arguments.xla), arguments.steps)
    return measurements + [measure_learner_end_to_end(rng, device, arguments.steps)]


def run(argv: list[str]) -> None:
    arguments = parse_arguments(argv)
    if arguments.cpu:
        use_cpu_only()
    tf.random.set_seed(BENCHMARK_SEED)
    rng = np.random.default_rng(BENCHMARK_SEED)
    measurements = collect_measurements(arguments, rng, visible_device_label())
    print_measurements(measurements)
    write_csv(CSV_PATH, measurements)


if __name__ == "__main__":
    run(sys.argv[1:])
