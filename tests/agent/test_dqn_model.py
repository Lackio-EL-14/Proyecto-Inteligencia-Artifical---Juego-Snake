import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from agent.agent_spec import AgentSpec
from agent.batch import TransitionBatch, batch_from_sample
from agent.dqn_model import DQNAgent
from utils.device import use_cpu_only

SPEC = AgentSpec(
    state_size=11,
    action_size=4,
    hidden_units=(16, 16),
    learning_rate=0.01,
    gradient_clipnorm=10.0,
    gamma=0.9,
    huber_delta=1.0,
    target_rule="double",
    use_xla=False,
)
BATCH_SIZE = 32


def setUpModule() -> None:
    use_cpu_only()


def make_agent(seed: int = 0) -> DQNAgent:
    return DQNAgent(SPEC, np.random.default_rng(seed))


def make_batch(seed: int, batch_size: int = BATCH_SIZE) -> TransitionBatch:
    rng = np.random.default_rng(seed)
    return batch_from_sample(
        (
            rng.random((batch_size, SPEC.state_size), dtype=np.float32),
            rng.integers(SPEC.action_size, size=batch_size),
            rng.standard_normal(batch_size).astype(np.float32),
            rng.random((batch_size, SPEC.state_size), dtype=np.float32),
            rng.random(batch_size) < 0.2,
        )
    )


def random_states(seed: int, rows: int) -> np.ndarray:
    return np.random.default_rng(seed).random((rows, SPEC.state_size), dtype=np.float32)


class ActionTests(unittest.TestCase):
    def test_greedy_action_matches_argmax_of_q_values(self) -> None:
        agent = make_agent()
        for state in random_states(1, 10):
            self.assertEqual(agent.greedy_action(state), int(np.argmax(agent.q_values(state[None])[0])))

    def test_act_with_epsilon_one_is_random_and_in_range(self) -> None:
        agent = make_agent(seed=3)
        state = random_states(2, 1)[0]
        actions = {agent.act(state, 1.0) for _ in range(200)}
        self.assertEqual(actions, set(range(SPEC.action_size)))

    def test_act_with_epsilon_zero_is_greedy(self) -> None:
        agent = make_agent()
        state = random_states(4, 1)[0]
        self.assertEqual(agent.act(state, 0.0), agent.greedy_action(state))

    def test_q_values_shape(self) -> None:
        self.assertEqual(make_agent().q_values(random_states(5, 7)).shape, (7, SPEC.action_size))


class TrainingTests(unittest.TestCase):
    def test_train_step_reduces_loss_on_fixed_batch(self) -> None:
        agent = make_agent()
        batch = make_batch(seed=6)
        first_loss = float(agent.train_step(batch).loss)
        for _ in range(200):
            last = agent.train_step(batch)
        self.assertLess(float(last.loss), first_loss)

    def test_train_step_returns_one_td_error_per_transition(self) -> None:
        result = make_agent().train_step(make_batch(seed=7))
        self.assertEqual(tuple(result.td_errors.shape), (BATCH_SIZE,))

    def test_train_step_does_not_retrace(self) -> None:
        agent = make_agent()
        for seed in range(5):
            agent.train_step(make_batch(seed))
        agent.train_step(make_batch(seed=9, batch_size=8))
        self.assertEqual(agent._train.experimental_get_tracing_count(), 1)

    def test_sync_target_makes_outputs_equal(self) -> None:
        agent = make_agent()
        states = random_states(8, 6)
        for seed in range(20):
            agent.train_step(make_batch(seed))
        self.assertFalse(np.allclose(agent.model(states).numpy(), agent._target(states).numpy()))
        agent.sync_target()
        np.testing.assert_allclose(agent.model(states).numpy(), agent._target(states).numpy())

    def test_classic_dqn_rule_also_trains(self) -> None:
        agent = DQNAgent(replace(SPEC, target_rule="dqn"), np.random.default_rng(0))
        self.assertTrue(np.isfinite(float(agent.train_step(make_batch(seed=10)).loss)))

    def test_unknown_target_rule_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            DQNAgent(replace(SPEC, target_rule="desconocida"), np.random.default_rng(0))


class PersistenceTests(unittest.TestCase):
    def test_save_load_roundtrip(self) -> None:
        agent = make_agent()
        for seed in range(5):
            agent.train_step(make_batch(seed))
        states = random_states(11, 5)
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "models" / "agent.weights.h5")
            agent.save(path)
            restored = make_agent(seed=1)
            restored.load(path)
        np.testing.assert_allclose(restored.q_values(states), agent.q_values(states))
        np.testing.assert_allclose(restored._target(states).numpy(), restored.model(states).numpy())


if __name__ == "__main__":
    unittest.main()
