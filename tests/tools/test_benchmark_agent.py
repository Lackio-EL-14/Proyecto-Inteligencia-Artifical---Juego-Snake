import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.benchmark_agent import (
    COMPARISON_HIDDEN_UNITS,
    CSV_FIELDS,
    NEW_VERSION,
    NEW_XLA_VERSION,
    Measurement,
    calls_per_second,
    legacy_fill_targets,
    milliseconds_per_call,
    new_agent_spec,
    new_version_label,
    run_until_steps,
    time_calls,
    write_csv,
)
from tests.training.fakes import WIN_OUTCOME, FakeAgent, FakeEnvironment, FakeMemory
from training.episode_runner import StepHook
from training.learner import Learner
from training.training_spec import LearnerSpec


class TimingMathTests(unittest.TestCase):
    def test_milliseconds_per_call(self) -> None:
        self.assertAlmostEqual(milliseconds_per_call(2.0, 4), 500.0)

    def test_calls_per_second_is_inverse_of_milliseconds(self) -> None:
        self.assertAlmostEqual(calls_per_second(4.0), 250.0)

    def test_time_calls_runs_warmup_plus_measured_calls(self) -> None:
        counter = []
        time_calls(lambda: counter.append(1), warmup=3, measured=5)
        self.assertEqual(len(counter), 8)


class LegacyBellmanTests(unittest.TestCase):
    def test_fill_targets_matches_original_bellman_loop(self) -> None:
        targets = np.zeros((2, 3), dtype=np.float32)
        q_next = np.array([[1.0, 4.0, 2.0], [9.0, 0.0, 0.0]], dtype=np.float32)
        batch = (None, np.array([1, 2]), np.array([0.5, -1.0]), None, np.array([False, True]))
        legacy_fill_targets(targets, q_next, batch, 0.5)
        expected = np.array([[0.0, 2.5, 0.0], [0.0, 0.0, -1.0]], dtype=np.float32)
        np.testing.assert_allclose(targets, expected)


class NewVersionTests(unittest.TestCase):
    def test_version_label_reflects_xla(self) -> None:
        self.assertEqual(new_version_label(False), NEW_VERSION)
        self.assertEqual(new_version_label(True), NEW_XLA_VERSION)

    def test_new_agent_spec_matches_legacy_architecture(self) -> None:
        spec = new_agent_spec(True)
        self.assertEqual(spec.hidden_units, COMPARISON_HIDDEN_UNITS)
        self.assertTrue(spec.use_xla)


class RunUntilStepsTests(unittest.TestCase):
    def test_runs_whole_episodes_until_reaching_minimum_steps(self) -> None:
        agent, memory = FakeAgent(), FakeMemory()
        hook = StepHook(memory, Learner(agent, memory, LearnerSpec(2, 0, 1, 1)))
        steps = run_until_steps(FakeEnvironment(WIN_OUTCOME, steps_to_finish=3), agent, hook, minimum_steps=7)
        self.assertEqual(steps, 9)
        self.assertEqual(len(memory.transitions), 9)


class CsvTests(unittest.TestCase):
    def test_write_csv_writes_header_once_and_appends_rows(self) -> None:
        measurement = Measurement("anterior", "prueba", "CPU", 2.0, 10)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics" / "bench.csv"
            write_csv(str(path), [measurement])
            write_csv(str(path), [measurement])
            with path.open(encoding="utf-8") as handle:
                rows = list(csv.reader(handle))
        self.assertEqual(rows[0], list(CSV_FIELDS))
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[1], ["anterior", "prueba", "CPU", "2.0000", "500.0", "10"])


if __name__ == "__main__":
    unittest.main()
