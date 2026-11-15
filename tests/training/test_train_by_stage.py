import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from tests.training.fakes import TIME_LIMIT_OUTCOME, WIN_OUTCOME, FakeAgent, FakeEnvironment, FakeEpsilon, FakeMemory, StepOutcome
from training.checkpoints import final_checkpoint_path, stage_checkpoint_path
from training.train import TrainingComponents, train_by_stage
from training.training_spec import LearnerSpec, TrainingSpec

LEARNER_SPEC = LearnerSpec(batch_size=2, warmup_steps=0, train_every=1, target_update_every=1)


def tiny_spec(run_dir: str, level_count: int) -> TrainingSpec:
    return TrainingSpec(
        level_paths=tuple(f"fake-level-{index}.json" for index in range(level_count)),
        learner=LEARNER_SPEC,
        max_steps_per_episode=5,
        max_episodes_per_stage=1,
        window_size=1,
        target_win_rate=1.0,
        eval_every=1,
        eval_episodes=1,
        eval_seed=100,
        rehearsal_prob=0.0,
        run_dir=run_dir,
        seed=7,
        progress_every=1,
    )


class TrainingRun:
    def __init__(self, outcome: StepOutcome, level_count: int = 1) -> None:
        self.agent = FakeAgent()
        self.memory = FakeMemory()
        self.epsilon = FakeEpsilon()
        self.environments: list[FakeEnvironment] = []
        self.outcome = outcome
        self.level_count = level_count

    def make_env(self, level_path: str, max_steps: int) -> FakeEnvironment:
        env = FakeEnvironment(self.outcome)
        self.environments.append(env)
        return env

    def run(self, run_dir: str) -> object:
        components = TrainingComponents(self.agent, self.memory, self.make_env, self.epsilon, np.random.default_rng(0))
        with patch("builtins.print"):
            return train_by_stage(components, tiny_spec(run_dir, self.level_count))


class TrainByStageTests(unittest.TestCase):
    def test_training_consumes_new_api_and_preserves_terminal_mask(self) -> None:
        training = TrainingRun(WIN_OUTCOME)
        with tempfile.TemporaryDirectory() as run_dir:
            training.run(run_dir)
            self.assertEqual(len(training.memory.transitions), 1)
            self.assertTrue(training.memory.transitions[0][4])
            self.assertEqual(training.agent.saved_paths[-1], final_checkpoint_path(run_dir))

    def test_training_does_not_mark_truncation_as_terminal(self) -> None:
        training = TrainingRun(TIME_LIMIT_OUTCOME)
        with tempfile.TemporaryDirectory() as run_dir:
            training.run(run_dir)
        self.assertEqual(len(training.memory.transitions), 1)
        self.assertFalse(training.memory.transitions[0][4])

    def test_restart_for_stage_is_called_once_per_new_stage(self) -> None:
        training = TrainingRun(WIN_OUTCOME, level_count=3)
        with tempfile.TemporaryDirectory() as run_dir:
            training.run(run_dir)
        self.assertEqual(training.epsilon.restart_calls, 2)
        self.assertEqual(training.epsilon.step_calls, 3)

    def test_saves_best_and_last_checkpoints_per_stage(self) -> None:
        training = TrainingRun(WIN_OUTCOME, level_count=2)
        with tempfile.TemporaryDirectory() as run_dir:
            training.run(run_dir)
            for stage in (1, 2):
                self.assertIn(stage_checkpoint_path(run_dir, stage, "best"), training.agent.saved_paths)
                self.assertIn(stage_checkpoint_path(run_dir, stage, "last"), training.agent.saved_paths)

    def test_report_contains_curriculum_matrix_and_passed_stages(self) -> None:
        training = TrainingRun(WIN_OUTCOME, level_count=2)
        with tempfile.TemporaryDirectory() as run_dir:
            report = training.run(run_dir)
        self.assertEqual(report.matrix, ((1.0, 1.0), (1.0, 1.0)))
        self.assertTrue(all(stage.passed for stage in report.stages))

    def test_training_environments_are_seeded_once_with_offsets(self) -> None:
        training = TrainingRun(WIN_OUTCOME, level_count=2)
        with tempfile.TemporaryDirectory() as run_dir:
            training.run(run_dir)
        train_envs = training.environments[:2]
        self.assertEqual([env.reset_seeds[0] for env in train_envs], [7, 8])
        self.assertTrue(all(seed is None for env in train_envs for seed in env.reset_seeds[1:]))

    def test_writes_metrics_and_plots_inside_run_dir(self) -> None:
        training = TrainingRun(WIN_OUTCOME, level_count=2)
        with tempfile.TemporaryDirectory() as run_dir:
            training.run(run_dir)
            created = {path.relative_to(run_dir).as_posix() for path in Path(run_dir).rglob("*.*")}
        expected = {
            "metrics/episodes.csv",
            "metrics/evaluations.csv",
            "metrics/curriculum_matrix.csv",
            "plots/learning_curve_level_1.png",
            "plots/learning_curve_level_2.png",
            "plots/curriculum_matrix.png",
            "plots/eval_win_rate.png",
        }
        self.assertEqual(created, expected)


if __name__ == "__main__":
    unittest.main()
