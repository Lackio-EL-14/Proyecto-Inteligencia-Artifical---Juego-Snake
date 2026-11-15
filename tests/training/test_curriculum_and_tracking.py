import unittest

import numpy as np

from training.curriculum import choose_level_for_episode
from training.curriculum_report import curriculum_matrix, matrix_level_ids
from training.episode_runner import EpisodeResult
from training.evaluation import EvaluationSummary
from training.stage_tracker import StageTracker, should_report_progress, window_win_rate

DRAWS = 200


def result_with_reason(reason: str, reward: float = 0.0) -> EpisodeResult:
    return EpisodeResult(1, reward, 0, 1, reason, None)


def summary(level_id: int, win_rate: float) -> EvaluationSummary:
    return EvaluationSummary(level_id, win_rate, 0.0, 0.0, {})


class ChooseLevelTests(unittest.TestCase):
    def test_stage_zero_always_returns_zero(self) -> None:
        rng = np.random.default_rng(0)
        self.assertEqual({choose_level_for_episode(0, 1.0, rng) for _ in range(DRAWS)}, {0})

    def test_zero_probability_always_returns_current_stage(self) -> None:
        rng = np.random.default_rng(0)
        self.assertEqual({choose_level_for_episode(3, 0.0, rng) for _ in range(DRAWS)}, {3})

    def test_full_probability_always_returns_previous_levels(self) -> None:
        rng = np.random.default_rng(0)
        self.assertEqual({choose_level_for_episode(3, 1.0, rng) for _ in range(DRAWS)}, {0, 1, 2})


class WindowWinRateTests(unittest.TestCase):
    def test_empty_sequence_is_zero(self) -> None:
        self.assertEqual(window_win_rate([]), 0.0)

    def test_all_wins_is_one(self) -> None:
        self.assertEqual(window_win_rate([1, 1, 1]), 1.0)

    def test_mixed_sequence(self) -> None:
        self.assertAlmostEqual(window_win_rate([1, 0, 0, 1]), 0.5)


class StageTrackerTests(unittest.TestCase):
    def test_window_forgets_old_episodes(self) -> None:
        tracker = StageTracker(window_size=2)
        for reason in ("win", "wall", "wall"):
            tracker.record(result_with_reason(reason))
        self.assertEqual(tracker.train_win_rate(), 0.0)
        self.assertEqual(tracker.win_rates(), [1.0, 0.5, 0.0])

    def test_rewards_are_recorded_in_order(self) -> None:
        tracker = StageTracker(window_size=5)
        tracker.record(result_with_reason("win", 1.5))
        tracker.record(result_with_reason("wall", -1.0))
        self.assertEqual(tracker.rewards(), [1.5, -1.0])

    def test_should_report_progress_on_multiples(self) -> None:
        self.assertTrue(should_report_progress(20, 10))
        self.assertFalse(should_report_progress(21, 10))


class CurriculumMatrixTests(unittest.TestCase):
    def test_matrix_rows_are_stages_and_columns_are_levels(self) -> None:
        evaluations = [[summary(1, 0.9), summary(2, 0.1)], [summary(1, 0.7), summary(2, 0.8)]]
        self.assertEqual(curriculum_matrix(evaluations), ((0.9, 0.1), (0.7, 0.8)))
        self.assertEqual(matrix_level_ids(evaluations), (1, 2))

    def test_empty_matrix_has_no_level_ids(self) -> None:
        self.assertEqual(matrix_level_ids([]), ())


if __name__ == "__main__":
    unittest.main()
