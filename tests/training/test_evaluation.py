import unittest

import numpy as np

from tests.training.fakes import STATE_SIZE, FakeAgent
from training.episode_runner import EpisodeResult
from training.evaluation import evaluate_all_levels, evaluate_level, is_improvement, should_evaluate, stage_is_passed, summarize_results


class SeedDependentEnvironment:
    def __init__(self, level_id: int) -> None:
        self.level_id = level_id
        self.reset_seeds: list[int | None] = []

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict]:
        self.reset_seeds.append(seed)
        return np.zeros(STATE_SIZE, dtype=np.float32), {"level_id": self.level_id}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        reason = "win" if self.reset_seeds[-1] % 2 == 0 else "wall"
        info = {"reason": reason, "apples_eaten": int(reason == "win") * 2}
        return np.zeros(STATE_SIZE, dtype=np.float32), 0.0, True, False, info


class EvaluateLevelTests(unittest.TestCase):
    def test_exact_win_rate_and_seeds(self) -> None:
        env = SeedDependentEnvironment(level_id=3)
        summary = evaluate_level(FakeAgent(), env, episodes=4, base_seed=10)
        self.assertEqual(env.reset_seeds, [10, 11, 12, 13])
        self.assertEqual(summary.win_rate, 0.5)
        self.assertEqual(summary.mean_apples, 1.0)
        self.assertEqual(summary.mean_steps, 1.0)
        self.assertEqual(summary.reasons, {"wall": 2, "win": 2})
        self.assertEqual(summary.level_id, 3)

    def test_evaluate_all_levels_keeps_level_order(self) -> None:
        envs = [SeedDependentEnvironment(1), SeedDependentEnvironment(2)]
        summaries = evaluate_all_levels(FakeAgent(), envs, episodes=2, base_seed=0)
        self.assertEqual([summary.level_id for summary in summaries], [1, 2])

    def test_summarize_rejects_empty_results(self) -> None:
        with self.assertRaises(ValueError):
            summarize_results([])

    def test_summarize_single_loss(self) -> None:
        summary = summarize_results([EpisodeResult(1, -1.0, 0, 7, "body", None)])
        self.assertEqual((summary.win_rate, summary.mean_steps, summary.reasons), (0.0, 7.0, {"body": 1}))


class DecisionTests(unittest.TestCase):
    def test_should_evaluate_on_multiples(self) -> None:
        self.assertTrue(should_evaluate(50, 50))
        self.assertFalse(should_evaluate(49, 50))

    def test_first_evaluation_is_always_an_improvement(self) -> None:
        self.assertTrue(is_improvement(0.0, None))

    def test_improvement_must_be_strict(self) -> None:
        self.assertFalse(is_improvement(0.5, 0.5))
        self.assertTrue(is_improvement(0.6, 0.5))

    def test_stage_is_passed_at_target(self) -> None:
        self.assertTrue(stage_is_passed(0.9, 0.9))
        self.assertFalse(stage_is_passed(0.89, 0.9))


if __name__ == "__main__":
    unittest.main()
