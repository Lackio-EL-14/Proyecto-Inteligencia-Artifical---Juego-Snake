import unittest

from tests.training.fakes import (
    FAKE_LEVEL_ID,
    TIME_LIMIT_OUTCOME,
    WIN_OUTCOME,
    FakeAgent,
    FakeEnvironment,
    FakeMemory,
)
from training.episode_runner import EpisodeResult, StepHook, is_win, mean_loss, run_greedy_episode, run_training_episode, watch_greedy_episode
from training.learner import Learner
from training.training_spec import LearnerSpec

ALWAYS_TRAIN = LearnerSpec(batch_size=2, warmup_steps=0, train_every=1, target_update_every=100)


def make_hook(agent: FakeAgent, memory: FakeMemory) -> StepHook:
    return StepHook(memory=memory, learner=Learner(agent, memory, ALWAYS_TRAIN))


class TrainingEpisodeTests(unittest.TestCase):
    def test_terminal_transition_is_stored_as_done(self) -> None:
        agent, memory = FakeAgent(), FakeMemory()
        run_training_episode(FakeEnvironment(WIN_OUTCOME), agent, 0.0, make_hook(agent, memory))
        self.assertTrue(memory.transitions[-1][4])

    def test_truncated_transition_is_not_stored_as_done(self) -> None:
        agent, memory = FakeAgent(), FakeMemory()
        run_training_episode(FakeEnvironment(TIME_LIMIT_OUTCOME), agent, 0.0, make_hook(agent, memory))
        self.assertFalse(memory.transitions[-1][4])

    def test_result_accumulates_reward_steps_and_reason(self) -> None:
        agent, memory = FakeAgent(), FakeMemory()
        result = run_training_episode(FakeEnvironment(WIN_OUTCOME, steps_to_finish=3), agent, 0.0, make_hook(agent, memory))
        self.assertEqual((result.steps, result.reason, result.level_id), (3, "win", FAKE_LEVEL_ID))
        self.assertAlmostEqual(result.total_reward, 1.0)
        self.assertEqual(len(memory.transitions), 3)

    def test_mean_loss_averages_once_per_episode(self) -> None:
        agent, memory = FakeAgent(), FakeMemory(reported_size=10)
        result = run_training_episode(FakeEnvironment(WIN_OUTCOME, steps_to_finish=3), agent, 0.0, make_hook(agent, memory))
        self.assertAlmostEqual(result.mean_loss, 2.0)

    def test_mean_loss_is_none_without_training(self) -> None:
        self.assertIsNone(mean_loss([]))


class GreedyEpisodeTests(unittest.TestCase):
    def test_greedy_episode_uses_given_seed_and_no_memory(self) -> None:
        env = FakeEnvironment(WIN_OUTCOME, steps_to_finish=2)
        result = run_greedy_episode(env, FakeAgent(), seed=33)
        self.assertEqual(env.reset_seeds, [33])
        self.assertTrue(is_win(result))

    def test_watch_calls_observer_on_reset_and_each_step(self) -> None:
        frames = []
        watch_greedy_episode(FakeEnvironment(WIN_OUTCOME, steps_to_finish=4), FakeAgent(), 1, frames.append)
        self.assertEqual(len(frames), 5)

    def test_is_win_only_for_win_reason(self) -> None:
        self.assertFalse(is_win(EpisodeResult(1, 0.0, 0, 1, "wall", None)))


if __name__ == "__main__":
    unittest.main()
