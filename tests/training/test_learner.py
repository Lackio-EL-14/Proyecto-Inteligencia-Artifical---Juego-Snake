import unittest

from tests.training.fakes import FakeAgent, FakeMemory
from training.learner import Learner, should_sync_target, should_train
from training.training_spec import LearnerSpec

SPEC = LearnerSpec(batch_size=4, warmup_steps=10, train_every=4, target_update_every=3)


class ShouldTrainTests(unittest.TestCase):
    def test_false_before_warmup(self) -> None:
        self.assertFalse(should_train(env_steps=8, memory_size=9, spec=SPEC))

    def test_false_between_multiples_of_train_every(self) -> None:
        self.assertFalse(should_train(env_steps=13, memory_size=100, spec=SPEC))

    def test_true_on_exact_multiple_after_warmup(self) -> None:
        self.assertTrue(should_train(env_steps=12, memory_size=10, spec=SPEC))

    def test_batch_size_also_bounds_the_warmup(self) -> None:
        spec = LearnerSpec(batch_size=64, warmup_steps=0, train_every=1, target_update_every=1)
        self.assertFalse(should_train(env_steps=1, memory_size=63, spec=spec))


class ShouldSyncTargetTests(unittest.TestCase):
    def test_multiples_sync(self) -> None:
        self.assertTrue(should_sync_target(6, 3))

    def test_non_multiples_do_not_sync(self) -> None:
        self.assertFalse(should_sync_target(7, 3))


class LearnerTests(unittest.TestCase):
    def test_returns_none_when_not_training(self) -> None:
        learner = Learner(FakeAgent(), FakeMemory(reported_size=0), SPEC)
        self.assertIsNone(learner.after_env_step())
        self.assertEqual(learner.env_steps, 1)

    def test_trains_every_train_every_steps(self) -> None:
        agent = FakeAgent()
        learner = Learner(agent, FakeMemory(reported_size=100), SPEC)
        results = [learner.after_env_step() for _ in range(12)]
        self.assertEqual(agent.train_calls, 3)
        self.assertEqual([index for index, result in enumerate(results) if result is not None], [3, 7, 11])

    def test_syncs_target_exactly_every_target_update_every_gradient_steps(self) -> None:
        agent = FakeAgent()
        spec = LearnerSpec(batch_size=2, warmup_steps=0, train_every=1, target_update_every=3)
        learner = Learner(agent, FakeMemory(reported_size=10), spec)
        for _ in range(9):
            learner.after_env_step()
        self.assertEqual(learner.gradient_steps, 9)
        self.assertEqual(agent.sync_calls, 3)


if __name__ == "__main__":
    unittest.main()
