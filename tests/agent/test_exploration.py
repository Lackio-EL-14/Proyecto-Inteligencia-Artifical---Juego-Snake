import unittest

from agent.exploration import EpsilonSchedule, decay_epsilon


class DecayEpsilonTests(unittest.TestCase):
    def test_decay_multiplies_by_decay(self) -> None:
        self.assertAlmostEqual(decay_epsilon(1.0, 0.5, 0.1), 0.5)

    def test_decay_never_goes_below_minimum(self) -> None:
        self.assertAlmostEqual(decay_epsilon(0.15, 0.5, 0.1), 0.1)


class EpsilonScheduleTests(unittest.TestCase):
    def make_schedule(self) -> EpsilonSchedule:
        return EpsilonSchedule(start=1.0, stage_start=0.5, minimum=0.05, decay=0.5)

    def test_value_starts_at_start(self) -> None:
        self.assertEqual(self.make_schedule().value, 1.0)

    def test_epsilon_schedule_decays_to_minimum_and_restarts(self) -> None:
        schedule = self.make_schedule()
        for _ in range(20):
            schedule.step()
        self.assertAlmostEqual(schedule.value, 0.05)
        schedule.restart_for_stage()
        self.assertAlmostEqual(schedule.value, 0.5)

    def test_restart_never_lowers_a_higher_value(self) -> None:
        schedule = self.make_schedule()
        schedule.restart_for_stage()
        self.assertEqual(schedule.value, 1.0)


if __name__ == "__main__":
    unittest.main()
