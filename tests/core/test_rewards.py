import unittest

from core.rewards import UNREACHABLE_POTENTIAL, RewardSpec, hunger_limit, move_reward, path_potential, potential_shaping

SPEC = RewardSpec(apple=1.0, death=-1.0, step=-0.01, shaping_gamma=0.9)


class PotentialTests(unittest.TestCase):
    def test_potential_is_negative_normalized_distance(self):
        self.assertAlmostEqual(path_potential(5, 20), -0.25)

    def test_unreachable_apple_has_minimum_potential(self):
        self.assertEqual(path_potential(None, 20), UNREACHABLE_POTENTIAL)

    def test_shaping_formula(self):
        self.assertAlmostEqual(potential_shaping(-0.5, -0.4, 0.9), 0.9 * -0.4 + 0.5)

    def test_closed_loop_telescopes_to_zero_with_unit_gamma(self):
        potentials = [-0.3, -0.2, -0.3, -0.4, -0.3]
        total = sum(potential_shaping(a, b, 1.0) for a, b in zip(potentials, potentials[1:]))
        self.assertAlmostEqual(total, 0.0)

    def test_move_reward_adds_step_cost(self):
        self.assertAlmostEqual(move_reward(SPEC, -0.5, -0.4), -0.01 + 0.9 * -0.4 + 0.5)


class HungerLimitTests(unittest.TestCase):
    def test_disabled_without_factor(self):
        self.assertIsNone(hunger_limit((10, 10), None))

    def test_scales_with_board_cells(self):
        self.assertEqual(hunger_limit((10, 12), 1.0), 120)

    def test_never_below_one_step(self):
        self.assertEqual(hunger_limit((2, 2), 0.01), 1)


if __name__ == "__main__":
    unittest.main()
