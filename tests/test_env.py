import unittest

import numpy as np

from config import REWARD_APPLE, REWARD_DEATH, REWARD_STEP
from snake_env import DOWN, RIGHT, STRAIGHT, TURN_LEFT, TURN_RIGHT, UP, SnakeEnv, turn
from state import build_state

LEVEL_1 = "levels/level_1.json"
LEVEL_3 = "levels/level_3.json"


def env_with_apple(apple: tuple[int, int], level: str = LEVEL_1) -> SnakeEnv:
    env = SnakeEnv(level)
    env.reset(0)
    env.apple = apple
    return env


class TurnTest(unittest.TestCase):
    def test_relative_turns_from_right(self) -> None:
        self.assertEqual(turn(RIGHT, STRAIGHT), RIGHT)
        self.assertEqual(turn(RIGHT, TURN_RIGHT), DOWN)
        self.assertEqual(turn(RIGHT, TURN_LEFT), UP)


class SnakeEnvTest(unittest.TestCase):
    def test_reset_places_snake_facing_right(self) -> None:
        env = env_with_apple((0, 0))
        self.assertEqual(env.body, [(4, 4), (4, 3)])
        self.assertEqual(env.direction, RIGHT)

    def test_move_gives_step_reward(self) -> None:
        env = env_with_apple((0, 0))
        self.assertEqual(env.step(STRAIGHT), (REWARD_STEP, False, False, "move"))
        self.assertEqual(env.body, [(4, 5), (4, 4)])

    def test_wall_collision_ends_episode(self) -> None:
        env = env_with_apple((0, 0))
        for _ in range(3):
            env.step(STRAIGHT)
        self.assertEqual(env.step(STRAIGHT), (REWARD_DEATH, True, False, "wall"))

    def test_obstacle_collision_ends_episode(self) -> None:
        env = env_with_apple((0, 0), LEVEL_3)
        env.body = [(4, 3), (4, 2)]
        env.direction = UP
        self.assertEqual(env.step(STRAIGHT)[3], "obstacle")

    def test_eating_without_growth_keeps_length(self) -> None:
        env = env_with_apple((4, 5))
        env.apples_to_win = 2
        self.assertEqual(env.step(STRAIGHT), (REWARD_APPLE, False, False, "apple"))
        self.assertEqual(len(env.body), 2)

    def test_eating_with_growth_adds_segment(self) -> None:
        env = env_with_apple((4, 5))
        env.apples_to_win, env.grows = 2, True
        env.step(STRAIGHT)
        self.assertEqual(len(env.body), 3)

    def test_last_apple_wins(self) -> None:
        env = env_with_apple((4, 5))
        self.assertEqual(env.step(STRAIGHT), (REWARD_APPLE, True, False, "win"))

    def test_hunger_truncates_without_terminating(self) -> None:
        env = env_with_apple((0, 0))
        env.hunger_limit = 1
        self.assertEqual(env.step(STRAIGHT), (REWARD_STEP, False, True, "starvation"))


class StateTest(unittest.TestCase):
    def test_apple_ahead_marks_straight_as_closer(self) -> None:
        state = build_state(env_with_apple((4, 7)))
        np.testing.assert_array_equal(state[:6], [0, 1, 0, 0, 0, 0])
        self.assertAlmostEqual(state[6], 3 / 64)

    def test_wall_marks_move_as_blocked(self) -> None:
        env = env_with_apple((0, 0))
        env.body = [(4, 7), (4, 6)]
        np.testing.assert_array_equal(build_state(env)[0:2], [1, 0])

    def test_closer_follows_path_around_wall(self) -> None:
        env = env_with_apple((2, 0), LEVEL_3)
        env.body = [(4, 1), (4, 0)]
        np.testing.assert_array_equal(build_state(env)[:6], [0, 1, 0, 0, 1, 0])


if __name__ == "__main__":
    unittest.main()
