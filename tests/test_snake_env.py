import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from core.snake_env import SnakeEnv
from core.movements import LEFT, RIGHT, UP
from training import train as training_module


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class SnakeEnvTests(unittest.TestCase):
    def make_env(self, config, max_steps=None):
        temporary_file = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", encoding="utf-8", delete=False
        )
        with temporary_file:
            json.dump(config, temporary_file)
        self.addCleanup(Path(temporary_file.name).unlink, missing_ok=True)
        return SnakeEnv(temporary_file.name, max_steps=max_steps)

    def base_config(self):
        return {
            "level_id": 99,
            "grid_size": [6, 6],
            "start_position": [3, 3],
            "start_length": 2,
            "apples_to_win": 2,
            "obstacles": [],
        }

    def test_existing_level_files_load_and_return_fixed_observation(self):
        level_paths = sorted((PROJECT_ROOT / "levels").glob("level_*.json"))
        self.assertEqual(len(level_paths), 4)

        for level_path in level_paths:
            with self.subTest(level=level_path.name):
                env = SnakeEnv(level_path)
                observation, info = env.reset(seed=13)
                self.assertEqual(observation.shape, (11,))
                self.assertEqual(observation.dtype, np.float32)
                self.assertEqual(info["level_id"], env.config["level_id"])

    def test_reset_with_same_seed_repeats_apple_position(self):
        env = self.make_env(self.base_config())
        env.reset(seed=21)
        first_apple = env.apple_pos.copy()
        env.reset(seed=21)
        np.testing.assert_array_equal(env.apple_pos, first_apple)

    def test_apple_is_not_on_snake_or_obstacle(self):
        config = self.base_config()
        config["obstacles"] = [[0, 0], [5, 5]]
        env = self.make_env(config)
        env.reset(seed=17)

        self.assertNotIn(tuple(env.apple_pos), env.obstacles)
        self.assertFalse(any(np.array_equal(env.apple_pos, part) for part in env.snake_body))

    def test_rejects_invalid_grid_size(self):
        config = self.base_config()
        config["grid_size"] = [0, 6]
        with self.assertRaisesRegex(ValueError, "grid_size"):
            self.make_env(config)

    def test_rejects_start_position_outside_board(self):
        config = self.base_config()
        config["start_position"] = [6, 3]
        with self.assertRaisesRegex(ValueError, "start_position"):
            self.make_env(config)

    def test_rejects_initial_snake_outside_board(self):
        config = self.base_config()
        config["start_position"] = [3, 0]
        with self.assertRaisesRegex(ValueError, "no cabe"):
            self.make_env(config)

    def test_rejects_obstacle_overlapping_initial_snake(self):
        config = self.base_config()
        config["obstacles"] = [[3, 2]]
        with self.assertRaisesRegex(ValueError, "superpone"):
            self.make_env(config)

    def test_collision_terminates_episode(self):
        config = self.base_config()
        config["start_position"] = [0, 2]
        env = self.make_env(config)
        env.reset(seed=5)

        observation, reward, terminated, truncated, info = env.step(UP)

        self.assertEqual(observation.shape, (11,))
        self.assertEqual(reward, -1.0)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["reason"], "wall")

    def test_eating_target_apple_wins_and_grows_snake(self):
        config = self.base_config()
        config["apples_to_win"] = 1
        env = self.make_env(config)
        env.reset(seed=5)
        env.apple_pos = np.array([3, 4])

        _, reward, terminated, truncated, info = env.step(RIGHT)

        self.assertEqual(reward, 1.0)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["reason"], "win")
        self.assertEqual(len(env.snake_body), 3)

    def test_reverse_action_is_ignored(self):
        env = self.make_env(self.base_config())
        env.reset(seed=3)
        env.apple_pos = np.array([0, 0])

        _, _, terminated, truncated, _ = env.step(LEFT)

        np.testing.assert_array_equal(env.snake_body[0], [3, 4])
        self.assertEqual(env.direction, RIGHT)
        self.assertFalse(terminated)
        self.assertFalse(truncated)

    def test_entering_vacating_tail_cell_is_safe(self):
        env = self.make_env(self.base_config())
        env.reset(seed=7)
        env.snake_body = [
            np.array([1, 1]),
            np.array([2, 1]),
            np.array([2, 2]),
            np.array([1, 2]),
        ]
        env.direction = RIGHT
        env.apple_pos = np.array([5, 5])

        _, _, terminated, truncated, info = env.step(RIGHT)

        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["reason"], "moved")
        np.testing.assert_array_equal(env.snake_body[0], [1, 2])

    def test_max_steps_truncates_without_marking_terminal_for_replay(self):
        env = self.make_env(self.base_config(), max_steps=1)
        env.reset(seed=11)
        env.apple_pos = np.array([0, 0])

        _, _, terminated, truncated, info = env.step(UP)

        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertEqual(info["reason"], "time_limit")
        self.assertTrue(env.done)

    def test_step_after_episode_end_raises(self):
        config = self.base_config()
        config["start_position"] = [0, 2]
        env = self.make_env(config)
        env.reset(seed=5)
        env.step(UP)

        with self.assertRaises(RuntimeError):
            env.step(RIGHT)

    def test_training_consumes_new_api_and_preserves_terminal_mask(self):
        class FakeEnvironment:
            def __init__(self, level_path, max_steps):
                self.max_steps = max_steps

            def reset(self):
                return np.zeros(11, dtype=np.float32), {"level_id": 1}

            def step(self, action):
                return np.ones(11, dtype=np.float32), 1.0, True, False, {"reason": "win"}

        class FakeMemory:
            def __init__(self):
                self.transitions = []

            def remember(self, *transition):
                self.transitions.append(transition)

            def __len__(self):
                return 0

        class FakeModel:
            def save_weights(self, path):
                self.saved_path = path

        class FakeAgent:
            def __init__(self):
                self.model = FakeModel()

            def act(self, state, epsilon):
                return RIGHT

        memory = FakeMemory()
        agent = FakeAgent()

        with (
            patch.object(training_module, "WINDOW_SIZE", 1),
            patch.object(training_module, "TARGET_WIN_RATE", 1.0),
            patch.object(training_module, "MAX_EPISODES_PER_STAGE", 1),
            patch.object(training_module, "MODELS_DIR", tempfile.gettempdir()),
            patch.object(training_module, "WEIGHTS_FILE", "unused.weights.h5"),
            patch.object(training_module, "plot_learning_curve"),
        ):
            training_module.train_by_stage(
                agent=agent,
                memory=memory,
                env_class=FakeEnvironment,
                level_paths=["fake-level.json"],
            )

        self.assertEqual(len(memory.transitions), 1)
        self.assertTrue(memory.transitions[0][4])
        self.assertEqual(agent.model.saved_path, "unused.weights.h5")

    def test_training_does_not_mark_truncation_as_terminal(self):
        class FakeEnvironment:
            def __init__(self, level_path, max_steps):
                self.max_steps = max_steps

            def reset(self):
                return np.zeros(11, dtype=np.float32), {}

            def step(self, action):
                return np.ones(11, dtype=np.float32), -0.1, False, True, {
                    "reason": "time_limit"
                }

        class FakeMemory:
            def __init__(self):
                self.transitions = []

            def remember(self, *transition):
                self.transitions.append(transition)

            def __len__(self):
                return 0

        class FakeModel:
            def save_weights(self, path):
                self.saved_path = path

        class FakeAgent:
            def __init__(self):
                self.model = FakeModel()

            def act(self, state, epsilon):
                return RIGHT

        memory = FakeMemory()
        agent = FakeAgent()

        with (
            patch.object(training_module, "WINDOW_SIZE", 1),
            patch.object(training_module, "TARGET_WIN_RATE", 1.0),
            patch.object(training_module, "MAX_EPISODES_PER_STAGE", 1),
            patch.object(training_module, "MODELS_DIR", tempfile.gettempdir()),
            patch.object(training_module, "WEIGHTS_FILE", "unused.weights.h5"),
            patch.object(training_module, "plot_learning_curve"),
        ):
            training_module.train_by_stage(
                agent=agent,
                memory=memory,
                env_class=FakeEnvironment,
                level_paths=["fake-level.json"],
            )

        self.assertEqual(len(memory.transitions), 1)
        self.assertFalse(memory.transitions[0][4])


if __name__ == "__main__":
    unittest.main()
