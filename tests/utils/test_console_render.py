import unittest
from types import SimpleNamespace

import numpy as np

from utils.console_render import render_board


class RenderBoardTests(unittest.TestCase):
    def test_small_board_exact_output(self) -> None:
        env = SimpleNamespace(
            grid_size=(3, 4),
            obstacles={(0, 0), (2, 3)},
            apple_pos=np.array([0, 3]),
            snake_body=[np.array([1, 2]), np.array([1, 1])],
        )
        self.assertEqual(render_board(env), "#..@\n.oO.\n...#")

    def test_board_without_apple(self) -> None:
        env = SimpleNamespace(grid_size=(1, 3), obstacles=set(), apple_pos=None, snake_body=[np.array([0, 0])])
        self.assertEqual(render_board(env), "O..")


if __name__ == "__main__":
    unittest.main()
