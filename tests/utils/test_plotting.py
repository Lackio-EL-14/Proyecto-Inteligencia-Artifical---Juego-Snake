import tempfile
import unittest
from pathlib import Path

import matplotlib
import numpy as np

from utils.plotting import (
    EvaluationPoint,
    learning_curve_path,
    moving_average,
    plot_curriculum_matrix,
    plot_eval_win_rate,
    plot_learning_curve,
)


class MovingAverageTests(unittest.TestCase):
    def test_short_series_is_returned_as_is(self) -> None:
        np.testing.assert_array_equal(moving_average([1.0, 2.0], window=3), [1.0, 2.0])

    def test_valid_window_average(self) -> None:
        np.testing.assert_allclose(moving_average([1.0, 2.0, 3.0, 4.0], window=2), [1.5, 2.5, 3.5])


class PlotFileTests(unittest.TestCase):
    def test_backend_writes_files_only(self) -> None:
        self.assertEqual(matplotlib.get_backend().lower(), "agg")

    def test_learning_curve_is_saved_in_save_dir(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = plot_learning_curve([0.0, 1.0, -1.0], [0.0, 0.5, 0.33], 2, directory)
            self.assertEqual(path, learning_curve_path(directory, 2))
            self.assertTrue(Path(path).is_file())

    def test_curriculum_matrix_and_eval_plots_are_saved(self) -> None:
        points = [EvaluationPoint(1, 50, 0.4), EvaluationPoint(1, 100, 0.9), EvaluationPoint(2, 50, 0.1)]
        with tempfile.TemporaryDirectory() as directory:
            matrix_path = plot_curriculum_matrix(((1.0, 0.2), (0.8, 0.6)), (1, 2), str(Path(directory) / "m.png"))
            eval_path = plot_eval_win_rate(points, str(Path(directory) / "plots" / "e.png"))
            self.assertTrue(Path(matrix_path).is_file())
            self.assertTrue(Path(eval_path).is_file())


if __name__ == "__main__":
    unittest.main()
