import unittest

import numpy as np

from core.movements import DOWN, LEFT, RIGHT, UP
from core.spatial_sensors import (
    SPATIAL_FEATURE_COUNT,
    blocked_cells,
    bfs_distances,
    build_spatial_context,
    flood_fill_size,
    head_path_distance,
    move_features,
    relative_move_cells,
    spatial_features,
)

GRID = (3, 3)
NO_OBSTACLES: set[tuple[int, int]] = set()


def body(*cells: tuple[int, int]) -> list[np.ndarray]:
    return [np.array(cell) for cell in cells]


class BlockedCellsTests(unittest.TestCase):
    def test_includes_obstacles_and_body_but_not_tail(self):
        blocked = blocked_cells({(0, 0)}, body((1, 1), (1, 2), (2, 2)))
        self.assertEqual(blocked, frozenset({(0, 0), (1, 1), (1, 2)}))


class BfsTests(unittest.TestCase):
    def test_distances_go_around_a_wall(self):
        blocked = frozenset({(0, 1), (1, 1)})
        distances = bfs_distances(GRID, blocked, (0, 0))
        self.assertEqual(distances[(0, 2)], 6)
        self.assertNotIn((0, 1), distances)

    def test_blocked_source_has_no_distances(self):
        self.assertEqual(bfs_distances(GRID, frozenset({(0, 0)}), (0, 0)), {})

    def test_flood_fill_counts_only_the_enclosed_region(self):
        blocked = frozenset({(0, 1), (1, 1), (2, 1)})
        self.assertEqual(flood_fill_size(GRID, blocked, (0, 0)), 3)


class RelativeMoveTests(unittest.TestCase):
    def test_straight_right_left_for_each_direction(self):
        self.assertEqual(relative_move_cells((1, 1), UP), ((0, 1), (1, 2), (1, 0)))
        self.assertEqual(relative_move_cells((1, 1), RIGHT), ((1, 2), (2, 1), (0, 1)))
        self.assertEqual(relative_move_cells((1, 1), DOWN), ((2, 1), (1, 0), (1, 2)))
        self.assertEqual(relative_move_cells((1, 1), LEFT), ((1, 0), (0, 1), (2, 1)))


class FeatureTests(unittest.TestCase):
    def test_colliding_move_reports_no_path_and_no_area(self):
        context = build_spatial_context(GRID, NO_OBSTACLES, body((0, 0), (1, 0)), (2, 2))
        self.assertEqual(move_features(context, (-1, 0)), (1.0, 0.0, 0.0))

    def test_open_move_reports_normalized_distance_area_and_reachability(self):
        context = build_spatial_context(GRID, NO_OBSTACLES, body((0, 0), (1, 0)), (2, 2))
        distance, area, reachable = move_features(context, (0, 1))
        self.assertAlmostEqual(distance, 3 / 9)
        self.assertAlmostEqual(area, 8 / 9)
        self.assertEqual(reachable, 1.0)

    def test_unreachable_apple_keeps_area_but_no_path(self):
        context = build_spatial_context(GRID, {(0, 1), (1, 1), (2, 1)}, body((0, 0), (1, 0)), (2, 2))
        distance, area, reachable = move_features(context, (1, 0))
        self.assertEqual((distance, reachable), (1.0, 0.0))
        self.assertAlmostEqual(area, 2 / 6)

    def test_spatial_features_have_three_values_per_relative_move(self):
        context = build_spatial_context(GRID, NO_OBSTACLES, body((1, 1), (1, 0)), (0, 2))
        self.assertEqual(len(spatial_features(context, (1, 1), RIGHT)), SPATIAL_FEATURE_COUNT)

    def test_missing_apple_marks_every_move_unreachable(self):
        context = build_spatial_context(GRID, NO_OBSTACLES, body((1, 1), (1, 0)), None)
        features = spatial_features(context, (1, 1), RIGHT)
        self.assertEqual(features[2::3], [0.0, 0.0, 0.0])


class HeadDistanceTests(unittest.TestCase):
    def test_blocked_head_uses_best_neighbor_plus_one(self):
        context = build_spatial_context(GRID, NO_OBSTACLES, body((0, 0), (1, 0)), (0, 2))
        self.assertEqual(head_path_distance(context, (0, 0)), 2)

    def test_single_cell_snake_uses_its_own_distance(self):
        context = build_spatial_context(GRID, NO_OBSTACLES, body((0, 0)), (2, 2))
        self.assertEqual(head_path_distance(context, (0, 0)), 4)

    def test_enclosed_head_has_no_distance(self):
        context = build_spatial_context(GRID, {(0, 1), (1, 1)}, body((0, 0), (1, 0), (2, 0)), (2, 2))
        self.assertIsNone(head_path_distance(context, (0, 0)))


if __name__ == "__main__":
    unittest.main()
