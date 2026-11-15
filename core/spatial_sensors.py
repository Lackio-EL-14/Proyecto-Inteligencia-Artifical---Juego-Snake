from collections import deque
from dataclasses import dataclass

import numpy as np

from core.movements import ACTION_DELTAS

Cell = tuple[int, int]
GridSize = tuple[int, int]

NO_PATH_DISTANCE = 1.0
BLOCKED_AREA = 0.0
RIGHT_TURN = 1
LEFT_TURN = 3
DIRECTION_COUNT = 4
FEATURES_PER_MOVE = 3
RELATIVE_MOVES = 3
SPATIAL_FEATURE_COUNT = FEATURES_PER_MOVE * RELATIVE_MOVES


@dataclass(frozen=True)
class SpatialContext:
    grid_size: GridSize
    blocked: frozenset[Cell]
    apple_distances: dict[Cell, int]
    free_cells: int


def as_cell(position: np.ndarray) -> Cell:
    return int(position[0]), int(position[1])


def blocked_cells(obstacles: set[Cell], snake_body: list[np.ndarray]) -> frozenset[Cell]:
    body_without_tail = {as_cell(part) for part in snake_body[:-1]}
    return frozenset(obstacles | body_without_tail)


def free_cell_count(grid_size: GridSize, obstacles: set[Cell]) -> int:
    return grid_size[0] * grid_size[1] - len(obstacles)


def is_open(cell: Cell, grid_size: GridSize, blocked: frozenset[Cell]) -> bool:
    inside = 0 <= cell[0] < grid_size[0] and 0 <= cell[1] < grid_size[1]
    return inside and cell not in blocked


def neighbor_cells(cell: Cell) -> list[Cell]:
    return [(cell[0] + int(delta[0]), cell[1] + int(delta[1])) for delta in ACTION_DELTAS.values()]


def bfs_distances(grid_size: GridSize, blocked: frozenset[Cell], source: Cell) -> dict[Cell, int]:
    if not is_open(source, grid_size, blocked):
        return {}
    distances = {source: 0}
    frontier = deque([source])
    while frontier:
        cell = frontier.popleft()
        for neighbor in neighbor_cells(cell):
            if neighbor not in distances and is_open(neighbor, grid_size, blocked):
                distances[neighbor] = distances[cell] + 1
                frontier.append(neighbor)
    return distances


def flood_fill_size(grid_size: GridSize, blocked: frozenset[Cell], source: Cell) -> int:
    return len(bfs_distances(grid_size, blocked, source))


def relative_move_cells(head: Cell, direction: int) -> tuple[Cell, Cell, Cell]:
    turns = (0, RIGHT_TURN, LEFT_TURN)
    moves = [ACTION_DELTAS[(direction + turn) % DIRECTION_COUNT] for turn in turns]
    straight, right, left = [(head[0] + int(delta[0]), head[1] + int(delta[1])) for delta in moves]
    return straight, right, left


def build_spatial_context(grid_size: GridSize, obstacles: set[Cell], snake_body: list[np.ndarray], apple: Cell | None) -> SpatialContext:
    blocked = blocked_cells(obstacles, snake_body)
    apple_distances = {} if apple is None else bfs_distances(grid_size, blocked, apple)
    return SpatialContext(grid_size, blocked, apple_distances, free_cell_count(grid_size, obstacles))


def move_features(context: SpatialContext, cell: Cell) -> tuple[float, float, float]:
    if not is_open(cell, context.grid_size, context.blocked):
        return NO_PATH_DISTANCE, BLOCKED_AREA, 0.0
    area = flood_fill_size(context.grid_size, context.blocked, cell) / context.free_cells
    if cell not in context.apple_distances:
        return NO_PATH_DISTANCE, area, 0.0
    return context.apple_distances[cell] / context.free_cells, area, 1.0


def spatial_features(context: SpatialContext, head: Cell, direction: int) -> list[float]:
    features: list[float] = []
    for cell in relative_move_cells(head, direction):
        features.extend(move_features(context, cell))
    return features


def head_path_distance(context: SpatialContext, head: Cell) -> int | None:
    if head in context.apple_distances:
        return context.apple_distances[head]
    reachable = [context.apple_distances[cell] for cell in neighbor_cells(head) if cell in context.apple_distances]
    return min(reachable) + 1 if reachable else None
