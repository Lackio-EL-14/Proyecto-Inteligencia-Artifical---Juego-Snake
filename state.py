from collections import deque

import numpy as np

from config import USE_APPLE_DISTANCE
from snake_env import DIRECTION_DELTAS, STRAIGHT, TURN_LEFT, TURN_RIGHT, Cell, SnakeEnv, next_cell, turn

RELATIVE_ACTIONS = (STRAIGHT, TURN_RIGHT, TURN_LEFT)
FEATURES_PER_MOVE = 2
MOVE_FEATURE_COUNT = FEATURES_PER_MOVE * len(RELATIVE_ACTIONS)


def neighbors(cell: Cell) -> list[Cell]:
    return [next_cell(cell, direction) for direction in DIRECTION_DELTAS]


def is_walkable(env: SnakeEnv, cell: Cell) -> bool:
    inside = 0 <= cell[0] < env.rows and 0 <= cell[1] < env.cols
    return inside and cell not in env.obstacles and cell not in env.body[1:-1]


def bfs_distances(env: SnakeEnv, source: Cell) -> dict[Cell, int]:
    distances = {source: 0}
    frontier = deque([source])
    while frontier:
        cell = frontier.popleft()
        for neighbor in neighbors(cell):
            if neighbor not in distances and is_walkable(env, neighbor):
                distances[neighbor] = distances[cell] + 1
                frontier.append(neighbor)
    return distances


def relative_cells(env: SnakeEnv) -> list[Cell]:
    return [next_cell(env.head, turn(env.direction, action)) for action in RELATIVE_ACTIONS]


def is_blocked(env: SnakeEnv, cell: Cell) -> bool:
    return env.collision(cell, cell == env.apple) is not None


def gets_closer(distances: dict[Cell, int], cell: Cell, head: Cell) -> bool:
    return distances.get(cell, np.inf) < distances.get(head, np.inf)


def move_features(env: SnakeEnv, distances: dict[Cell, int], cell: Cell) -> list[float]:
    blocked = is_blocked(env, cell)
    closer = not blocked and gets_closer(distances, cell, env.head)
    return [float(blocked), float(closer)]


def walkable_cell_count(env: SnakeEnv) -> int:
    return env.rows * env.cols - len(env.obstacles)


def apple_distance(env: SnakeEnv, distances: dict[Cell, int]) -> float:
    total = walkable_cell_count(env)
    return min(distances.get(env.head, total), total) / total


def build_state(env: SnakeEnv) -> np.ndarray:
    distances = {} if env.apple is None else bfs_distances(env, env.apple)
    features: list[float] = []
    for cell in relative_cells(env):
        features.extend(move_features(env, distances, cell))
    if USE_APPLE_DISTANCE:
        features.append(apple_distance(env, distances))
    return np.asarray(features, dtype=np.float32)


def move_view(state: np.ndarray) -> np.ndarray:
    return state[:MOVE_FEATURE_COUNT].reshape(len(RELATIVE_ACTIONS), FEATURES_PER_MOVE)
