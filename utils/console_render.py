from collections.abc import Iterable

from core.snake_env import SnakeEnv

EMPTY_CELL = "."
OBSTACLE_CELL = "#"
HEAD_CELL = "O"
BODY_CELL = "o"
APPLE_CELL = "@"

Grid = list[list[str]]


def empty_grid(rows: int, cols: int) -> Grid:
    return [[EMPTY_CELL] * cols for _ in range(rows)]


def paint_cells(grid: Grid, cells: Iterable[tuple[int, int]], symbol: str) -> None:
    for row, col in cells:
        grid[row][col] = symbol


def snake_cells(env: SnakeEnv) -> list[tuple[int, int]]:
    return [(int(part[0]), int(part[1])) for part in env.snake_body]


def render_board(env: SnakeEnv) -> str:
    rows, cols = env.grid_size
    grid = empty_grid(rows, cols)
    paint_cells(grid, env.obstacles, OBSTACLE_CELL)
    if env.apple_pos is not None:
        paint_cells(grid, [(int(env.apple_pos[0]), int(env.apple_pos[1]))], APPLE_CELL)
    body = snake_cells(env)
    paint_cells(grid, body[1:], BODY_CELL)
    paint_cells(grid, body[:1], HEAD_CELL)
    return "\n".join("".join(row) for row in grid)
