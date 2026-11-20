import json

import numpy as np

from config import HUNGER_FACTOR, REWARD_APPLE, REWARD_DEATH, REWARD_STEP, SNAKE_GROWS

Cell = tuple[int, int]
StepResult = tuple[float, bool, bool, str]

UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3
DIRECTION_DELTAS = {UP: (-1, 0), RIGHT: (0, 1), DOWN: (1, 0), LEFT: (0, -1)}
STRAIGHT, TURN_RIGHT, TURN_LEFT = 0, 1, 2
TURN_OFFSETS = {STRAIGHT: 0, TURN_RIGHT: 1, TURN_LEFT: 3}
REWARDS = {"apple": REWARD_APPLE, "win": REWARD_APPLE, "death": REWARD_DEATH, "move": REWARD_STEP}


def load_level(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as level_file:
        return json.load(level_file)


def turn(direction: int, relative_action: int) -> int:
    return (direction + TURN_OFFSETS[relative_action]) % len(DIRECTION_DELTAS)


def next_cell(cell: Cell, direction: int) -> Cell:
    delta = DIRECTION_DELTAS[direction]
    return cell[0] + delta[0], cell[1] + delta[1]


def reward_for(event: str) -> float:
    return REWARDS[event]


class SnakeEnv:
    def __init__(self, level_path: str) -> None:
        level = load_level(level_path)
        self.level_id: int = level["level_id"]
        self.rows, self.cols = level["grid_size"]
        self.start: Cell = tuple(level["start_position"])
        self.start_length: int = level["start_length"]
        self.apples_to_win: int = level["apples_to_win"]
        self.obstacles: set[Cell] = {tuple(cell) for cell in level["obstacles"] or []}
        self.grows: bool = level.get("snake_grows", SNAKE_GROWS)
        self.hunger_limit = int(HUNGER_FACTOR * self.rows * self.cols)
        self.rng = np.random.default_rng()
        self.reset()

    @property
    def head(self) -> Cell:
        return self.body[0]

    def reset(self, seed: int | None = None) -> None:
        self.rng = np.random.default_rng(seed)
        self.body: list[Cell] = [(self.start[0], self.start[1] - index) for index in range(self.start_length)]
        self.direction = RIGHT
        self.apples_eaten = 0
        self.steps_since_apple = 0
        self.apple = self._spawn_apple()

    def step(self, relative_action: int) -> StepResult:
        self.direction = turn(self.direction, relative_action)
        new_head = next_cell(self.head, self.direction)
        eats = new_head == self.apple
        crash = self.collision(new_head, eats)
        if crash is not None:
            return reward_for("death"), True, False, crash
        self._move_body(new_head, eats)
        if eats:
            return self._eat_apple()
        return self._advance_hunger()

    def collision(self, cell: Cell, eats: bool) -> str | None:
        if not (0 <= cell[0] < self.rows and 0 <= cell[1] < self.cols):
            return "wall"
        if cell in self.obstacles:
            return "obstacle"
        if cell in self._body_that_stays(eats):
            return "body"
        return None

    def _body_that_stays(self, eats: bool) -> list[Cell]:
        if eats and self.grows:
            return self.body
        return self.body[:-1]

    def _move_body(self, new_head: Cell, eats: bool) -> None:
        self.body.insert(0, new_head)
        if not (eats and self.grows):
            self.body.pop()

    def _eat_apple(self) -> StepResult:
        self.apples_eaten += 1
        self.steps_since_apple = 0
        self.apple = self._spawn_apple()
        if self.apples_eaten >= self.apples_to_win or self.apple is None:
            return reward_for("win"), True, False, "win"
        return reward_for("apple"), False, False, "apple"

    def _advance_hunger(self) -> StepResult:
        self.steps_since_apple += 1
        starving = self.steps_since_apple >= self.hunger_limit
        return reward_for("move"), False, starving, "starvation" if starving else "move"

    def free_cells(self) -> list[Cell]:
        occupied = self.obstacles | set(self.body)
        return [(row, col) for row in range(self.rows) for col in range(self.cols) if (row, col) not in occupied]

    def _spawn_apple(self) -> Cell | None:
        cells = self.free_cells()
        if not cells:
            return None
        return cells[int(self.rng.integers(len(cells)))]
