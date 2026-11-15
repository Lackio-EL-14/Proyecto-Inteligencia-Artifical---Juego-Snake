import json

import numpy as np

from core.movements import DOWN, LEFT, RIGHT, UP, get_next_position
from core.rewards import RewardSpec, hunger_limit, move_reward, path_potential
from core.spatial_sensors import (
    SpatialContext,
    as_cell,
    build_spatial_context,
    head_path_distance,
    is_open,
    relative_move_cells,
    spatial_features,
)
from utils.config import APPLE_REWARD, DEATH_REWARD, GAMMA, STEP_REWARD

DIRECTION_COUNT = 4
StepResult = tuple[np.ndarray, float, bool, bool, dict]


def default_reward_spec() -> RewardSpec:
    return RewardSpec(apple=APPLE_REWARD, death=DEATH_REWARD, step=STEP_REWARD, shaping_gamma=GAMMA)


def direction_one_hot(direction: int) -> list[float]:
    return [float(direction == candidate) for candidate in (UP, RIGHT, DOWN, LEFT)]


def apple_direction(head: np.ndarray, apple: np.ndarray | None) -> list[float]:
    if apple is None:
        return [0.0, 0.0, 0.0, 0.0]
    return [float(apple[0] < head[0]), float(apple[1] > head[1]), float(apple[0] > head[0]), float(apple[1] < head[1])]


class SnakeEnv:
    def __init__(self, level_config_path: str, max_steps: int | None = None, reward_spec: RewardSpec | None = None, hunger_factor: float | None = None) -> None:
        self.config = self._load_config(level_config_path)
        self.grid_size = self._validate_grid_size(self.config.get("grid_size"))
        self.start_position = self._validate_position(self.config.get("start_position"), "start_position", within_grid=True)
        self.start_length = self._validate_positive_int(self.config.get("start_length", 2), "start_length")
        self.apples_to_win = self._validate_positive_int(self.config.get("apples_to_win"), "apples_to_win")
        self.obstacles = self._validate_obstacles(self.config.get("obstacles"))
        self.max_steps = self._validate_positive_int(max_steps, "max_steps", allow_none=True)
        self._validate_initial_snake()
        self._validate_capacity()
        self.reward_spec = reward_spec or default_reward_spec()
        self.hunger_limit = hunger_limit(self.grid_size, self._validate_hunger_factor(hunger_factor))
        self.np_random = np.random.default_rng()
        self.snake_body: list[np.ndarray] = []
        self.apple_pos: np.ndarray | None = None
        self.apples_eaten = 0
        self.direction = RIGHT
        self.steps = 0
        self.steps_since_apple = 0
        self.potential = 0.0
        self.done = False

    def _load_config(self, level_config_path: str) -> dict:
        with open(level_config_path, "r", encoding="utf-8") as level_file:
            config = json.load(level_file)
        if not isinstance(config, dict):
            raise ValueError("La configuración del nivel debe ser un objeto JSON.")
        return config

    def _validate_capacity(self) -> None:
        available_cells = self.grid_size[0] * self.grid_size[1]
        available_cells -= len(self.obstacles) + self.start_length
        if available_cells <= 0:
            raise ValueError("El nivel no deja casillas libres para generar una manzana.")
        if self.apples_to_win > available_cells:
            raise ValueError("'apples_to_win' supera la capacidad libre del tablero.")

    def _validate_hunger_factor(self, hunger_factor: float | None) -> float | None:
        if hunger_factor is None:
            return None
        if isinstance(hunger_factor, bool) or not isinstance(hunger_factor, (int, float)) or hunger_factor <= 0:
            raise ValueError("'hunger_factor' debe ser None o un número positivo.")
        return float(hunger_factor)

    def _validate_grid_size(self, grid_size: object) -> tuple[int, int]:
        if not isinstance(grid_size, list) or len(grid_size) != 2:
            raise ValueError("'grid_size' debe ser una lista [filas, columnas].")
        return tuple(
            self._validate_positive_int(size, f"grid_size[{index}]")
            for index, size in enumerate(grid_size)
        )

    def _validate_positive_int(self, value: object, field_name: str, allow_none: bool = False) -> int | None:
        if allow_none and value is None:
            return None
        if type(value) is not int or value <= 0:
            if allow_none:
                raise ValueError(f"'{field_name}' debe ser None o un entero positivo.")
            raise ValueError(f"'{field_name}' debe ser un entero positivo.")
        return value

    def _validate_position(self, position: object, field_name: str, within_grid: bool = False) -> tuple[int, int]:
        if not isinstance(position, list) or len(position) != 2:
            raise ValueError(f"'{field_name}' debe ser una lista [fila, columna].")
        if any(type(value) is not int for value in position):
            raise ValueError(f"'{field_name}' debe contener coordenadas enteras.")

        row, col = position
        if within_grid and not (0 <= row < self.grid_size[0] and 0 <= col < self.grid_size[1]):
            raise ValueError(
                f"'{field_name}' {position} está fuera del tablero "
                f"de tamaño {list(self.grid_size)}."
            )
        return row, col

    def _validate_obstacles(self, obstacles: object) -> set[tuple[int, int]]:
        if obstacles is None:
            return set()
        if not isinstance(obstacles, list):
            raise ValueError("'obstacles' debe ser una lista de coordenadas o null.")

        validated = set()
        for index, obstacle in enumerate(obstacles):
            position = self._validate_position(obstacle, f"obstacles[{index}]", within_grid=True)
            if position in validated:
                raise ValueError(f"El obstáculo {position} está duplicado.")
            validated.add(position)
        return validated

    def _validate_initial_snake(self) -> None:
        row, col = self.start_position
        if col - self.start_length + 1 < 0:
            raise ValueError(
                "La serpiente inicial no cabe en el tablero mirando en direccion la derecha."
            )

        initial_body = {(row, col - index) for index in range(self.start_length)}
        overlap = initial_body & self.obstacles
        if overlap:
            raise ValueError(
                f"La serpiente inicial se superpone con los obstáculos que se encuentran {sorted(overlap)}."
            )

    def _validate_action(self, action: object) -> int:
        if (
            not isinstance(action, (int, np.integer))
            or isinstance(action, (bool, np.bool_))
            or action not in (UP, RIGHT, DOWN, LEFT)
        ):
            raise ValueError("'action' debe ser un movimiento valido como UP, RIGHT, DOWN o LEFT.")
        return int(action)

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        if seed is not None:
            if type(seed) is not int:
                raise ValueError("'seed' debe ser un entero o None.")
            self.np_random = np.random.default_rng(seed)
        if options is not None and not isinstance(options, dict):
            raise ValueError("'options' debe ser de un tipo de dato diccionario o None.")

        self.apples_eaten = 0
        self.done = False
        self.steps = 0
        self.steps_since_apple = 0
        self.direction = RIGHT
        row, col = self.start_position
        self.snake_body = [np.array([row, col - index], dtype=int) for index in range(self.start_length)]

        if not self._spawn_apple():
            raise ValueError("No existe una casilla libre para colocar la manzana inicial.")

        context = self._spatial_context()
        self.potential = self._potential(context)
        info = {"level_id": self.config.get("level_id"), "apples_eaten": self.apples_eaten}
        return self._observation(context), info

    def step(self, action: object) -> StepResult:
        if self.done:
            raise RuntimeError("El episodio ya terminó, se debe hacer una restauracion con reset().")
        action = self._apply_direction(self._validate_action(action))
        self.steps += 1

        new_head = get_next_position(self.snake_body[0], action)
        will_eat = np.array_equal(new_head, self.apple_pos)
        collision_type = self._check_collision(new_head, allow_tail=not will_eat)
        if collision_type != "safe":
            return self._end_episode({"reason": collision_type}, self.reward_spec.death)

        self.snake_body.insert(0, new_head)
        if will_eat:
            return self._eat_apple()
        self.snake_body.pop()
        return self._move()

    def _apply_direction(self, action: int) -> int:
        if len(self.snake_body) > 1 and action == (self.direction + 2) % DIRECTION_COUNT:
            action = self.direction
        self.direction = action
        return action

    def _eat_apple(self) -> StepResult:
        self.apples_eaten += 1
        self.steps_since_apple = 0
        if self.apples_eaten >= self.apples_to_win:
            return self._end_episode({"reason": "win"}, self.reward_spec.apple)
        if not self._spawn_apple():
            info = {"reason": "win", "win_condition": "board_filled"}
            return self._end_episode(info, self.reward_spec.apple)
        context = self._spatial_context()
        self.potential = self._potential(context)
        return self._continue_or_truncate(context, self.reward_spec.apple)

    def _move(self) -> StepResult:
        self.steps_since_apple += 1
        context = self._spatial_context()
        previous_potential, self.potential = self.potential, self._potential(context)
        reward = move_reward(self.reward_spec, previous_potential, self.potential)
        return self._continue_or_truncate(context, reward)

    def _continue_or_truncate(self, context: SpatialContext, reward: float) -> StepResult:
        reason = self._truncation_reason()
        info = {"reason": reason or "moved", "apples_eaten": self.apples_eaten}
        if reason is not None:
            self.done = True
        return self._observation(context), reward, False, reason is not None, info

    def _truncation_reason(self) -> str | None:
        if self.max_steps is not None and self.steps >= self.max_steps:
            return "time_limit"
        if self.hunger_limit is not None and self.steps_since_apple >= self.hunger_limit:
            return "starvation"
        return None

    def _end_episode(self, info: dict, reward: float) -> StepResult:
        self.done = True
        info["apples_eaten"] = self.apples_eaten
        return self.observe(), reward, True, False, info

    def _check_collision(self, new_head: np.ndarray, allow_tail: bool = False) -> str:
        row, col = int(new_head[0]), int(new_head[1])

        if row < 0 or row >= self.grid_size[0] or col < 0 or col >= self.grid_size[1]:
            return "wall"
        if (row, col) in self.obstacles:
            return "obstacle"

        body = self.snake_body[:-1] if allow_tail and self.snake_body else self.snake_body
        if any(np.array_equal(new_head, body_part) for body_part in body):
            return "body"
        return "safe"

    def _spawn_apple(self) -> bool:
        occupied = self.obstacles | {tuple(part) for part in self.snake_body}
        free_cells = [
            (row, col)
            for row in range(self.grid_size[0])
            for col in range(self.grid_size[1])
            if (row, col) not in occupied
        ]
        if not free_cells:
            self.apple_pos = None
            return False

        apple_index = int(self.np_random.integers(len(free_cells)))
        self.apple_pos = np.array(free_cells[apple_index], dtype=int)
        return True

    def _spatial_context(self) -> SpatialContext:
        apple = None if self.apple_pos is None else as_cell(self.apple_pos)
        return build_spatial_context(self.grid_size, self.obstacles, self.snake_body, apple)

    def _potential(self, context: SpatialContext) -> float:
        return path_potential(head_path_distance(context, as_cell(self.snake_body[0])), context.free_cells)

    def _danger_features(self, context: SpatialContext) -> list[float]:
        cells = relative_move_cells(as_cell(self.snake_body[0]), self.direction)
        return [float(not is_open(cell, context.grid_size, context.blocked)) for cell in cells]

    def _observation(self, context: SpatialContext) -> np.ndarray:
        head = self.snake_body[0]
        state = self._danger_features(context)
        state += direction_one_hot(self.direction)
        state += apple_direction(head, self.apple_pos)
        state += spatial_features(context, as_cell(head), self.direction)
        return np.asarray(state, dtype=np.float32)

    def observe(self) -> np.ndarray:
        return self._observation(self._spatial_context())
