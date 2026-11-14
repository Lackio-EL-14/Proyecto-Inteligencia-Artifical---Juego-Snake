import json

import numpy as np

from core.movements import DOWN, LEFT, RIGHT, UP, get_next_position


class SnakeEnv:
    def __init__(self, level_config_path, max_steps=None):
        with open(level_config_path, "r", encoding="utf-8") as level_file:
            self.config = json.load(level_file)

        if not isinstance(self.config, dict):
            raise ValueError("La configuración del nivel debe ser un objeto JSON.")

        self.grid_size = self._validate_grid_size(self.config.get("grid_size"))
        self.start_position = self._validate_position(
            self.config.get("start_position"), "start_position", within_grid=True
        )
        self.start_length = self.config.get("start_length", 2)
        self.apples_to_win = self.config.get("apples_to_win")
        self.obstacles = self._validate_obstacles(self.config.get("obstacles"))

        if type(self.start_length) is not int or self.start_length <= 0:
            raise ValueError("'start_length' debe ser un entero positivo.")

        if type(self.apples_to_win) is not int or self.apples_to_win <= 0:
            raise ValueError("'apples_to_win' debe ser un entero positivo.")

        if max_steps is not None and (type(max_steps) is not int or max_steps <= 0):
            raise ValueError("'max_steps' debe ser None o un entero positivo.")

        self.max_steps = max_steps
        self._validate_initial_snake()

        available_cells = self.grid_size[0] * self.grid_size[1]
        available_cells -= len(self.obstacles) + self.start_length
        if available_cells <= 0:
            raise ValueError("El nivel no deja casillas libres para generar una manzana.")
        if self.apples_to_win > available_cells:
            raise ValueError("'apples_to_win' supera la capacidad libre del tablero.")

        self.np_random = np.random.default_rng()
        self.snake_body = []
        self.apple_pos = None
        self.apples_eaten = 0
        self.direction = RIGHT
        self.steps = 0
        self.done = False

    def _validate_grid_size(self, grid_size):
        if not isinstance(grid_size, list) or len(grid_size) != 2:
            raise ValueError("'grid_size' debe ser una lista [filas, columnas].")
        if any(type(size) is not int or size <= 0 for size in grid_size):
            raise ValueError("'grid_size' debe contener dos enteros positivos.")
        return tuple(grid_size)

    def _validate_position(self, position, field_name, within_grid=False):
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

    def _validate_obstacles(self, obstacles):
        if obstacles is None:
            return set()
        if not isinstance(obstacles, list):
            raise ValueError("'obstacles' debe ser una lista de coordenadas o null.")

        validated = set()
        for index, obstacle in enumerate(obstacles):
            position = self._validate_position(
                obstacle, f"obstacles[{index}]", within_grid=True
            )
            if position in validated:
                raise ValueError(f"El obstáculo {position} está duplicado.")
            validated.add(position)
        return validated

    def _validate_initial_snake(self):
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

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            if type(seed) is not int:
                raise ValueError("'seed' debe ser un entero o None.")
            self.np_random = np.random.default_rng(seed)
        if options is not None and not isinstance(options, dict):
            raise ValueError("'options' debe ser de un tipo de dato diccionario o None.")

        self.apples_eaten = 0
        self.done = False
        self.steps = 0
        self.direction = RIGHT
        row, col = self.start_position
        self.snake_body = [
            np.array([row, col - index], dtype=int)
            for index in range(self.start_length)
        ]

        if not self._spawn_apple():
            raise ValueError("No existe una casilla libre para colocar la manzana inicial.")

        info = {
            "level_id": self.config.get("level_id"),
            "apples_eaten": self.apples_eaten,
        }
        return self.observe(), info

    def step(self, action):
        if self.done:
            raise RuntimeError("El episodio ya terminó, se debe hacer una restauracion con reset().")
        if (
            not isinstance(action, (int, np.integer))
            or isinstance(action, (bool, np.bool_))
            or action not in (UP, RIGHT, DOWN, LEFT)
        ):
            raise ValueError("'action' debe ser un movimiento valido como UP, RIGHT, DOWN o LEFT.")
        action = int(action)

        if len(self.snake_body) > 1 and action == (self.direction + 2) % 4:
            action = self.direction
        self.direction = action
        self.steps += 1

        head = self.snake_body[0]
        new_head = get_next_position(head, action)
        will_eat = np.array_equal(new_head, self.apple_pos)
        collision_type = self._check_collision(new_head, allow_tail=not will_eat)

        if collision_type != "safe":
            self.done = True
            info = {"reason": collision_type, "apples_eaten": self.apples_eaten}
            return self.observe(), -1.0, True, False, info

        self.snake_body.insert(0, new_head)

        if will_eat:
            self.apples_eaten += 1
            reward = 1.0

            if self.apples_eaten >= self.apples_to_win:
                self.done = True
                info = {"reason": "win", "apples_eaten": self.apples_eaten}
                return self.observe(), reward, True, False, info

            if not self._spawn_apple():
                self.done = True
                info = {
                    "reason": "win",
                    "win_condition": "board_filled",
                    "apples_eaten": self.apples_eaten,
                }
                return self.observe(), reward, True, False, info
        else:
            self.snake_body.pop()
            old_distance = abs(head[0] - self.apple_pos[0]) + abs(
                head[1] - self.apple_pos[1]
            )
            new_distance = abs(new_head[0] - self.apple_pos[0]) + abs(
                new_head[1] - self.apple_pos[1]
            )
            reward = 0.1 if new_distance < old_distance else -0.1

        truncated = self.max_steps is not None and self.steps >= self.max_steps
        if truncated:
            self.done = True
            info = {"reason": "time_limit", "apples_eaten": self.apples_eaten}
        else:
            info = {"reason": "moved", "apples_eaten": self.apples_eaten}

        return self.observe(), reward, False, truncated, info

    def _check_collision(self, new_head, allow_tail=False):
        row, col = int(new_head[0]), int(new_head[1])

        if row < 0 or row >= self.grid_size[0] or col < 0 or col >= self.grid_size[1]:
            return "wall"
        if (row, col) in self.obstacles:
            return "obstacle"

        body = self.snake_body[:-1] if allow_tail and self.snake_body else self.snake_body
        if any(np.array_equal(new_head, body_part) for body_part in body):
            return "body"
        return "safe"

    def _spawn_apple(self):
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

    def observe(self):
        head = self.snake_body[0]
        point_u = get_next_position(head, UP)
        point_r = get_next_position(head, RIGHT)
        point_d = get_next_position(head, DOWN)
        point_l = get_next_position(head, LEFT)

        dir_u = self.direction == UP
        dir_r = self.direction == RIGHT
        dir_d = self.direction == DOWN
        dir_l = self.direction == LEFT

        def is_danger(point):
            return self._check_collision(point, allow_tail=True) != "safe"

        danger_straight = (
            (dir_r and is_danger(point_r))
            or (dir_l and is_danger(point_l))
            or (dir_u and is_danger(point_u))
            or (dir_d and is_danger(point_d))
        )
        danger_right = (
            (dir_u and is_danger(point_r))
            or (dir_d and is_danger(point_l))
            or (dir_l and is_danger(point_u))
            or (dir_r and is_danger(point_d))
        )
        danger_left = (
            (dir_u and is_danger(point_l))
            or (dir_d and is_danger(point_r))
            or (dir_r and is_danger(point_u))
            or (dir_l and is_danger(point_d))
        )

        if self.apple_pos is None:
            apple_u = apple_r = apple_d = apple_l = False
        else:
            apple_u = self.apple_pos[0] < head[0]
            apple_d = self.apple_pos[0] > head[0]
            apple_l = self.apple_pos[1] < head[1]
            apple_r = self.apple_pos[1] > head[1]

        state = [
            int(danger_straight),
            int(danger_right),
            int(danger_left),
            int(dir_u),
            int(dir_r),
            int(dir_d),
            int(dir_l),
            int(apple_u),
            int(apple_r),
            int(apple_d),
            int(apple_l),
        ]
        return np.asarray(state, dtype=np.float32)
