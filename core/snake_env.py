# core/snake_env.py
import numpy as np
import json
import random
from core.movements import UP, RIGHT, DOWN, LEFT, get_next_position

class SnakeEnv:
    def __init__(self, level_config_path):
        # Configuración del JSON
        with open(level_config_path, 'r') as f:
            self.config = json.load(f)
            
        self.grid_size = self.config['grid_size']
        print(self.grid_size)
        if len(self.grid_size) != 2 or not isinstance(self.grid_size, list):
            raise ValueError("El campo de 'grid_size' debe ser una lista con los valores [filas, columnas]")
        
        if any(type(size) != int for size in self.grid_size):
            raise ValueError("El campo de 'grid_size debe tener dos numeros enteros en el formato [filas, columnas]")
        
        self.grid_size = tuple(self.grid_size)
        
        self.start_position = self.config['start_position']
        
        if not isinstance(self.start_position, list) or len(self.start_position) != 2: 
            raise ValueError("El campo de 'start_position' debe ser una lista con los valores [filas, columnas]")
        
        if any(type(position) != int for position in self.start_position):
            raise ValueError("El campo de 'start_position' debe tener valores de tipo 'int'")
        
        row, col = self.start_position
        
        grid_row, grid_col = self.grid_size
        
        if not(0 <= row < grid_row or 0 <= col < grid_col):
            raise ValueError(f"""Los valores de 'start_position' se encuentran fuera del tablero 
                             'start_position': {self.start_position}
                             'tamaño del tablero: {self.grid_size}
                             """)
        
        self.start_length = self.config.get('start_length', 2) # Por defecto 2 bloques
        self.apples_to_win = self.config['apples_to_win']
        # Si get() devuelve None, el operador 'or' usará la lista vacía []
        self.obstacles = [tuple(obs) for obs in (self.config.get('obstacles') or [])]
        
        # Variables de estado que cambiarán en cada episodio
        self.snake_body = []
        self.apple_pos = None
        self.apples_eaten = 0
        self.done = False

    def reset(self):
        self.apples_eaten = 0
        self.done = False
        
        # Inicializar dirección actual (por defecto a la derecha)
        self.direction = RIGHT
        
        # Inicializar el cuerpo de la serpiente (cabeza en start_position)
        self.snake_body = [np.array(self.start_position)]
        for i in range(1, self.start_length):
            # Construye el cuerpo inicial hacia la izquierda para que mire a la derecha
            self.snake_body.append(np.array([self.start_position[0], self.start_position[1] - i]))
            
        self._spawn_apple()
        return self.observe()

    def step(self, action):
        if self.done:
            raise RuntimeError("El episodio ya terminó. Llama a reset().")

        # Actualizar la dirección actual con la acción tomada
        self.direction = action

        # 1. Calcular nueva posición de la cabeza
        head = self.snake_body[0]
        new_head = get_next_position(head, action)

        # 2. Verificar colisiones (Muros, Obstáculos, o su propio cuerpo)
        collision_type = self._check_collision(new_head)
        
        if collision_type in ['wall', 'obstacle', 'body']:
            self.done = True
            reward = -1.0 
            return self.observe(), reward, self.done, {"reason": collision_type}

        # 3. Mover la serpiente
        self.snake_body.insert(0, new_head)

        # 4. Verificar si comió la manzana
        if np.array_equal(new_head, self.apple_pos):
            self.apples_eaten += 1
            reward = 1.0 # Premio por comer
            
            if self.apples_eaten >= self.apples_to_win:
                self.done = True
                return self.observe(), reward, self.done, {"reason": "win"}
            
            self._spawn_apple()
        else:
            self.snake_body.pop()
            
            # Recompensas densas (acercarse/alejarse)
            old_dist = abs(head[0] - self.apple_pos[0]) + abs(head[1] - self.apple_pos[1])
            new_dist = abs(new_head[0] - self.apple_pos[0]) + abs(new_head[1] - self.apple_pos[1])
            
            if new_dist < old_dist:
                reward = 0.1
            else:
                reward = -0.1

        return self.observe(), reward, self.done, {"reason": "moved"}


    def _check_collision(self, new_head):
        row, col = new_head[0], new_head[1]
        
        # Muros
        if row < 0 or row >= self.grid_size[0] or col < 0 or col >= self.grid_size[1]:
            return 'wall'
            
        # Obstáculos fijos
        if tuple(new_head) in self.obstacles:
            return 'obstacle'
            
        # Su propio cuerpo (excluyendo la punta de la cola si se va a mover, 
        # pero por simplicidad inicial, chocará con cualquier parte actual)
        for body_part in self.snake_body:
            if np.array_equal(new_head, body_part):
                return 'body'
                
        return 'safe'

    def _spawn_apple(self):
        # Generar posiciones aleatorias hasta encontrar una libre
        while True:
            row = random.randint(0, self.grid_size[0] - 1)
            col = random.randint(0, self.grid_size[1] - 1)
            pos = np.array([row, col])
            
            # Verificar que no caiga en obstáculos ni en la serpiente
            if tuple(pos) not in self.obstacles:
                is_on_snake = any(np.array_equal(pos, part) for part in self.snake_body)
                if not is_on_snake:
                    self.apple_pos = pos
                    break

    def observe(self):
        """
        Devuelve el estado relativo de la serpiente (Vector de 11 booleanos/enteros).
        """
        head = self.snake_body[0]
        
        # Puntos adyacentes absolutos
        point_u = get_next_position(head, UP)
        point_r = get_next_position(head, RIGHT)
        point_d = get_next_position(head, DOWN)
        point_l = get_next_position(head, LEFT)
        
        # Direcciones booleanas actuales
        dir_u = self.direction == UP
        dir_r = self.direction == RIGHT
        dir_d = self.direction == DOWN
        dir_l = self.direction == LEFT
        
        # Función auxiliar para detectar peligro letal en una coordenada
        def is_danger(pt):
            return self._check_collision(pt) in ['wall', 'obstacle', 'body']
            
        # Peligro relativo a la cabeza
        danger_straight = (dir_r and is_danger(point_r)) or (dir_l and is_danger(point_l)) or \
                          (dir_u and is_danger(point_u)) or (dir_d and is_danger(point_d))
                          
        danger_right = (dir_u and is_danger(point_r)) or (dir_d and is_danger(point_l)) or \
                       (dir_l and is_danger(point_u)) or (dir_r and is_danger(point_d))
                       
        danger_left = (dir_u and is_danger(point_l)) or (dir_d and is_danger(point_r)) or \
                      (dir_r and is_danger(point_u)) or (dir_l and is_danger(point_d))
                      
        # Posición de la manzana relativa a la cabeza
        apple_u = self.apple_pos[0] < head[0]
        apple_d = self.apple_pos[0] > head[0]
        apple_l = self.apple_pos[1] < head[1]
        apple_r = self.apple_pos[1] > head[1]
        
        # Vector de estado de 11 dimensiones
        state = [
            int(danger_straight), int(danger_right), int(danger_left),
            int(dir_u), int(dir_r), int(dir_d), int(dir_l),
            int(apple_u), int(apple_r), int(apple_d), int(apple_l)
        ]
        
        return np.array(state, dtype=float)

