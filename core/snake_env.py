import numpy as np
import json
import random

class SnakeEnv:
    """
    Entorno dinámico para el juego de Snake.
    Maneja la matriz, el crecimiento del agente, los obstáculos y las colisiones.
    """
    def __init__(self, level_config_path):
        """
        Carga la configuración del nivel desde el JSON e inicializa las variables base.
        """
        pass

    def reset(self):
        """
        Reinicia el entorno al estado inicial del nivel actual.
        
        Returns:
            np.array: El estado inicial del tablero (matriz aplanada o 2D) visto por el agente.
        """
        pass

    def step(self, action):
        """
        Ejecuta una acción (movimiento) en el entorno.
        
        Args:
            action (int): 0 (Arriba), 1 (Derecha), 2 (Abajo), 3 (Izquierda).
            
        Returns:
            tuple: (next_state, reward, done, info)
                - next_state (np.array): La nueva visión del tablero.
                - reward (float): Recompensa calculada para este paso.
                - done (bool): True si el agente gana, muere o se atasca. False si el juego continúa.
                - info (dict): Información adicional (ej. manzanas comidas, pasos dados).
        """
        pass

    def _spawn_apple(self):
        """
        Método privado. Genera una manzana en una coordenada libre al azar.
        Debe verificar que la coordenada no sea un muro ni parte del cuerpo de la serpiente.
        """
        pass

    def _check_collision(self, new_head_pos):
        """
        Método privado. Evalúa si la nueva posición de la cabeza resulta en colisión.
        
        Args:
            new_head_pos (list): Coordenada [fila, columna].
            
        Returns:
            str: 'wall', 'body', 'obstacle' o 'safe'.
        """
        pass

    def observe(self):
        """
        Traduce el estado actual del juego (cuerpo, obstáculos, manzana) a una matriz numérica.
        Similar al 'draw_env' del baseline, pero enfocado en la entrada de la Red Neuronal.
        
        Returns:
            np.array: Representación numérica del entorno.
        """
        pass
