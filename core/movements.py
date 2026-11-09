import numpy as np 

# Constantes de acción (Salidas de la Red Neuronal)
UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

# Traductor físico: Mapea la acción a un vector de desplazamiento (delta_fila, delta_columna)
# En una matriz de NumPy: 
# - Ir arriba (UP) resta 1 a la fila.
# - Ir abajo (DOWN) suma 1 a la fila.
# - Ir a la derecha (RIGHT) suma 1 a la columna.
# - Ir a la izquierda (LEFT) resta 1 a la columna.
ACTION_DELTAS = {
    UP: np.array([-1, 0]),
    RIGHT: np.array([0, 1]),
    DOWN: np.array([1, 0]),
    LEFT: np.array([0, -1])
}

def get_next_position(current_pos, action):
    """
    Calcula la siguiente coordenada matemática basada en la posición actual y la acción.
    
    Args:
        current_pos (np.array o list): Coordenada actual [fila, columna].
        action (int): Acción a ejecutar (0, 1, 2, 3).
        
    Returns:
        np.array: Nueva coordenada [fila, columna].
    """
    delta = ACTION_DELTAS.get(action, np.array([0, 0]))
    return np.array(current_pos) + delta
