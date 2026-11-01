import numpy as np
import random

class ExperienceReplay:
    """
    Buffer de memoria (Replay Buffer) para almacenar episodios pasados y 
    permitir que la red neuronal aprenda de experiencias aleatorias.
    """
    def __init__(self, max_memory=1000):
        """
        Inicializa la memoria con un tamaño máximo.
        """
        pass

    def remember(self, state, action, reward, next_state, done):
        """
        Guarda una tupla de transición en la memoria. Si se supera el límite, 
        elimina el recuerdo más antiguo.
        """
        pass

    def sample_batch(self, batch_size):
        """
        Extrae un lote aleatorio de memorias para entrenar la red neuronal.
        Rompe la correlación secuencial de los movimientos (crucial para Deep Q-Learning).
        
        Args:
            batch_size (int): Cantidad de muestras a extraer.
            
        Returns:
            tuple: Arrays de NumPy con (states, actions, rewards, next_states, dones).
        """
        pass
