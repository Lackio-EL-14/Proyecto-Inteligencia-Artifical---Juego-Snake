# core/experience.py

import numpy as np
import random
from collections import deque

class ExperienceReplay:
    """
    Buffer de memoria (Replay Buffer) para almacenar episodios pasados.
    Permite a la red neuronal aprender de minilotes (batches) aleatorios,
    rompiendo la correlación temporal de las acciones secuenciales.
    """
    def __init__(self, max_memory=10000):
        # deque automáticamente descarta el elemento más antiguo si se supera max_memory
        self.memory = deque(maxlen=max_memory)

    def remember(self, state, action, reward, next_state, done):
        """
        Guarda una tupla de transición en la memoria.
        """
        self.memory.append((state, action, reward, next_state, done))

    def sample_batch(self, batch_size):
        """
        Extrae un lote aleatorio de memorias para entrenar la red neuronal.
        
        Args:
            batch_size (int): Cantidad de muestras a extraer.
            
        Returns:
            tuple: (states, actions, rewards, next_states, dones) como arrays de NumPy.
        """
        # Extraemos un tamaño de lote válido (por si la memoria aún no tiene suficientes datos)
        batch = random.sample(self.memory, min(len(self.memory), batch_size))
        
        # Desempaquetamos el lote en arrays individuales para optimizar la entrada a TensorFlow
        states = np.array([i[0] for i in batch])
        actions = np.array([i[1] for i in batch])
        rewards = np.array([i[2] for i in batch])
        next_states = np.array([i[3] for i in batch])
        dones = np.array([i[4] for i in batch])
        
        return states, actions, rewards, next_states, dones
        
    def __len__(self):
        return len(self.memory)
