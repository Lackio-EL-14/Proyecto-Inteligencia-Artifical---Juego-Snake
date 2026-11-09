# agent/dqn_model.py

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
import numpy as np
import random

def build_feedforward_model(input_shape, num_actions, learning_rate=0.001):
    model = Sequential([
        Input(shape=input_shape),
        Dense(64, activation='relu'),
        Dense(64, activation='relu'),
        Dense(num_actions, activation='linear')
    ])
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
    return model

# ... resto de la clase DQNAgent sin cambios ...

class DQNAgent:
    """
    Agente inteligente que utiliza el modelo Feed-Forward para predecir acciones 
    y gestionar la política de exploración vs explotación (Epsilon-Greedy).
    """
    def __init__(self, state_size, action_size, learning_rate=0.001):
        self.state_size = state_size
        self.action_size = action_size
        
        # Instanciamos el Cerebro del agente (Topología Feed-Forward)
        self.model = build_feedforward_model((state_size,), action_size, learning_rate)

    def act(self, state, epsilon):
        """
        Decide la siguiente acción basándose en la política Epsilon-Greedy.
        
        Args:
            state (np.array): Estado actual del entorno.
            epsilon (float): Probabilidad de tomar una acción aleatoria (0.0 a 1.0).
            
        Returns:
            int: Índice de la acción seleccionada (0, 1, 2, 3).
        """
        # Exploración: Tomar una decisión aleatoria
        if np.random.rand() <= epsilon:
            return random.randrange(self.action_size)
            
        # Explotación: Forward Propagation a través de la red neuronal
        # El modelo recibe el estado y devuelve un array con 4 Q-Values
        state_batch = np.expand_dims(state, axis=0)
        q_values = self.model.predict(state_batch, verbose=0) 
        
        # Seleccionamos la acción que la red considera que otorgará mayor recompensa
        return np.argmax(q_values[0])
