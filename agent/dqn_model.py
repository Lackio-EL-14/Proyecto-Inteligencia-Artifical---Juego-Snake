import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation

def build_feedforward_model(input_shape, num_actions, learning_rate=0.001):
    """
    Construye y compila la Red Neuronal Feedforward.
    
    Args:
        input_shape (tuple): Dimensiones de la entrada (visión del agente).
        num_actions (int): Cantidad de salidas (generalmente 4 direcciones).
        learning_rate (float): Tasa de aprendizaje del optimizador.
        
    Returns:
        tf.keras.Model: Modelo compilado listo para entrenar.
    """
    pass

class DQNAgent:
    """
    Agente inteligente que utiliza el modelo Feedforward para predecir acciones 
    y gestionar la política de Exploración vs Explotación (Epsilon-Greedy).
    """
    def __init__(self, state_size, action_size):
        pass
        
    def act(self, state, epsilon):
        """
        Decide la siguiente acción basándose en la política Epsilon-Greedy.
        """
        pass
