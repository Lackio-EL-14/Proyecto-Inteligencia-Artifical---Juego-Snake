from core.snake_env import SnakeEnv
from core.experience import ExperienceReplay
from agent.dqn_model import DQNAgent, build_feedforward_model

def train_agent(level_path, epochs=1000, batch_size=32):
    """
    Orquesta el entrenamiento. 
    1. Inicializa el entorno, el agente y la memoria.
    2. Ejecuta el bucle de episodios (vidas de la serpiente).
    3. Ajusta las recompensas (Fitness Function).
    4. Solicita el entrenamiento del modelo por lotes.
    5. Guarda los pesos del mejor modelo.
    """
    pass
