import os
from core.snake_env import SnakeEnv
from core.experience import ExperienceReplay
from agent.dqn_model import DQNAgent
from training.train import train_by_stage

def main():
    # 1. Definir los niveles a entrenar
    # Asegúrate de que las rutas a los JSON sean correctas.
    level_paths = [
        "levels/level_1.json",
        "levels/level_2.json",
        "levels/level_3.json"
    ]
    
    # 2. Configurar el tamaño del estado (visión relativa) y las acciones
    # El estado ahora es un vector fijo de 11 valores (Peligros, Dirección, Manzana)
    state_size = 11
    action_size = 4  # UP, RIGHT, DOWN, LEFT
    
    # 3. Inicializar los componentes
    print("Inicializando componentes...")
    agent = DQNAgent(state_size=state_size, action_size=action_size, learning_rate=0.001)
    memory = ExperienceReplay(max_memory=10000)
    
    # 4. Iniciar el entrenamiento curricular
    print("Iniciando entrenamiento...")
    train_by_stage(
        agent=agent,
        memory=memory,
        env_class=SnakeEnv,
        level_paths=level_paths,
        batch_size=64,   # Actualizado a 64 para mayor estabilidad en el entrenamiento
        gamma=0.95
    )

if __name__ == "__main__":
    main()
