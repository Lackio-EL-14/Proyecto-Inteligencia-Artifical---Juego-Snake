from core.snake_env import SnakeEnv
from core.experience import ExperienceReplay
from agent.dqn_model import DQNAgent
from training.train import train_by_stage
from utils.config import (
    LEVEL_PATHS, STATE_SIZE, ACTION_SIZE,
    LEARNING_RATE, MAX_MEMORY, BATCH_SIZE, GAMMA,
)


def main():
    # 1. Inicializar los componentes (todos los valores salen de utils/config.py)
    print("Inicializando componentes...")
    agent = DQNAgent(state_size=STATE_SIZE, action_size=ACTION_SIZE, learning_rate=LEARNING_RATE)
    memory = ExperienceReplay(max_memory=MAX_MEMORY)

    # 2. Iniciar el entrenamiento curricular
    print("Iniciando entrenamiento...")
    train_by_stage(
        agent=agent,
        memory=memory,
        env_class=SnakeEnv,
        level_paths=LEVEL_PATHS,
        batch_size=BATCH_SIZE,
        gamma=GAMMA,
    )


if __name__ == "__main__":
    main()