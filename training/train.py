import os
import numpy as np
from collections import deque

from utils.config import (
    LEVEL_PATHS, BATCH_SIZE, GAMMA,
    EPSILON_START, EPSILON_MIN, EPSILON_DECAY,
    TARGET_WIN_RATE, WINDOW_SIZE,
    MAX_EPISODES_PER_STAGE, MAX_STEPS_PER_EPISODE,
    MODELS_DIR, WEIGHTS_FILE,
)
from utils.plotting import plot_learning_curve


def train_by_stage(agent, memory, env_class, level_paths=LEVEL_PATHS,
                   batch_size=BATCH_SIZE, gamma=GAMMA):
    """
    Entrenamiento Curricular por Etapas.
    El agente debe dominar un nivel (alcanzar un win rate alto) para pasar al siguiente.
    Si no lo logra en MAX_EPISODES_PER_STAGE episodios, se avisa y se pasa al siguiente.

    Args:
        agent: Instancia de DQNAgent (Red Neuronal Feed-Forward).
        memory: Instancia de ExperienceReplay.
        env_class: Clase base del entorno (SnakeEnv).
        level_paths (list): Lista de rutas a los archivos JSON de los niveles.
        batch_size (int): Tamaño del lote para el entrenamiento.
        gamma (float): Factor de descuento para la Ecuación de Bellman.
    """
    # Epsilon (tasa de exploración) global que decrece conforme el agente aprende
    epsilon = EPSILON_START

    # Entrenar nivel por nivel (Curriculum Learning)
    for stage_idx, level_path in enumerate(level_paths):
        print(f"\n{'='*40}")
        print(f"Iniciando Entrenamiento: ETAPA {stage_idx + 1} ({level_path})")
        print(f"{'='*40}")

        env = env_class(level_path)

        # Historial para evaluar cuándo cambiar de nivel y para las gráficas
        win_history = deque(maxlen=WINDOW_SIZE)
        rewards_history = []
        win_rate_history = []
        episodes_in_stage = 0
        stage_passed = False

        while episodes_in_stage < MAX_EPISODES_PER_STAGE:
            episodes_in_stage += 1
            state = env.reset()
            done = False
            total_reward = 0
            steps = 0
            info = {}

            while not done:
                # 1. El agente decide su acción (Exploración vs Explotación)
                action = agent.act(state, epsilon)

                # 2. El entorno ejecuta la acción y devuelve el nuevo estado físico
                next_state, reward, done, info = env.step(action)
                total_reward += reward
                steps += 1

                # 3. Guardar la memoria para romper la correlación secuencial
                memory.remember(state, action, reward, next_state, done)
                state = next_state

                # 4. Entrenar la Red Neuronal con un lote aleatorio si hay suficientes memorias
                if len(memory) > batch_size:
                    states, actions, rewards, next_states, dones = memory.sample_batch(batch_size)

                    # --- MATEMÁTICA DEL Q-LEARNING (ECUACIÓN DE BELLMAN) ---
                    targets = agent.model.predict(states, verbose=0)
                    q_next = agent.model.predict(next_states, verbose=0)

                    for i in range(len(states)):
                        if dones[i]:
                            targets[i, actions[i]] = rewards[i]
                        else:
                            targets[i, actions[i]] = rewards[i] + gamma * np.amax(q_next[i])

                    agent.model.train_on_batch(states, targets)

                # 5. Cortar partidas que no terminan (la serpiente da vueltas sin comer)
                if not done and steps >= MAX_STEPS_PER_EPISODE:
                    info = {"reason": "timeout"}
                    break

            # Reducir la exploración poco a poco después de cada episodio
            if epsilon > EPSILON_MIN:
                epsilon *= EPSILON_DECAY

            # Registrar si el episodio terminó en victoria (1) o derrota (0)
            win_history.append(1 if info.get("reason") == "win" else 0)
            current_win_rate = sum(win_history) / len(win_history)

            # Guardar datos para la gráfica de este nivel
            rewards_history.append(total_reward)
            win_rate_history.append(current_win_rate)

            if episodes_in_stage % 10 == 0:
                print(f"Etapa {stage_idx + 1} | Episodio {episodes_in_stage} | Epsilon: {epsilon:.3f} | Reward: {total_reward:.2f} | Win Rate: {current_win_rate:.2f}")

            # Condición de victoria de la etapa
            if len(win_history) == WINDOW_SIZE and current_win_rate >= TARGET_WIN_RATE:
                print(f"¡Etapa {stage_idx + 1} superada en {episodes_in_stage} episodios!")
                stage_passed = True
                break

        if not stage_passed:
            print(f"Etapa {stage_idx + 1} NO superada tras {episodes_in_stage} episodios "
                  f"(win rate final: {current_win_rate:.2f}). Se continúa con la siguiente.")

        # Gráfica de rendimiento del nivel
        plot_learning_curve(rewards_history, win_rate_history, stage_idx + 1)

    print("\n¡Entrenamiento Curricular completado!")
    # Guardar los pesos finales para no tener que reentrenar al evaluar
    os.makedirs(MODELS_DIR, exist_ok=True)
    agent.model.save_weights(WEIGHTS_FILE)
    print(f"Modelo guardado en '{WEIGHTS_FILE}'")