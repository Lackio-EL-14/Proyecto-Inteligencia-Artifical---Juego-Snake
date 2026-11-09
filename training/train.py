import numpy as np
from collections import deque
import os

def train_by_stage(agent, memory, env_class, level_paths, batch_size=32, gamma=0.95):
    """
    Entrenamiento Curricular por Etapas.
    El agente debe dominar un nivel (alcanzar un win_rate alto) para pasar al siguiente.
    
    Args:
        agent: Instancia de DQNAgent (Red Neuronal Feed-Forward).
        memory: Instancia de ExperienceReplay.
        env_class: Clase base del entorno (SnakeEnv).
        level_paths (list): Lista de rutas a los archivos JSON de los niveles.
        batch_size (int): Tamaño del lote para el entrenamiento.
        gamma (float): Factor de descuento para la Ecuación de Bellman (recompensas futuras).
    """
    # Epsilon (tasa de exploración) global que decrece conforme el agente aprende
    epsilon = 1.0  
    epsilon_min = 0.05
    epsilon_decay = 0.995

    # Entrenar nivel por nivel (Curriculum Learning)
    for stage_idx, level_path in enumerate(level_paths):
        print(f"\n{'='*40}")
        print(f"Iniciando Entrenamiento: ETAPA {stage_idx + 1} ({level_path})")
        print(f"{'='*40}")
        
        env = env_class(level_path)
        
        # Historial de victorias para evaluar cuándo cambiar de nivel
        win_history = deque(maxlen=100) 
        episodes_in_stage = 0
        
        while True:
            episodes_in_stage += 1
            state = env.reset()
            done = False
            total_reward = 0
            
            while not done:
                # 1. El agente decide su acción (Exploración vs Explotación)
                action = agent.act(state, epsilon)
                
                # 2. El entorno ejecuta la acción y devuelve el nuevo estado físico
                next_state, reward, done, info = env.step(action)
                total_reward += reward
                
                # 3. Guardar la memoria para romper la correlación secuencial
                memory.remember(state, action, reward, next_state, done)
                state = next_state
                
                # 4. Entrenar la Red Neuronal con un lote aleatorio si hay suficientes memorias
                if len(memory) > batch_size:
                    states, actions, rewards, next_states, dones = memory.sample_batch(batch_size)
                    
                    # --- MATEMÁTICA DEL Q-LEARNING (ECUACIÓN DE BELLMAN) ---
                    # Predecimos los Q-Values actuales para los estados
                    targets = agent.model.predict(states, verbose=0)
                    
                    # Predecimos los Q-Values futuros para los siguientes estados
                    q_next = agent.model.predict(next_states, verbose=0)
                    
                    # Actualizamos el Q-Value específico de la acción tomada 
                    for i in range(batch_size):
                        if dones[i]:
                            targets[i, actions[i]] = rewards[i]
                        else:
                            # Recompensa inmediata + Recompensa máxima futura descontada (Gamma)
                            targets[i, actions[i]] = rewards[i] + gamma * np.amax(q_next[i])
                    
                    # Backpropagation: Ajustamos los pesos de la red neuronal para que 
                    # su predicción se acerque a nuestros 'targets' calculados.
                    agent.model.train_on_batch(states, targets)
                    #agent.model.fit(states, targets, epochs=1, verbose=0)

            # Reducir la exploración poco a poco después de cada episodio
            if epsilon > epsilon_min:
                epsilon *= epsilon_decay
            
            # Registrar si el episodio terminó en victoria (1) o derrota (0)
            if info.get("reason") == "win":
                win_history.append(1)
            else:
                win_history.append(0)
                
            # Evaluar el rendimiento (Win Rate)
            current_win_rate = sum(win_history) / len(win_history) if len(win_history) > 0 else 0.0
            
            if episodes_in_stage % 10 == 0:
                print(f"Etapa {stage_idx + 1} | Episodio {episodes_in_stage} | Epsilon: {epsilon:.3f} | Reward: {total_reward:.2f} | Win Rate: {current_win_rate:.2f}")
            
            # Condición de victoria de la etapa: Si gana el 90% de las últimas 100 partidas
            if len(win_history) == 100 and current_win_rate >= 0.90:
                print(f"¡Etapa {stage_idx + 1} superada en {episodes_in_stage} episodios!")
                break
                
    print("\n¡Entrenamiento Curricular completado con éxito!")
    # Guardar los pesos finales para no tener que reentrenar al evaluar
    os.makedirs("models", exist_ok=True)
    agent.model.save_weights("models/dqn_snake_weights.h5")
    print("Modelo guardado en 'models/dqn_snake_weights.h5'")
