# utils/plotting.py

import matplotlib.pyplot as plt
import os

def plot_learning_curve(rewards, win_rates, stage, save_dir="plots"):
    """Genera y guarda las gráficas de rendimiento del agente (Recompensas y Win Rate)."""
    os.makedirs(save_dir, exist_ok=True)
    
    fig, ax1 = plt.subplots(figsize=(10, 5))

    color = 'tab:blue'
    ax1.set_xlabel('Episodios')
    ax1.set_ylabel('Recompensa Total', color=color)
    ax1.plot(rewards, color=color, alpha=0.6, label='Recompensa')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  
    color = 'tab:red'
    ax2.set_ylabel('Win Rate (Últimos 100)', color=color)  
    ax2.plot(win_rates, color=color, label='Win Rate')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.axhline(y=0.9, color='green', linestyle='--', label='Meta (90%)')

    fig.tight_layout()  
    plt.title(f'Curva de Aprendizaje - Nivel {stage}')
    
    filename = os.path.join(save_dir, f'learning_curve_level_{stage}.png')
    plt.savefig(filename)
    plt.close()
    print(f"Gráfica guardada en {filename}")
