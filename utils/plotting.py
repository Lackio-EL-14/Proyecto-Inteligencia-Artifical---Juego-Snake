# utils/plotting.py

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Solo guarda archivos, no abre ventanas
import matplotlib.pyplot as plt

from utils.config import PLOTS_DIR, TARGET_WIN_RATE, WINDOW_SIZE


def moving_average(values, window=WINDOW_SIZE):
    """Promedio móvil. Si hay menos datos que la ventana, devuelve los valores tal cual."""
    values = np.asarray(values, dtype=float)
    if len(values) < window:
        return values
    return np.convolve(values, np.ones(window) / window, mode="valid")


def plot_learning_curve(rewards, win_rates, stage, save_dir=PLOTS_DIR):
    """Guarda la curva de aprendizaje de un nivel (recompensa y win rate vs. episodios)."""
    os.makedirs(save_dir, exist_ok=True)
    n = len(rewards)
    episodes = np.arange(1, n + 1)

    fig, ax1 = plt.subplots(figsize=(10, 5))

    # Recompensa por episodio (tenue) y su promedio móvil (marcado)
    ax1.plot(episodes, rewards, color="tab:blue", alpha=0.3, label="Recompensa por episodio")
    avg = moving_average(rewards)
    ax1.plot(np.arange(n - len(avg) + 1, n + 1), avg, color="tab:blue",
             label=f"Promedio móvil ({WINDOW_SIZE})")
    ax1.set_xlabel("Episodios")
    ax1.set_ylabel("Recompensa total", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.grid(alpha=0.3)

    # Win rate en el eje derecho, con la meta del currículo
    ax2 = ax1.twinx()
    ax2.plot(episodes[:len(win_rates)], win_rates, color="tab:red", label=f"Win rate (últimos {WINDOW_SIZE})")
    ax2.axhline(y=TARGET_WIN_RATE, color="green", linestyle="--",
                label=f"Meta ({TARGET_WIN_RATE:.0%})")
    ax2.set_ylabel("Win rate", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")
    ax2.set_ylim(0, 1.05)

    # Leyenda única con las líneas de ambos ejes
    lines = ax1.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
    labels = ax1.get_legend_handles_labels()[1] + ax2.get_legend_handles_labels()[1]
    ax1.legend(lines, labels, loc="lower right")

    ax1.set_title(f"Curva de aprendizaje - Nivel {stage}")
    fig.tight_layout()

    filename = os.path.join(save_dir, f"learning_curve_level_{stage}.png")
    fig.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"Gráfica guardada en {filename}")