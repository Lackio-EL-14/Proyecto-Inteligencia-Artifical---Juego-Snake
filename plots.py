import argparse
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from baselines import BASELINES_CSV
from config import LEVEL_PATHS, RUNS_DIR

SERIES_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
TEXT_COLOR = "#52514e"
GRID_COLOR = "#e4e3df"
TARGET_WIN_RATE = 0.9
SMOOTHING_WINDOW = 50
ABLATION_CSV = os.path.join(RUNS_DIR, "ablaciones.csv")
LEVEL_IDS = list(range(1, len(LEVEL_PATHS) + 1))


def read_rows(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def style_axis(axis: plt.Axes, title: str, ylabel: str) -> None:
    axis.set_title(title, loc="left", fontsize=11, color=TEXT_COLOR)
    axis.set_ylabel(ylabel, color=TEXT_COLOR)
    axis.grid(axis="y", color=GRID_COLOR, linewidth=0.8)
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(colors=TEXT_COLOR)


def moving_average(values: list[float], window: int) -> np.ndarray:
    size = min(window, len(values))
    return np.convolve(values, np.ones(size) / size, mode="valid")


def save_figure(figure: plt.Figure, path: str) -> str:
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def plot_learning_curve(run_dir: str) -> str:
    rows = read_rows(os.path.join(run_dir, "episodes.csv"))
    figure, (reward_axis, epsilon_axis) = plt.subplots(2, 1, figsize=(8, 6), sharex=True, height_ratios=(3, 1))
    for level, color in zip(LEVEL_IDS, SERIES_COLORS):
        level_rows = [row for row in rows if int(row["level"]) == level]
        smoothed = moving_average([float(row["reward"]) for row in level_rows], SMOOTHING_WINDOW // 4)
        episodes = [int(row["episode"]) for row in level_rows][len(level_rows) - len(smoothed):]
        reward_axis.plot(episodes, smoothed, color=color, linewidth=2, label=f"Nivel {level}")
    reward_axis.legend(frameon=False, ncols=4, loc="upper left")
    style_axis(reward_axis, "Recompensa por episodio (media móvil por nivel)", "recompensa")
    epsilon_axis.plot([int(row["episode"]) for row in rows], [float(row["epsilon"]) for row in rows], color=TEXT_COLOR, linewidth=2)
    style_axis(epsilon_axis, "Exploración ε", "ε")
    epsilon_axis.set_xlabel("episodio", color=TEXT_COLOR)
    return save_figure(figure, os.path.join(run_dir, "learning_curve.png"))


def plot_win_rates(run_dir: str) -> str:
    rows = read_rows(os.path.join(run_dir, "evaluations.csv"))
    figure, axes = plt.subplots(1, len(LEVEL_IDS), figsize=(11, 3), sharey=True)
    for axis, level, color in zip(axes, LEVEL_IDS, SERIES_COLORS):
        level_rows = [row for row in rows if int(row["level"]) == level]
        axis.plot([int(row["episode"]) for row in level_rows], [float(row["win_rate"]) for row in level_rows], color=color, linewidth=2, marker="o", markersize=4)
        axis.axhline(TARGET_WIN_RATE, color=TEXT_COLOR, linewidth=1, linestyle="--")
        style_axis(axis, f"Nivel {level}", "victorias (ε = 0)" if level == 1 else "")
        axis.set_xlabel("episodio", color=TEXT_COLOR)
    axes[0].set_ylim(-0.02, 1.05)
    return save_figure(figure, os.path.join(run_dir, "win_rates.png"))


def agent_win_rates(run_dir: str) -> list[float]:
    rows = read_rows(os.path.join(run_dir, "final_evaluation.csv"))
    return [np.mean([row["reason"] == "win" for row in rows if int(row["level"]) == level]) for level in LEVEL_IDS]


def baseline_win_rates(policy: str) -> list[float]:
    return [float(row["win_rate"]) for row in read_rows(BASELINES_CSV) if row["policy"] == policy]


def draw_grouped_bars(axis: plt.Axes, groups: dict[str, list[float]]) -> None:
    width = 0.8 / len(groups)
    for index, ((name, values), color) in enumerate(zip(groups.items(), SERIES_COLORS)):
        positions = np.arange(len(values)) + (index - (len(groups) - 1) / 2) * width
        bars = axis.bar(positions, values, width * 0.92, color=color, label=name)
        axis.bar_label(bars, labels=[f"{value:.0%}" for value in values], fontsize=8, color=TEXT_COLOR, padding=2)


def plot_baseline_comparison(run_dir: str) -> str:
    groups = {"Aleatoria": baseline_win_rates("random"), "Heurística (solo estado)": baseline_win_rates("heuristic"), "Double DQN": agent_win_rates(run_dir)}
    figure, axis = plt.subplots(figsize=(8, 4))
    draw_grouped_bars(axis, groups)
    axis.set_xticks(range(len(LEVEL_IDS)), [f"Nivel {level}" for level in LEVEL_IDS])
    axis.set_ylim(0, 1.15)
    axis.legend(frameon=False, ncols=3, loc="upper center")
    style_axis(axis, "Tasa de victorias por política (sin crecimiento)", "victorias")
    return save_figure(figure, os.path.join(run_dir, "baseline_comparison.png"))


SEED_OFFSETS = {"42": -0.12, "7": 0.0, "123": 0.12}


def ablation_order(row: dict) -> tuple[bool, float]:
    return row["distancia"] == "True", -float(row["learning_rate"])


def ablation_label(row: dict) -> str:
    state = "7 valores" if row["distancia"] == "True" else "6 valores"
    return f"{state}\nlr {float(row['learning_rate']):g}"


def plot_ablation(run_dir: str) -> str:
    rows = [row for row in read_rows(ABLATION_CSV) if not row["otros"]]
    labels = [ablation_label(row) for row in sorted(rows, key=ablation_order)]
    labels = list(dict.fromkeys(labels))
    figure, axis = plt.subplots(figsize=(8, 4))
    for row in rows:
        worst = min(float(row[f"final_n{level}"]) for level in LEVEL_IDS)
        position = labels.index(ablation_label(row)) + SEED_OFFSETS[row["seed"]]
        axis.scatter(position, worst, s=60, color=SERIES_COLORS[0], edgecolor="white", linewidth=2, zorder=3)
    axis.axhline(TARGET_WIN_RATE, color=TEXT_COLOR, linewidth=1, linestyle="--")
    axis.set_xticks(range(len(labels)), labels)
    axis.set_ylim(-0.02, 1.05)
    style_axis(axis, "Peor nivel del modelo final; marcas de izquierda a derecha: semillas 42, 7, 123", "victorias en el peor nivel")
    return save_figure(figure, os.path.join(run_dir, "ablation.png"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default=os.path.join(RUNS_DIR, "completa"))
    return parser.parse_args()


def main() -> None:
    run_dir = parse_args().run
    for plot in (plot_learning_curve, plot_win_rates, plot_baseline_comparison, plot_ablation):
        print(f"guardado: {plot(run_dir)}")


if __name__ == "__main__":
    main()
