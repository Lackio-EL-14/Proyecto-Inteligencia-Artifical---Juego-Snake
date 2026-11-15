import os
from collections.abc import Sequence
from dataclasses import dataclass

import matplotlib
import numpy as np

from utils.config import TARGET_WIN_RATE, WINDOW_SIZE


def use_file_only_backend() -> None:
    matplotlib.use("Agg")


use_file_only_backend()

import matplotlib.pyplot as plt

FIGURE_SIZE = (10, 5)
MATRIX_FIGURE_SIZE = (7, 5)
FIGURE_DPI = 150
RAW_REWARD_ALPHA = 0.3
GRID_ALPHA = 0.3
WIN_RATE_AXIS_TOP = 1.05
REWARD_COLOR = "tab:blue"
WIN_RATE_COLOR = "tab:red"
TARGET_COLOR = "green"
MATRIX_COLOR_MAP = "viridis"
ANNOTATION_CONTRAST_THRESHOLD = 0.5
DARK_TEXT_COLOR = "black"
LIGHT_TEXT_COLOR = "white"


@dataclass(frozen=True)
class EvaluationPoint:
    stage: int
    episode: int
    win_rate: float


def moving_average(values: Sequence[float], window: int = WINDOW_SIZE) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if len(values) < window:
        return values
    return np.convolve(values, np.ones(window) / window, mode="valid")


def learning_curve_path(save_dir: str, stage: int) -> str:
    return os.path.join(save_dir, f"learning_curve_level_{stage}.png")


def save_figure(figure: plt.Figure, path: str) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=FIGURE_DPI)
    plt.close(figure)
    return path


def draw_target_line(axis: plt.Axes) -> None:
    axis.axhline(y=TARGET_WIN_RATE, color=TARGET_COLOR, linestyle="--", label=f"Meta ({TARGET_WIN_RATE:.0%})")
    axis.set_ylim(0, WIN_RATE_AXIS_TOP)


def draw_rewards(axis: plt.Axes, rewards: Sequence[float]) -> None:
    count = len(rewards)
    axis.plot(np.arange(1, count + 1), rewards, color=REWARD_COLOR, alpha=RAW_REWARD_ALPHA, label="Recompensa por episodio")
    average = moving_average(rewards)
    axis.plot(
        np.arange(count - len(average) + 1, count + 1), average, color=REWARD_COLOR, label=f"Promedio móvil ({WINDOW_SIZE})"
    )
    axis.set_xlabel("Episodios del nivel (sin repaso)")
    axis.set_ylabel("Recompensa total", color=REWARD_COLOR)
    axis.tick_params(axis="y", labelcolor=REWARD_COLOR)
    axis.grid(alpha=GRID_ALPHA)


def draw_win_rates(axis: plt.Axes, win_rates: Sequence[float]) -> None:
    episodes = np.arange(1, len(win_rates) + 1)
    axis.plot(episodes, win_rates, color=WIN_RATE_COLOR, label=f"Win rate (últimos {WINDOW_SIZE})")
    draw_target_line(axis)
    axis.set_ylabel("Win rate", color=WIN_RATE_COLOR)
    axis.tick_params(axis="y", labelcolor=WIN_RATE_COLOR)


def merge_legends(primary: plt.Axes, secondary: plt.Axes) -> None:
    primary_lines, primary_labels = primary.get_legend_handles_labels()
    secondary_lines, secondary_labels = secondary.get_legend_handles_labels()
    primary.legend(primary_lines + secondary_lines, primary_labels + secondary_labels, loc="lower right")


def plot_learning_curve(rewards: Sequence[float], win_rates: Sequence[float], stage: int, save_dir: str) -> str:
    figure, reward_axis = plt.subplots(figsize=FIGURE_SIZE)
    draw_rewards(reward_axis, rewards)
    win_rate_axis = reward_axis.twinx()
    draw_win_rates(win_rate_axis, win_rates)
    merge_legends(reward_axis, win_rate_axis)
    reward_axis.set_title(f"Curva de aprendizaje - Nivel {stage}")
    return save_figure(figure, learning_curve_path(save_dir, stage))


def annotation_color(value: float) -> str:
    return LIGHT_TEXT_COLOR if value < ANNOTATION_CONTRAST_THRESHOLD else DARK_TEXT_COLOR


def annotate_cells(axis: plt.Axes, values: np.ndarray) -> None:
    for (row, col), value in np.ndenumerate(values):
        axis.text(col, row, f"{value:.2f}", ha="center", va="center", color=annotation_color(value))


def plot_curriculum_matrix(matrix: Sequence[Sequence[float]], level_ids: Sequence[int], path: str) -> str:
    values = np.asarray(matrix, dtype=float)
    figure, axis = plt.subplots(figsize=MATRIX_FIGURE_SIZE)
    image = axis.imshow(values, vmin=0.0, vmax=1.0, cmap=MATRIX_COLOR_MAP)
    annotate_cells(axis, values)
    axis.set_xticks(range(len(level_ids)), [f"Nivel {level_id}" for level_id in level_ids])
    axis.set_yticks(range(values.shape[0]), [f"Tras etapa {stage + 1}" for stage in range(values.shape[0])])
    axis.set_xlabel("Nivel evaluado")
    axis.set_ylabel("Etapa terminada")
    axis.set_title("Matriz curricular (win rate de la política voraz)")
    figure.colorbar(image, ax=axis)
    return save_figure(figure, path)


def draw_stage_series(axis: plt.Axes, evaluations: Sequence[EvaluationPoint]) -> None:
    for stage in sorted({point.stage for point in evaluations}):
        points = [point for point in evaluations if point.stage == stage]
        episodes = [point.episode for point in points]
        axis.plot(episodes, [point.win_rate for point in points], marker="o", label=f"Etapa {stage}")


def plot_eval_win_rate(evaluations: Sequence[EvaluationPoint], path: str) -> str:
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)
    draw_stage_series(axis, evaluations)
    draw_target_line(axis)
    axis.set_xlabel("Episodio dentro de la etapa")
    axis.set_ylabel("Win rate voraz (ε = 0)")
    axis.set_title("Evaluación periódica del nivel actual")
    axis.grid(alpha=GRID_ALPHA)
    axis.legend(loc="lower right")
    return save_figure(figure, path)
