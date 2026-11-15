import time
from collections.abc import Sequence

from training.curriculum_report import CurriculumReport, StageSummary
from training.episode_runner import EpisodeResult
from training.evaluation import EvaluationSummary
from training.metrics_log import EpisodeRecord, EvaluationRecord, format_reasons

SEPARATOR_WIDTH = 40
CELL_WIDTH = 9


def separator() -> str:
    return "=" * SEPARATOR_WIDTH


def format_progress(record: EpisodeRecord) -> str:
    result = record.result
    kind = " (repaso)" if record.is_rehearsal else ""
    return (
        f"Etapa {record.stage} | Episodio {record.episode}{kind} | Nivel {result.level_id} | "
        f"Epsilon: {record.epsilon:.3f} | Reward: {result.total_reward:.2f} | Win Rate: {record.train_win_rate:.2f}"
    )


def format_evaluation(record: EvaluationRecord) -> str:
    summary = record.summary
    return (
        f"  Evaluación voraz | Etapa {record.stage} | Episodio {record.episode} | Nivel {summary.level_id} | "
        f"Win Rate: {summary.win_rate:.2f} | Manzanas: {summary.mean_apples:.2f} | Motivos: {format_reasons(summary.reasons)}"
    )


def format_best_win_rate(best_win_rate: float | None) -> str:
    return "sin evaluar" if best_win_rate is None else f"{best_win_rate:.2f}"


def format_stage_end(summary: StageSummary) -> str:
    if summary.passed:
        return f"¡Etapa {summary.stage} superada en {summary.episodes} episodios!"
    return (
        f"Etapa {summary.stage} NO superada tras {summary.episodes} episodios "
        f"(mejor win rate voraz: {format_best_win_rate(summary.best_win_rate)}). Se continúa con la siguiente."
    )


def format_evaluation_table(summaries: Sequence[EvaluationSummary]) -> str:
    header = f"{'Nivel':>{CELL_WIDTH}}{'Win rate':>{CELL_WIDTH}}{'Manzanas':>{CELL_WIDTH}}{'Pasos':>{CELL_WIDTH}}  Motivos"
    rows = [
        f"{summary.level_id!s:>{CELL_WIDTH}}{summary.win_rate:>{CELL_WIDTH}.2f}{summary.mean_apples:>{CELL_WIDTH}.2f}"
        f"{summary.mean_steps:>{CELL_WIDTH}.1f}  {format_reasons(summary.reasons)}"
        for summary in summaries
    ]
    return "\n".join([header, *rows])


def format_matrix_table(matrix: Sequence[Sequence[float]], level_ids: Sequence[int]) -> str:
    header = f"{'Etapa':>{CELL_WIDTH}}" + "".join(f"{f'Nivel {level_id}':>{CELL_WIDTH}}" for level_id in level_ids)
    rows = [
        f"{stage_index + 1:>{CELL_WIDTH}}" + "".join(f"{value:>{CELL_WIDTH}.2f}" for value in row)
        for stage_index, row in enumerate(matrix)
    ]
    return "\n".join([header, *rows])


def format_episode_result(result: EpisodeResult) -> str:
    return f"Fin de la partida: {result.reason} | Manzanas: {result.apples_eaten} | Pasos: {result.steps}"


def print_training_start(run_dir: str) -> None:
    print(f"Iniciando entrenamiento curricular. Resultados en '{run_dir}'")


def print_stage_start(stage: int, level_path: str) -> None:
    print(f"\n{separator()}\nIniciando Entrenamiento: ETAPA {stage} ({level_path})\n{separator()}")


def print_progress(record: EpisodeRecord) -> None:
    print(format_progress(record))


def print_evaluation(record: EvaluationRecord) -> None:
    print(format_evaluation(record))


def print_stage_end(summary: StageSummary) -> None:
    print(format_stage_end(summary))


def print_saved_file(path: str) -> None:
    print(f"Archivo guardado en '{path}'")


def print_evaluation_table(title: str, summaries: Sequence[EvaluationSummary]) -> None:
    print(f"\n{title}\n{format_evaluation_table(summaries)}")


def print_final_summary(report: CurriculumReport) -> None:
    print("\n¡Entrenamiento Curricular completado!")
    print(f"Matriz curricular (win rate voraz):\n{format_matrix_table(report.matrix, report.level_ids)}")
    print(f"Modelo guardado en '{report.final_weights_path}'")


def print_frame(board: str, delay_seconds: float) -> None:
    print(f"\n{board}")
    time.sleep(delay_seconds)


def print_episode_result(result: EpisodeResult) -> None:
    print(format_episode_result(result))
