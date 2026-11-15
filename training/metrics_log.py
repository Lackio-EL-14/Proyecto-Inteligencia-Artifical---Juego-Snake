import csv
import os
from collections.abc import Sequence
from dataclasses import dataclass

from training.episode_runner import EpisodeResult
from training.evaluation import EvaluationSummary

METRICS_SUBDIR = "metrics"
EPISODES_CSV_NAME = "episodes.csv"
EVALUATIONS_CSV_NAME = "evaluations.csv"
CURRICULUM_MATRIX_CSV_NAME = "curriculum_matrix.csv"
PERIODIC_EVALUATION = "periodica"
END_OF_STAGE_EVALUATION = "fin_etapa"
EPISODE_FIELDS = (
    "stage", "episode", "level_id", "is_rehearsal", "total_reward", "apples_eaten",
    "steps", "reason", "epsilon", "mean_loss", "train_win_rate",
)
EVALUATION_FIELDS = ("stage", "episode", "kind", "level_id", "win_rate", "mean_apples", "mean_steps", "reasons")
MATRIX_STAGE_FIELD = "stage_terminada"
REASON_SEPARATOR = ";"
DECIMALS = 4


@dataclass(frozen=True)
class EpisodeRecord:
    stage: int
    episode: int
    is_rehearsal: bool
    epsilon: float
    train_win_rate: float
    result: EpisodeResult


@dataclass(frozen=True)
class EvaluationRecord:
    stage: int
    episode: int
    kind: str
    summary: EvaluationSummary


def metrics_path(run_dir: str, filename: str) -> str:
    return os.path.join(run_dir, METRICS_SUBDIR, filename)


def format_number(value: float | None) -> str:
    return "" if value is None else f"{value:.{DECIMALS}f}"


def format_reasons(reasons: dict[str, int]) -> str:
    return REASON_SEPARATOR.join(f"{reason}:{count}" for reason, count in reasons.items())


def episode_row(record: EpisodeRecord) -> dict[str, object]:
    result = record.result
    values = (
        record.stage, record.episode, result.level_id, int(record.is_rehearsal), format_number(result.total_reward),
        result.apples_eaten, result.steps, result.reason, format_number(record.epsilon),
        format_number(result.mean_loss), format_number(record.train_win_rate),
    )
    return dict(zip(EPISODE_FIELDS, values))


def evaluation_row(record: EvaluationRecord) -> dict[str, object]:
    summary = record.summary
    values = (
        record.stage, record.episode, record.kind, summary.level_id, format_number(summary.win_rate),
        format_number(summary.mean_apples), format_number(summary.mean_steps), format_reasons(summary.reasons),
    )
    return dict(zip(EVALUATION_FIELDS, values))


def matrix_fields(level_ids: Sequence[int]) -> tuple[str, ...]:
    return (MATRIX_STAGE_FIELD, *(f"nivel_{level_id}" for level_id in level_ids))


def matrix_rows(matrix: Sequence[Sequence[float]], level_ids: Sequence[int]) -> list[dict[str, object]]:
    fields = matrix_fields(level_ids)
    return [
        dict(zip(fields, (stage_index + 1, *(format_number(value) for value in row))))
        for stage_index, row in enumerate(matrix)
    ]


class CsvLogger:
    def __init__(self, path: str, fields: tuple[str, ...]) -> None:
        self.path = path
        self.fields = fields
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=fields).writeheader()

    def append(self, row: dict[str, object]) -> None:
        with open(self.path, "a", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=self.fields).writerow(row)


class EpisodeCsvLogger(CsvLogger):
    def __init__(self, path: str) -> None:
        super().__init__(path, EPISODE_FIELDS)

    def log(self, record: EpisodeRecord) -> None:
        self.append(episode_row(record))


class EvaluationCsvLogger(CsvLogger):
    def __init__(self, path: str) -> None:
        super().__init__(path, EVALUATION_FIELDS)

    def log(self, record: EvaluationRecord) -> None:
        self.append(evaluation_row(record))


def write_curriculum_matrix(path: str, matrix: Sequence[Sequence[float]], level_ids: Sequence[int]) -> None:
    logger = CsvLogger(path, matrix_fields(level_ids))
    for row in matrix_rows(matrix, level_ids):
        logger.append(row)
