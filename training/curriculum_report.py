from collections.abc import Sequence
from dataclasses import dataclass

from training.evaluation import EvaluationSummary


@dataclass(frozen=True)
class StageSummary:
    stage: int
    level_path: str
    passed: bool
    episodes: int
    best_win_rate: float | None


@dataclass(frozen=True)
class CurriculumReport:
    stages: tuple[StageSummary, ...]
    matrix: tuple[tuple[float, ...], ...]
    level_ids: tuple[int | None, ...]
    final_weights_path: str


def curriculum_matrix(evaluations_by_stage: Sequence[Sequence[EvaluationSummary]]) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(summary.win_rate for summary in stage) for stage in evaluations_by_stage)


def matrix_level_ids(evaluations_by_stage: Sequence[Sequence[EvaluationSummary]]) -> tuple[int | None, ...]:
    if not evaluations_by_stage:
        return ()
    return tuple(summary.level_id for summary in evaluations_by_stage[-1])
