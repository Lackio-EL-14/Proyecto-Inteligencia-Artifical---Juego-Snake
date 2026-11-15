import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from agent.dqn_model import DQNAgent
from agent.exploration import EpsilonSchedule
from core.experience import ExperienceReplay
from core.snake_env import SnakeEnv
from training import reporting
from training.checkpoints import BEST_TAG, LAST_TAG, final_checkpoint_path, save_stage_checkpoint
from training.curriculum import choose_level_for_episode
from training.curriculum_report import CurriculumReport, StageSummary, curriculum_matrix, matrix_level_ids
from training.episode_runner import StepHook, run_training_episode
from training.evaluation import (
    EvaluationSummary,
    evaluate_all_levels,
    evaluate_level,
    is_improvement,
    should_evaluate,
    stage_is_passed,
)
from training.learner import Learner
from training.metrics_log import (
    CURRICULUM_MATRIX_CSV_NAME,
    END_OF_STAGE_EVALUATION,
    EPISODES_CSV_NAME,
    EVALUATIONS_CSV_NAME,
    PERIODIC_EVALUATION,
    EpisodeCsvLogger,
    EpisodeRecord,
    EvaluationCsvLogger,
    EvaluationRecord,
    metrics_path,
    write_curriculum_matrix,
)
from training.stage_tracker import StageTracker, should_report_progress
from training.training_spec import TrainingSpec
from utils.plotting import EvaluationPoint, plot_curriculum_matrix, plot_eval_win_rate, plot_learning_curve

PLOTS_SUBDIR = "plots"
CURRICULUM_MATRIX_PLOT_NAME = "curriculum_matrix.png"
EVAL_WIN_RATE_PLOT_NAME = "eval_win_rate.png"

EnvFactory = Callable[[str, int], SnakeEnv]


@dataclass(frozen=True)
class TrainingComponents:
    agent: DQNAgent
    memory: ExperienceReplay
    env_factory: EnvFactory
    epsilon: EpsilonSchedule
    rng: np.random.Generator


@dataclass(frozen=True)
class TrainingSession:
    components: TrainingComponents
    spec: TrainingSpec
    train_envs: tuple[SnakeEnv, ...]
    eval_envs: tuple[SnakeEnv, ...]
    step_hook: StepHook
    episode_log: EpisodeCsvLogger
    evaluation_log: EvaluationCsvLogger
    evaluation_points: list[EvaluationPoint]


@dataclass
class StageState:
    stage_index: int
    tracker: StageTracker
    best_win_rate: float | None = None
    passed: bool = False
    episodes: int = 0

    @property
    def stage(self) -> int:
        return self.stage_index + 1


def plots_dir(run_dir: str) -> str:
    return os.path.join(run_dir, PLOTS_SUBDIR)


def build_environments(factory: EnvFactory, spec: TrainingSpec) -> tuple[SnakeEnv, ...]:
    return tuple(factory(path, spec.max_steps_per_episode) for path in spec.level_paths)


def seed_environments(envs: Sequence[SnakeEnv], seed: int) -> None:
    for offset, env in enumerate(envs):
        env.reset(seed=seed + offset)


def build_session(components: TrainingComponents, spec: TrainingSpec) -> TrainingSession:
    train_envs = build_environments(components.env_factory, spec)
    seed_environments(train_envs, spec.seed)
    learner = Learner(components.agent, components.memory, spec.learner)
    return TrainingSession(
        components=components,
        spec=spec,
        train_envs=train_envs,
        eval_envs=build_environments(components.env_factory, spec),
        step_hook=StepHook(components.memory, learner),
        episode_log=EpisodeCsvLogger(metrics_path(spec.run_dir, EPISODES_CSV_NAME)),
        evaluation_log=EvaluationCsvLogger(metrics_path(spec.run_dir, EVALUATIONS_CSV_NAME)),
        evaluation_points=[],
    )


def train_by_stage(components: TrainingComponents, spec: TrainingSpec) -> CurriculumReport:
    session = build_session(components, spec)
    reporting.print_training_start(spec.run_dir)
    stages: list[StageSummary] = []
    evaluations_by_stage: list[list[EvaluationSummary]] = []
    for stage_index in range(len(spec.level_paths)):
        summary, evaluations = run_curriculum_stage(session, stage_index)
        stages.append(summary)
        evaluations_by_stage.append(evaluations)
    return finish_training(session, stages, evaluations_by_stage)


def run_curriculum_stage(session: TrainingSession, stage_index: int) -> tuple[StageSummary, list[EvaluationSummary]]:
    if stage_index > 0:
        session.components.epsilon.restart_for_stage()
    reporting.print_stage_start(stage_index + 1, session.spec.level_paths[stage_index])
    summary = run_stage(session, stage_index)
    last_path = save_stage_checkpoint(session.components.agent, session.spec.run_dir, summary.stage, LAST_TAG)
    reporting.print_saved_file(last_path)
    evaluations = evaluate_end_of_stage(session, summary)
    reporting.print_stage_end(summary)
    return summary, evaluations


def run_stage(session: TrainingSession, stage_index: int) -> StageSummary:
    spec = session.spec
    state = StageState(stage_index, StageTracker(spec.window_size))
    while state.episodes < spec.max_episodes_per_stage and not state.passed:
        state.episodes += 1
        run_stage_episode(session, state)
        if should_evaluate(state.episodes, spec.eval_every):
            evaluate_current_level(session, state)
    plot_stage_curve(session, state)
    return StageSummary(state.stage, spec.level_paths[stage_index], state.passed, state.episodes, state.best_win_rate)


def run_stage_episode(session: TrainingSession, state: StageState) -> None:
    components, spec = session.components, session.spec
    level_index = choose_level_for_episode(state.stage_index, spec.rehearsal_prob, components.rng)
    epsilon = components.epsilon.value
    result = run_training_episode(session.train_envs[level_index], components.agent, epsilon, session.step_hook)
    components.epsilon.step()
    is_rehearsal = level_index != state.stage_index
    if not is_rehearsal:
        state.tracker.record(result)
    record = EpisodeRecord(state.stage, state.episodes, is_rehearsal, epsilon, state.tracker.train_win_rate(), result)
    session.episode_log.log(record)
    if should_report_progress(state.episodes, spec.progress_every):
        reporting.print_progress(record)


def evaluate_current_level(session: TrainingSession, state: StageState) -> None:
    spec, agent = session.spec, session.components.agent
    summary = evaluate_level(agent, session.eval_envs[state.stage_index], spec.eval_episodes, spec.eval_seed)
    record = EvaluationRecord(state.stage, state.episodes, PERIODIC_EVALUATION, summary)
    session.evaluation_log.log(record)
    session.evaluation_points.append(EvaluationPoint(state.stage, state.episodes, summary.win_rate))
    reporting.print_evaluation(record)
    if is_improvement(summary.win_rate, state.best_win_rate):
        state.best_win_rate = summary.win_rate
        reporting.print_saved_file(save_stage_checkpoint(agent, spec.run_dir, state.stage, BEST_TAG))
    state.passed = stage_is_passed(summary.win_rate, spec.target_win_rate)


def plot_stage_curve(session: TrainingSession, state: StageState) -> None:
    tracker = state.tracker
    path = plot_learning_curve(tracker.rewards(), tracker.win_rates(), state.stage, plots_dir(session.spec.run_dir))
    reporting.print_saved_file(path)


def evaluate_end_of_stage(session: TrainingSession, summary: StageSummary) -> list[EvaluationSummary]:
    spec = session.spec
    evaluations = evaluate_all_levels(session.components.agent, session.eval_envs, spec.eval_episodes, spec.eval_seed)
    for evaluation in evaluations:
        session.evaluation_log.log(EvaluationRecord(summary.stage, summary.episodes, END_OF_STAGE_EVALUATION, evaluation))
    reporting.print_evaluation_table(f"Evaluación de todos los niveles tras la etapa {summary.stage}:", evaluations)
    return evaluations


def save_curriculum_outputs(session: TrainingSession, report: CurriculumReport) -> None:
    run_dir = session.spec.run_dir
    write_curriculum_matrix(metrics_path(run_dir, CURRICULUM_MATRIX_CSV_NAME), report.matrix, report.level_ids)
    matrix_plot = os.path.join(plots_dir(run_dir), CURRICULUM_MATRIX_PLOT_NAME)
    reporting.print_saved_file(plot_curriculum_matrix(report.matrix, report.level_ids, matrix_plot))
    eval_plot = os.path.join(plots_dir(run_dir), EVAL_WIN_RATE_PLOT_NAME)
    reporting.print_saved_file(plot_eval_win_rate(session.evaluation_points, eval_plot))


def finish_training(
    session: TrainingSession, stages: list[StageSummary], evaluations_by_stage: list[list[EvaluationSummary]]
) -> CurriculumReport:
    final_path = final_checkpoint_path(session.spec.run_dir)
    session.components.agent.save(final_path)
    report = CurriculumReport(
        stages=tuple(stages),
        matrix=curriculum_matrix(evaluations_by_stage),
        level_ids=matrix_level_ids(evaluations_by_stage),
        final_weights_path=final_path,
    )
    save_curriculum_outputs(session, report)
    reporting.print_final_summary(report)
    return report
