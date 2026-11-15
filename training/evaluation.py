from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from agent.dqn_model import DQNAgent
from core.snake_env import SnakeEnv
from training.episode_runner import EpisodeResult, is_win, run_greedy_episode


@dataclass(frozen=True)
class EvaluationSummary:
    level_id: int | None
    win_rate: float
    mean_apples: float
    mean_steps: float
    reasons: dict[str, int]


def summarize_results(results: Sequence[EpisodeResult]) -> EvaluationSummary:
    if not results:
        raise ValueError("Se necesita al menos un episodio para resumir una evaluación.")
    count = len(results)
    return EvaluationSummary(
        level_id=results[0].level_id,
        win_rate=sum(is_win(result) for result in results) / count,
        mean_apples=sum(result.apples_eaten for result in results) / count,
        mean_steps=sum(result.steps for result in results) / count,
        reasons=dict(sorted(Counter(result.reason for result in results).items())),
    )


def should_evaluate(episode: int, every: int) -> bool:
    return episode % every == 0


def is_improvement(win_rate: float, best_win_rate: float | None) -> bool:
    return best_win_rate is None or win_rate > best_win_rate


def stage_is_passed(win_rate: float, target_win_rate: float) -> bool:
    return win_rate >= target_win_rate


def evaluate_level(agent: DQNAgent, env: SnakeEnv, episodes: int, base_seed: int) -> EvaluationSummary:
    results = [run_greedy_episode(env, agent, base_seed + offset) for offset in range(episodes)]
    return summarize_results(results)


def evaluate_all_levels(
    agent: DQNAgent, envs: Sequence[SnakeEnv], episodes: int, base_seed: int
) -> list[EvaluationSummary]:
    return [evaluate_level(agent, env, episodes, base_seed) for env in envs]
