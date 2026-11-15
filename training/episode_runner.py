from collections.abc import Callable
from dataclasses import dataclass, field

import tensorflow as tf

from agent.dqn_model import DQNAgent, TrainResult
from core.experience import ExperienceReplay
from core.snake_env import SnakeEnv
from training.learner import Learner

WIN_REASON = "win"
UNKNOWN_REASON = "unknown"

FrameObserver = Callable[[SnakeEnv], None]


@dataclass(frozen=True)
class EpisodeResult:
    level_id: int | None
    total_reward: float
    apples_eaten: int
    steps: int
    reason: str
    mean_loss: float | None


@dataclass(frozen=True)
class StepHook:
    memory: ExperienceReplay
    learner: Learner


def is_win(result: EpisodeResult) -> bool:
    return result.reason == WIN_REASON


def mean_loss(losses: list[tf.Tensor]) -> float | None:
    if not losses:
        return None
    return float(tf.reduce_mean(tf.stack(losses)))


@dataclass
class EpisodeTally:
    total_reward: float = 0.0
    steps: int = 0
    losses: list[tf.Tensor] = field(default_factory=list)
    last_info: dict = field(default_factory=dict)

    def record_step(self, reward: float, info: dict) -> None:
        self.total_reward += float(reward)
        self.steps += 1
        self.last_info = info

    def record_training(self, result: TrainResult | None) -> None:
        if result is not None:
            self.losses.append(result.loss)

    def to_result(self, level_id: int | None) -> EpisodeResult:
        return EpisodeResult(
            level_id=level_id,
            total_reward=self.total_reward,
            apples_eaten=int(self.last_info.get("apples_eaten", 0)),
            steps=self.steps,
            reason=str(self.last_info.get("reason", UNKNOWN_REASON)),
            mean_loss=mean_loss(self.losses),
        )


def run_training_episode(env: SnakeEnv, agent: DQNAgent, epsilon: float, step_hook: StepHook) -> EpisodeResult:
    state, reset_info = env.reset()
    tally = EpisodeTally()
    episode_done = False
    while not episode_done:
        action = agent.act(state, epsilon)
        next_state, reward, terminated, truncated, info = env.step(action)
        step_hook.memory.remember(state, action, reward, next_state, terminated)
        tally.record_training(step_hook.learner.after_env_step())
        tally.record_step(reward, info)
        state, episode_done = next_state, terminated or truncated
    return tally.to_result(reset_info.get("level_id"))


def ignore_frame(env: SnakeEnv) -> None:
    return None


def watch_greedy_episode(env: SnakeEnv, agent: DQNAgent, seed: int, observer: FrameObserver) -> EpisodeResult:
    state, reset_info = env.reset(seed=seed)
    observer(env)
    tally = EpisodeTally()
    episode_done = False
    while not episode_done:
        next_state, reward, terminated, truncated, info = env.step(agent.greedy_action(state))
        observer(env)
        tally.record_step(reward, info)
        state, episode_done = next_state, terminated or truncated
    return tally.to_result(reset_info.get("level_id"))


def run_greedy_episode(env: SnakeEnv, agent: DQNAgent, seed: int) -> EpisodeResult:
    return watch_greedy_episode(env, agent, seed, ignore_frame)
