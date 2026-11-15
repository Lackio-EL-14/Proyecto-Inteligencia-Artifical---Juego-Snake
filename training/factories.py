import os
from datetime import datetime

from agent.agent_spec import AgentSpec
from agent.exploration import EpsilonSchedule
from core.experience import ExperienceReplay
from core.snake_env import SnakeEnv
from training.training_spec import LearnerSpec, TrainingSpec
from utils.config import (
    ACTION_SIZE,
    BATCH_SIZE,
    EPSILON_DECAY,
    EPSILON_MIN,
    EPSILON_STAGE_START,
    EPSILON_START,
    EVAL_EPISODES,
    EVAL_EVERY,
    EVAL_SEED,
    GAMMA,
    GRADIENT_CLIPNORM,
    HUNGER_FACTOR,
    HIDDEN_UNITS,
    HUBER_DELTA,
    LEARNING_RATE,
    LEVEL_PATHS,
    MAX_EPISODES_PER_STAGE,
    MAX_MEMORY,
    MAX_STEPS_PER_EPISODE,
    PRIORITIZED_REPLAY,
    PROGRESS_EVERY,
    REHEARSAL_PROB,
    RUN_NAME_FORMAT,
    RUNS_DIR,
    STATE_SIZE,
    TARGET_RULE,
    TARGET_UPDATE_EVERY,
    TARGET_WIN_RATE,
    TRAIN_EVERY,
    USE_XLA,
    WARMUP_STEPS,
    WINDOW_SIZE,
)


def build_agent_spec() -> AgentSpec:
    return AgentSpec(
        state_size=STATE_SIZE,
        action_size=ACTION_SIZE,
        hidden_units=HIDDEN_UNITS,
        learning_rate=LEARNING_RATE,
        gradient_clipnorm=GRADIENT_CLIPNORM,
        gamma=GAMMA,
        huber_delta=HUBER_DELTA,
        target_rule=TARGET_RULE,
        use_xla=USE_XLA,
    )


def build_learner_spec() -> LearnerSpec:
    return LearnerSpec(
        batch_size=BATCH_SIZE,
        warmup_steps=WARMUP_STEPS,
        train_every=TRAIN_EVERY,
        target_update_every=TARGET_UPDATE_EVERY,
    )


def build_epsilon_schedule() -> EpsilonSchedule:
    return EpsilonSchedule(start=EPSILON_START, stage_start=EPSILON_STAGE_START, minimum=EPSILON_MIN, decay=EPSILON_DECAY)


def build_memory(seed: int) -> ExperienceReplay:
    return ExperienceReplay(max_memory=MAX_MEMORY, state_size=STATE_SIZE, prioritized=PRIORITIZED_REPLAY, seed=seed)


def make_environment(level_path: str, max_steps: int) -> SnakeEnv:
    return SnakeEnv(level_path, max_steps=max_steps, hunger_factor=HUNGER_FACTOR)


def build_training_spec(run_dir: str, seed: int) -> TrainingSpec:
    return TrainingSpec(
        level_paths=LEVEL_PATHS,
        learner=build_learner_spec(),
        max_steps_per_episode=MAX_STEPS_PER_EPISODE,
        max_episodes_per_stage=MAX_EPISODES_PER_STAGE,
        window_size=WINDOW_SIZE,
        target_win_rate=TARGET_WIN_RATE,
        eval_every=EVAL_EVERY,
        eval_episodes=EVAL_EPISODES,
        eval_seed=EVAL_SEED,
        rehearsal_prob=REHEARSAL_PROB,
        run_dir=run_dir,
        seed=seed,
        progress_every=PROGRESS_EVERY,
    )


def default_run_name(now: datetime) -> str:
    return now.strftime(RUN_NAME_FORMAT)


def run_directory(run_name: str) -> str:
    return os.path.join(RUNS_DIR, run_name)
