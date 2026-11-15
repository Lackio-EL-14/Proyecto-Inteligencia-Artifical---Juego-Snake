from dataclasses import dataclass


@dataclass(frozen=True)
class LearnerSpec:
    batch_size: int
    warmup_steps: int
    train_every: int
    target_update_every: int


@dataclass(frozen=True)
class TrainingSpec:
    level_paths: tuple[str, ...]
    learner: LearnerSpec
    max_steps_per_episode: int
    max_episodes_per_stage: int
    window_size: int
    target_win_rate: float
    eval_every: int
    eval_episodes: int
    eval_seed: int
    rehearsal_prob: float
    run_dir: str
    seed: int
    progress_every: int
