import os

from agent.dqn_model import DQNAgent

MODELS_SUBDIR = "models"
BEST_TAG = "best"
LAST_TAG = "last"
CHECKPOINT_TAGS = (BEST_TAG, LAST_TAG)
WEIGHTS_SUFFIX = ".weights.h5"
FINAL_CHECKPOINT_NAME = "final"


def stage_checkpoint_path(run_dir: str, stage: int, tag: str) -> str:
    if tag not in CHECKPOINT_TAGS:
        raise ValueError(f"Etiqueta de checkpoint desconocida: '{tag}'. Opciones: {CHECKPOINT_TAGS}")
    return os.path.join(run_dir, MODELS_SUBDIR, f"stage_{stage}_{tag}{WEIGHTS_SUFFIX}")


def final_checkpoint_path(run_dir: str) -> str:
    return os.path.join(run_dir, MODELS_SUBDIR, f"{FINAL_CHECKPOINT_NAME}{WEIGHTS_SUFFIX}")


def save_stage_checkpoint(agent: DQNAgent, run_dir: str, stage: int, tag: str) -> str:
    path = stage_checkpoint_path(run_dir, stage, tag)
    agent.save(path)
    return path
