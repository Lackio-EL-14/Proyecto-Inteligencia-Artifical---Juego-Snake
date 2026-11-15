import numpy as np


def choose_level_for_episode(stage_index: int, rehearsal_prob: float, rng: np.random.Generator) -> int:
    if stage_index == 0 or rng.random() >= rehearsal_prob:
        return stage_index
    return int(rng.integers(stage_index))
