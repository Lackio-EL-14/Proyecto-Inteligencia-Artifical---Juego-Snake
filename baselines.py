import csv
import os
from collections import Counter
from collections.abc import Callable

import numpy as np

from config import ACTION_COUNT, BASELINE_EPISODES, EVAL_SEED, LEVEL_PATHS, RUNS_DIR
from snake_env import SnakeEnv
from state import build_state, move_view

Policy = Callable[[np.ndarray, np.random.Generator], int]
BLOCKED, CLOSER = 0, 1
BASELINES_CSV = os.path.join(RUNS_DIR, "baselines.csv")


def random_policy(state: np.ndarray, rng: np.random.Generator) -> int:
    return int(rng.integers(ACTION_COUNT))


def heuristic_policy(state: np.ndarray, rng: np.random.Generator) -> int:
    moves = move_view(state)
    free = [action for action in range(ACTION_COUNT) if moves[action, BLOCKED] == 0]
    closer = [action for action in free if moves[action, CLOSER] == 1]
    return (closer or free or [0])[0]


def run_episode(env: SnakeEnv, policy: Policy, seed: int, rng: np.random.Generator) -> str:
    env.reset(seed)
    done = truncated = False
    reason = ""
    while not (done or truncated):
        _, done, truncated, reason = env.step(policy(build_state(env), rng))
    return reason


def episode_reasons(level_path: str, policy: Policy, episodes: int) -> Counter:
    env, rng = SnakeEnv(level_path), np.random.default_rng(EVAL_SEED)
    return Counter(run_episode(env, policy, EVAL_SEED + index, rng) for index in range(episodes))


def win_rate(reasons: Counter) -> float:
    return reasons["win"] / sum(reasons.values())


def measure(policies: dict[str, Policy], episodes: int) -> list[dict]:
    rows = []
    for name, policy in policies.items():
        for level_path in LEVEL_PATHS:
            reasons = episode_reasons(level_path, policy, episodes)
            rows.append({"policy": name, "level": level_path, "win_rate": win_rate(reasons), "reasons": dict(reasons)})
    return rows


def print_table(rows: list[dict]) -> None:
    print(f"{'policy':<10} {'level':<22} {'win_rate':>8}  reasons")
    for row in rows:
        print(f"{row['policy']:<10} {row['level']:<22} {row['win_rate']:>8.3f}  {row['reasons']}")


def save_csv(rows: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["policy", "level", "win_rate", "reasons"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    rows = measure({"random": random_policy, "heuristic": heuristic_policy}, BASELINE_EPISODES)
    print_table(rows)
    save_csv(rows, BASELINES_CSV)


if __name__ == "__main__":
    main()
