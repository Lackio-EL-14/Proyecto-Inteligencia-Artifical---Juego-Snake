import argparse
import csv
import os
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from agent import ReplayMemory, build_q_network, epsilon_at, epsilon_greedy, set_seed, sync_target, train_step
from config import (
    BATCH_SIZE,
    EPISODES,
    GAMMA,
    LEVEL_PATHS,
    MEMORY_SIZE,
    PROGRESS_EVERY,
    RUNS_DIR,
    SEED,
    TARGET_SYNC_STEPS,
    TRAIN_EVAL_EPISODES,
    TRAIN_EVAL_EVERY,
    TRAIN_EVAL_SEED,
    WARMUP_STEPS,
)
from evaluate import evaluate_level, win_rate
from snake_env import SnakeEnv
from state import build_state
from tensorflow.keras import Sequential

EPISODE_FIELDS = ["episode", "level", "reward", "steps", "apples", "reason", "epsilon", "loss"]
EVAL_FIELDS = ["episode", "level", "win_rate"]


@dataclass
class Learner:
    online: Sequential
    target: Sequential
    memory: ReplayMemory
    rng: np.random.Generator
    train_steps: int = 0
    losses: list[float] = field(default_factory=list)


@dataclass
class EpisodeResult:
    reward: float = 0.0
    steps: int = 0
    reason: str = ""


def make_learner(seed: int) -> Learner:
    set_seed(seed)
    online, target = build_q_network(), build_q_network()
    sync_target(online, target)
    return Learner(online, target, ReplayMemory(MEMORY_SIZE), np.random.default_rng(seed))


def make_envs(levels: list[int]) -> list[SnakeEnv]:
    return [SnakeEnv(LEVEL_PATHS[level - 1]) for level in levels]


def choose_level(envs: list[SnakeEnv], rng: np.random.Generator) -> SnakeEnv:
    return envs[int(rng.integers(len(envs)))]


def learn(learner: Learner) -> None:
    if len(learner.memory) < max(WARMUP_STEPS, BATCH_SIZE):
        return
    batch = learner.memory.sample(BATCH_SIZE, learner.rng)
    learner.losses.append(train_step(learner.online, learner.target, batch, GAMMA))
    learner.train_steps += 1
    if learner.train_steps % TARGET_SYNC_STEPS == 0:
        sync_target(learner.online, learner.target)


def run_training_episode(env: SnakeEnv, learner: Learner, epsilon: float, seed: int) -> EpisodeResult:
    env.reset(seed)
    result, state = EpisodeResult(), build_state(env)
    done = truncated = False
    while not (done or truncated):
        action = epsilon_greedy(learner.online, state, epsilon, learner.rng)
        reward, done, truncated, result.reason = env.step(action)
        next_state = build_state(env)
        learner.memory.add(state, action, reward, next_state, done)
        learn(learner)
        state, result.reward, result.steps = next_state, result.reward + reward, result.steps + 1
    return result


def append_row(path: str, fields: list[str], row: dict) -> None:
    is_new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def episode_row(episode: int, env: SnakeEnv, result: EpisodeResult, epsilon: float, learner: Learner) -> dict:
    loss = float(np.mean(learner.losses)) if learner.losses else float("nan")
    learner.losses.clear()
    return {
        "episode": episode, "level": env.level_id, "reward": round(result.reward, 3), "steps": result.steps,
        "apples": env.apples_eaten, "reason": result.reason, "epsilon": round(epsilon, 3), "loss": round(loss, 5),
    }


def print_progress(row: dict) -> None:
    print(f"ep {row['episode']:>5}  nivel {row['level']}  {row['reason']:<10} recompensa {row['reward']:>7.2f}  eps {row['epsilon']:.2f}  loss {row['loss']:.4f}")


def periodic_evaluation(learner: Learner, envs: list[SnakeEnv], episode: int, run_dir: str) -> float:
    rates = []
    for env in envs:
        rate = win_rate(evaluate_level(learner.online, LEVEL_PATHS[env.level_id - 1], TRAIN_EVAL_EPISODES, TRAIN_EVAL_SEED))
        append_row(os.path.join(run_dir, "evaluations.csv"), EVAL_FIELDS, {"episode": episode, "level": env.level_id, "win_rate": rate})
        rates.append(rate)
    print(f"evaluacion ep {episode}: " + "  ".join(f"N{env.level_id}={rate:.2f}" for env, rate in zip(envs, rates)))
    return float(np.mean(rates))


def save_if_best(learner: Learner, score: float, best_score: float, run_dir: str) -> float:
    if score < best_score:
        return best_score
    learner.online.save_weights(os.path.join(run_dir, "best.weights.h5"))
    return score


def train(levels: list[int], episodes: int, run_dir: str) -> None:
    learner, envs = make_learner(SEED), make_envs(levels)
    best_score = -1.0
    for episode in range(1, episodes + 1):
        env, epsilon = choose_level(envs, learner.rng), epsilon_at(episode, episodes)
        result = run_training_episode(env, learner, epsilon, SEED + episode)
        row = episode_row(episode, env, result, epsilon, learner)
        append_row(os.path.join(run_dir, "episodes.csv"), EPISODE_FIELDS, row)
        if episode % PROGRESS_EVERY == 0:
            print_progress(row)
        if episode % TRAIN_EVAL_EVERY == 0:
            best_score = save_if_best(learner, periodic_evaluation(learner, envs, episode, run_dir), best_score, run_dir)
    learner.online.save_weights(os.path.join(run_dir, "final.weights.h5"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--levels", type=int, nargs="+", default=list(range(1, len(LEVEL_PATHS) + 1)))
    parser.add_argument("--episodes", type=int, default=EPISODES)
    parser.add_argument("--run-name", default=datetime.now().strftime("%Y%m%d-%H%M%S"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = os.path.join(RUNS_DIR, args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    train(args.levels, args.episodes, run_dir)
    print(f"pesos guardados en {run_dir}")


if __name__ == "__main__":
    main()
