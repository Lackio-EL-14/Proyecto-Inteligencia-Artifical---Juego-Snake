import argparse
import csv

from agent import greedy_action, load_network
from config import EVAL_EPISODES, EVAL_SEED, LEVEL_PATHS
from snake_env import SnakeEnv, next_cell
from state import build_state
from tensorflow.keras import Sequential

EPISODE_FIELDS = ["level", "seed", "reason", "steps", "apples", "final_cell"]


def final_cell(env: SnakeEnv, reason: str) -> tuple[int, int]:
    if reason in ("wall", "obstacle", "body"):
        return next_cell(env.head, env.direction)
    return env.head


def greedy_episode(env: SnakeEnv, network: Sequential, seed: int) -> dict:
    env.reset(seed)
    done = truncated = False
    steps, reason = 0, ""
    while not (done or truncated):
        _, done, truncated, reason = env.step(greedy_action(network, build_state(env)))
        steps += 1
    return {"level": env.level_id, "seed": seed, "reason": reason, "steps": steps, "apples": env.apples_eaten, "final_cell": final_cell(env, reason)}


def evaluate_level(network: Sequential, level_path: str, episodes: int, first_seed: int) -> list[dict]:
    env = SnakeEnv(level_path)
    return [greedy_episode(env, network, first_seed + index) for index in range(episodes)]


def win_rate(results: list[dict]) -> float:
    return sum(result["reason"] == "win" for result in results) / len(results)


def reason_counts(results: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for result in results:
        counts[result["reason"]] = counts.get(result["reason"], 0) + 1
    return counts


def evaluate_all(network: Sequential, level_paths: list[str], episodes: int) -> list[dict]:
    results = []
    for level_path in level_paths:
        level_results = evaluate_level(network, level_path, episodes, EVAL_SEED)
        print(f"{level_path:<22} win_rate {win_rate(level_results):.3f}  {reason_counts(level_results)}")
        results.extend(level_results)
    return results


def save_results(results: list[dict], path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=EPISODE_FIELDS)
        writer.writeheader()
        writer.writerows(results)


def evaluation_path(weights_path: str) -> str:
    return weights_path.replace(".weights.h5", "_evaluation.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--levels", type=int, nargs="+", default=list(range(1, len(LEVEL_PATHS) + 1)))
    parser.add_argument("--episodes", type=int, default=EVAL_EPISODES)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = evaluate_all(load_network(args.weights), [LEVEL_PATHS[level - 1] for level in args.levels], args.episodes)
    output_path = evaluation_path(args.weights)
    save_results(results, output_path)
    print(f"guardado: {output_path}")


if __name__ == "__main__":
    main()
