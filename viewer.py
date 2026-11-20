import argparse
import csv
from dataclasses import dataclass, field

import numpy as np
import pygame

from agent import load_network, q_values
from evaluate import evaluation_path
from baselines import heuristic_policy, random_policy
from config import EVAL_EPISODES, EVAL_SEED, LEVEL_PATHS
from snake_env import Cell, SnakeEnv, next_cell
from state import build_state, move_view, relative_cells

CELL_PX = 44
PANEL_PX = 280
FPS_START = 8
FPS_LIMITS = (1, 60)
ACTION_NAMES = ("recto", "derecha", "izquierda")
POLICIES = {"heuristic": heuristic_policy, "random": random_policy}
COLORS = {
    "background": (24, 26, 33),
    "grid": (44, 47, 58),
    "obstacle": (120, 124, 140),
    "head": (80, 200, 120),
    "body": (40, 140, 80),
    "apple": (230, 70, 70),
    "trail": (70, 90, 160),
    "blocked": (200, 60, 60),
    "closer": (230, 200, 60),
    "neutral": (90, 160, 230),
    "crash": (255, 140, 0),
    "text": (230, 230, 235),
}


@dataclass
class Session:
    env: SnakeEnv
    policy_name: str
    seed: int
    rng: np.random.Generator
    seeds: list[int]
    network: object = None
    seed_index: int = 0
    trail: list[Cell] = field(default_factory=list)
    reason: str = ""
    action: int | None = None
    q_values: np.ndarray | None = None
    finished: bool = False
    paused: bool = False
    step_once: bool = False
    fps: int = FPS_START


def cell_rect(cell: Cell) -> pygame.Rect:
    return pygame.Rect(cell[1] * CELL_PX, cell[0] * CELL_PX, CELL_PX, CELL_PX)


def fill_cell(screen: pygame.Surface, cell: Cell, color: tuple[int, int, int], inset: int = 1) -> None:
    pygame.draw.rect(screen, color, cell_rect(cell).inflate(-2 * inset, -2 * inset))


def draw_grid(screen: pygame.Surface, env: SnakeEnv) -> None:
    for row in range(env.rows):
        for col in range(env.cols):
            pygame.draw.rect(screen, COLORS["grid"], cell_rect((row, col)), 1)
    for cell in env.obstacles:
        fill_cell(screen, cell, COLORS["obstacle"])


def draw_trail(screen: pygame.Surface, trail: list[Cell]) -> None:
    for cell in trail:
        pygame.draw.circle(screen, COLORS["trail"], cell_rect(cell).center, CELL_PX // 6)


def draw_snake(screen: pygame.Surface, env: SnakeEnv) -> None:
    for cell in env.body[1:]:
        fill_cell(screen, cell, COLORS["body"], 4)
    fill_cell(screen, env.head, COLORS["head"], 2)


def draw_apple(screen: pygame.Surface, env: SnakeEnv) -> None:
    if env.apple is not None:
        pygame.draw.circle(screen, COLORS["apple"], cell_rect(env.apple).center, CELL_PX // 3)


def sensor_color(blocked: float, closer: float) -> tuple[int, int, int]:
    if blocked:
        return COLORS["blocked"]
    return COLORS["closer"] if closer else COLORS["neutral"]


def draw_sensors(screen: pygame.Surface, env: SnakeEnv, state: np.ndarray) -> None:
    moves = move_view(state)
    for cell, (blocked, closer) in zip(relative_cells(env), moves):
        pygame.draw.rect(screen, sensor_color(blocked, closer), cell_rect(cell), 3)


def draw_crash(screen: pygame.Surface, session: Session) -> None:
    if session.finished and session.reason in ("wall", "obstacle", "body"):
        crash_cell = next_cell(session.env.head, session.env.direction)
        pygame.draw.rect(screen, COLORS["crash"], cell_rect(crash_cell), 5)


def q_value_lines(session: Session) -> list[str]:
    if session.q_values is None:
        return [f"{name}" + (" <" if index == session.action else "") for index, name in enumerate(ACTION_NAMES)]
    pairs = zip(ACTION_NAMES, session.q_values)
    return [f"Q {name:<9} {value:+.3f}" + (" <" if index == session.action else "") for index, (name, value) in enumerate(pairs)]


def hud_lines(session: Session) -> list[str]:
    env = session.env
    header = [
        f"nivel {env.level_id}  politica {session.policy_name}",
        f"semilla {session.seed}  ({session.seed_index + 1}/{len(session.seeds)})",
        f"manzanas {env.apples_eaten}/{env.apples_to_win}",
        f"hambre {env.steps_since_apple}/{env.hunger_limit}",
        f"fps {session.fps}{'  PAUSA' if session.paused else ''}",
        "",
    ]
    footer = ["", f"fin: {session.reason}" if session.finished else "", "", "espacio pausa  n paso", "+/- velocidad  r siguiente", "esc salir"]
    return header + q_value_lines(session) + footer


def draw_hud(screen: pygame.Surface, font: pygame.font.Font, session: Session) -> None:
    left = session.env.cols * CELL_PX + 12
    for index, line in enumerate(hud_lines(session)):
        screen.blit(font.render(line, True, COLORS["text"]), (left, 12 + index * 22))


def draw_frame(screen: pygame.Surface, font: pygame.font.Font, session: Session) -> None:
    screen.fill(COLORS["background"])
    draw_grid(screen, session.env)
    draw_trail(screen, session.trail)
    draw_apple(screen, session.env)
    draw_snake(screen, session.env)
    if not session.finished:
        draw_sensors(screen, session.env, build_state(session.env))
    draw_crash(screen, session)
    draw_hud(screen, font, session)
    pygame.display.flip()


def start_episode(session: Session, seed: int) -> None:
    session.seed = seed
    session.env.reset(seed)
    session.trail, session.reason, session.action, session.finished = [], "", None, False


def next_episode(session: Session) -> None:
    session.seed_index = (session.seed_index + 1) % len(session.seeds)
    start_episode(session, session.seeds[session.seed_index])


def choose_action(session: Session, state: np.ndarray) -> int:
    if session.network is None:
        return POLICIES[session.policy_name](state, session.rng)
    session.q_values = q_values(session.network, state)[0]
    return int(np.argmax(session.q_values))


def advance(session: Session) -> None:
    state = build_state(session.env)
    session.action = choose_action(session, state)
    session.trail.append(session.env.head)
    _, done, truncated, session.reason = session.env.step(session.action)
    if session.reason == "apple":
        session.trail = []
    session.finished = done or truncated


def change_speed(session: Session, factor: float) -> None:
    session.fps = int(np.clip(round(session.fps * factor), *FPS_LIMITS))


def handle_key(session: Session, key: int) -> bool:
    if key == pygame.K_ESCAPE:
        return False
    if key == pygame.K_SPACE:
        session.paused = not session.paused
    if key == pygame.K_n:
        session.step_once = True
    if key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
        change_speed(session, 2.0)
    if key in (pygame.K_MINUS, pygame.K_KP_MINUS):
        change_speed(session, 0.5)
    if key == pygame.K_r:
        next_episode(session)
    return True


def handle_events(session: Session) -> bool:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return False
        if event.type == pygame.KEYDOWN and not handle_key(session, event.key):
            return False
    return True


def should_advance(session: Session) -> bool:
    return not session.finished and (not session.paused or session.step_once)


def run_viewer(session: Session) -> None:
    pygame.init()
    size = (session.env.cols * CELL_PX + PANEL_PX, max(session.env.rows * CELL_PX, 420))
    screen, font, clock = pygame.display.set_mode(size), pygame.font.SysFont("monospace", 16), pygame.time.Clock()
    pygame.display.set_caption("Snake Double DQN")
    while handle_events(session):
        if should_advance(session):
            advance(session)
            session.step_once = False
        draw_frame(screen, font, session)
        clock.tick(session.fps)
    pygame.quit()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=int, choices=range(1, len(LEVEL_PATHS) + 1), default=1)
    parser.add_argument("--policy", choices=sorted(POLICIES) + ["agent"], default="heuristic")
    parser.add_argument("--weights")
    parser.add_argument("--seed", type=int, default=EVAL_SEED)
    parser.add_argument("--episodes", type=int, default=EVAL_EPISODES)
    parser.add_argument("--failures", action="store_true")
    return parser.parse_args()


def failed_seeds(weights_path: str, level_id: int) -> list[int]:
    with open(evaluation_path(weights_path), encoding="utf-8") as csv_file:
        rows = csv.DictReader(csv_file)
        return [int(row["seed"]) for row in rows if int(row["level"]) == level_id and row["reason"] != "win"]


def episode_seeds(args: argparse.Namespace) -> list[int]:
    if args.failures:
        return failed_seeds(args.weights, args.level) or [args.seed]
    return list(range(args.seed, args.seed + args.episodes))


def policy_network(args: argparse.Namespace) -> object:
    if args.policy != "agent":
        return None
    return load_network(args.weights)


def main() -> None:
    args = parse_args()
    env = SnakeEnv(LEVEL_PATHS[args.level - 1])
    seeds = episode_seeds(args)
    session = Session(env, args.policy, seeds[0], np.random.default_rng(args.seed), seeds, policy_network(args))
    start_episode(session, seeds[0])
    run_viewer(session)


if __name__ == "__main__":
    main()
