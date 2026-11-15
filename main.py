import argparse
from datetime import datetime

from agent.dqn_model import DQNAgent
from training import reporting
from core.snake_env import SnakeEnv
from training.episode_runner import FrameObserver, watch_greedy_episode
from training.evaluation import evaluate_all_levels
from training.factories import (
    build_agent_spec,
    build_epsilon_schedule,
    build_memory,
    build_training_spec,
    default_run_name,
    make_environment,
    run_directory,
)
from training.train import TrainingComponents, train_by_stage
from utils.config import EVAL_EPISODES, EVAL_SEED, FORCE_CPU, LEVEL_PATHS, MAX_STEPS_PER_EPISODE, PLAY_DELAY_SECONDS, SEED
from utils.console_render import render_board
from utils.device import use_cpu_only
from utils.seeding import seed_everything


def configure_device() -> None:
    if FORCE_CPU:
        use_cpu_only()


def load_agent(weights_path: str, seed: int) -> DQNAgent:
    agent = DQNAgent(build_agent_spec(), seed_everything(seed))
    agent.load(weights_path)
    return agent


def command_train(arguments: argparse.Namespace) -> None:
    rng = seed_everything(arguments.seed)
    run_dir = run_directory(arguments.run_name or default_run_name(datetime.now()))
    components = TrainingComponents(
        agent=DQNAgent(build_agent_spec(), rng),
        memory=build_memory(arguments.seed),
        env_factory=make_environment,
        epsilon=build_epsilon_schedule(),
        rng=rng,
    )
    train_by_stage(components, build_training_spec(run_dir, arguments.seed))


def command_evaluate(arguments: argparse.Namespace) -> None:
    agent = load_agent(arguments.weights, arguments.seed)
    envs = [make_environment(path, MAX_STEPS_PER_EPISODE) for path in LEVEL_PATHS]
    summaries = evaluate_all_levels(agent, envs, arguments.episodes, EVAL_SEED)
    reporting.print_evaluation_table("Evaluación voraz de todos los niveles:", summaries)


def frame_printer(delay_seconds: float) -> FrameObserver:
    def show_frame(env: SnakeEnv) -> None:
        reporting.print_frame(render_board(env), delay_seconds)

    return show_frame


def command_play(arguments: argparse.Namespace) -> None:
    agent = load_agent(arguments.weights, arguments.seed)
    env = make_environment(LEVEL_PATHS[arguments.level - 1], MAX_STEPS_PER_EPISODE)
    result = watch_greedy_episode(env, agent, arguments.seed, frame_printer(arguments.delay))
    reporting.print_episode_result(result)


def add_train_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("train", help="entrena el agente con el curriculum de niveles")
    parser.add_argument("--run-name", default=None, help="nombre de la carpeta en runs/ (por defecto, fecha y hora)")
    parser.add_argument("--seed", type=int, default=SEED, help="semilla de la corrida")
    parser.set_defaults(handler=command_train)


def add_evaluate_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("evaluate", help="evalúa unos pesos guardados en todos los niveles")
    parser.add_argument("--weights", required=True, help="ruta a un archivo .weights.h5")
    parser.add_argument("--episodes", type=int, default=EVAL_EPISODES, help="episodios por nivel")
    parser.add_argument("--seed", type=int, default=SEED, help="semilla del agente")
    parser.set_defaults(handler=command_evaluate)


def add_play_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("play", help="juega una partida voraz y la dibuja en consola")
    parser.add_argument("--weights", required=True, help="ruta a un archivo .weights.h5")
    parser.add_argument("--level", type=int, required=True, choices=range(1, len(LEVEL_PATHS) + 1), help="nivel a jugar")
    parser.add_argument("--delay", type=float, default=PLAY_DELAY_SECONDS, help="segundos entre cuadros")
    parser.add_argument("--seed", type=int, default=SEED, help="semilla de la partida")
    parser.set_defaults(handler=command_play)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Agente DQN para Snake con curriculum learning.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    add_train_parser(subparsers)
    add_evaluate_parser(subparsers)
    add_play_parser(subparsers)
    return parser


def main(argv: list[str] | None = None) -> None:
    arguments = build_parser().parse_args(argv)
    configure_device()
    arguments.handler(arguments)


if __name__ == "__main__":
    main()
