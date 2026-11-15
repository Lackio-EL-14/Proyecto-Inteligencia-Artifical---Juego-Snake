from dataclasses import dataclass

UNREACHABLE_POTENTIAL = -1.0


@dataclass(frozen=True)
class RewardSpec:
    apple: float
    death: float
    step: float
    shaping_gamma: float


def path_potential(path_distance: int | None, free_cells: int) -> float:
    if path_distance is None:
        return UNREACHABLE_POTENTIAL
    return -path_distance / free_cells


def potential_shaping(previous_potential: float, current_potential: float, gamma: float) -> float:
    return gamma * current_potential - previous_potential


def move_reward(spec: RewardSpec, previous_potential: float, current_potential: float) -> float:
    return spec.step + potential_shaping(previous_potential, current_potential, spec.shaping_gamma)


def hunger_limit(grid_size: tuple[int, int], cells_factor: float | None) -> int | None:
    if cells_factor is None:
        return None
    return max(1, int(cells_factor * grid_size[0] * grid_size[1]))
