from dataclasses import dataclass, field


def decay_epsilon(epsilon: float, decay: float, minimum: float) -> float:
    return max(minimum, epsilon * decay)


@dataclass
class EpsilonSchedule:
    start: float
    stage_start: float
    minimum: float
    decay: float
    value: float = field(init=False)

    def __post_init__(self) -> None:
        self.value = self.start

    def step(self) -> None:
        self.value = decay_epsilon(self.value, self.decay, self.minimum)

    def restart_for_stage(self) -> None:
        self.value = max(self.value, self.stage_start)
