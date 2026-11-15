from dataclasses import dataclass


@dataclass(frozen=True)
class AgentSpec:
    state_size: int
    action_size: int
    hidden_units: tuple[int, ...]
    learning_rate: float
    gradient_clipnorm: float
    gamma: float
    huber_delta: float
    target_rule: str
    use_xla: bool
