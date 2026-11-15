from dataclasses import dataclass

import numpy as np

from core.movements import RIGHT

STATE_SIZE = 11
FAKE_LEVEL_ID = 1


@dataclass(frozen=True)
class FakeTrainResult:
    loss: float


@dataclass(frozen=True)
class StepOutcome:
    reward: float
    terminated: bool
    truncated: bool
    reason: str


WIN_OUTCOME = StepOutcome(1.0, True, False, "win")
TIME_LIMIT_OUTCOME = StepOutcome(-0.1, False, True, "time_limit")


class FakeAgent:
    def __init__(self) -> None:
        self.saved_paths: list[str] = []
        self.sync_calls = 0
        self.train_calls = 0

    def act(self, state: np.ndarray, epsilon: float) -> int:
        return RIGHT

    def greedy_action(self, state: np.ndarray) -> int:
        return RIGHT

    def train_step(self, batch: object) -> FakeTrainResult:
        self.train_calls += 1
        return FakeTrainResult(loss=float(self.train_calls))

    def sync_target(self) -> None:
        self.sync_calls += 1

    def save(self, path: str) -> None:
        self.saved_paths.append(path)


class FakeMemory:
    def __init__(self, reported_size: int = 0) -> None:
        self.transitions: list[tuple] = []
        self.reported_size = reported_size

    def remember(self, *transition: object) -> None:
        self.transitions.append(transition)

    def __len__(self) -> int:
        return self.reported_size

    def sample_batch(self, batch_size: int) -> tuple[np.ndarray, ...]:
        states = np.zeros((batch_size, STATE_SIZE), dtype=np.float32)
        actions = np.zeros(batch_size, dtype=np.int32)
        rewards = np.zeros(batch_size, dtype=np.float32)
        return states, actions, rewards, states, np.zeros(batch_size, dtype=np.bool_)


class FakeEnvironment:
    def __init__(self, outcome: StepOutcome, steps_to_finish: int = 1) -> None:
        self.outcome = outcome
        self.steps_to_finish = steps_to_finish
        self.reset_seeds: list[int | None] = []
        self.steps = 0

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict]:
        self.reset_seeds.append(seed)
        self.steps = 0
        return np.zeros(STATE_SIZE, dtype=np.float32), {"level_id": FAKE_LEVEL_ID}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        self.steps += 1
        if self.steps < self.steps_to_finish:
            return np.ones(STATE_SIZE, dtype=np.float32), 0.0, False, False, {"reason": "moved", "apples_eaten": 0}
        info = {"reason": self.outcome.reason, "apples_eaten": int(self.outcome.reason == "win")}
        return np.ones(STATE_SIZE, dtype=np.float32), self.outcome.reward, self.outcome.terminated, self.outcome.truncated, info


@dataclass
class FakeEpsilon:
    value: float = 1.0
    step_calls: int = 0
    restart_calls: int = 0

    def step(self) -> None:
        self.step_calls += 1

    def restart_for_stage(self) -> None:
        self.restart_calls += 1
