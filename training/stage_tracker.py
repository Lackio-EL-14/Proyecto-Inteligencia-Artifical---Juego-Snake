from collections import deque
from collections.abc import Sequence

from training.episode_runner import EpisodeResult, is_win


def window_win_rate(history: Sequence[int]) -> float:
    if len(history) == 0:
        return 0.0
    return sum(history) / len(history)


def should_report_progress(episode: int, every: int) -> bool:
    return episode % every == 0


class StageTracker:
    def __init__(self, window_size: int) -> None:
        self._window: deque[int] = deque(maxlen=window_size)
        self._rewards: list[float] = []
        self._win_rates: list[float] = []

    def record(self, result: EpisodeResult) -> None:
        self._window.append(int(is_win(result)))
        self._rewards.append(result.total_reward)
        self._win_rates.append(self.train_win_rate())

    def train_win_rate(self) -> float:
        return window_win_rate(self._window)

    def rewards(self) -> list[float]:
        return list(self._rewards)

    def win_rates(self) -> list[float]:
        return list(self._win_rates)
