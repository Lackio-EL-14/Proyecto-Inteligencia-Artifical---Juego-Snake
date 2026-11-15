from agent.batch import batch_from_sample
from agent.dqn_model import DQNAgent, TrainResult
from core.experience import ExperienceReplay
from training.training_spec import LearnerSpec


def should_train(env_steps: int, memory_size: int, spec: LearnerSpec) -> bool:
    has_enough_memory = memory_size >= max(spec.warmup_steps, spec.batch_size)
    return has_enough_memory and env_steps % spec.train_every == 0


def should_sync_target(gradient_steps: int, every: int) -> bool:
    return gradient_steps % every == 0


class Learner:
    def __init__(self, agent: DQNAgent, memory: ExperienceReplay, spec: LearnerSpec) -> None:
        self.agent = agent
        self.memory = memory
        self.spec = spec
        self.env_steps = 0
        self.gradient_steps = 0

    def after_env_step(self) -> TrainResult | None:
        self.env_steps += 1
        if not should_train(self.env_steps, len(self.memory), self.spec):
            return None
        return self._train_once()

    def _train_once(self) -> TrainResult:
        batch = batch_from_sample(self.memory.sample_batch(self.spec.batch_size))
        result = self.agent.train_step(batch)
        self.gradient_steps += 1
        if should_sync_target(self.gradient_steps, self.spec.target_update_every):
            self.agent.sync_target()
        return result
