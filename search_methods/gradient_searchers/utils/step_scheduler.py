from abc import ABC, abstractmethod


class StepSchedulerBase(ABC):
    def __init__(self, init_n_steps: int, init_n_tokens_to_optimize: int, **kwargs):
        self.init_n_steps = init_n_steps
        self.init_n_optimize_tokens = init_n_tokens_to_optimize

    @abstractmethod
    def get_n_steps(self, n_tokens_optimized: int) -> int:
        ...


class IdentityStepScheduler(StepSchedulerBase):
    def get_n_steps(self, n_tokens_optimized: int) -> int:
        return self.init_n_steps


class ExponentialStepScheduler(StepSchedulerBase):
    def __init__(self, init_n_steps: int, init_n_tokens_to_optimize: int, min_n_steps: int = 5):
        super().__init__(init_n_steps, init_n_tokens_to_optimize)
        self.min_n_steps = min_n_steps

    def get_n_steps(self, n_tokens_to_optimize: int) -> int:
        n_tokens_optimized = self.init_n_optimize_tokens - n_tokens_to_optimize
        return max(self.init_n_steps // (2 ** n_tokens_optimized), self.min_n_steps)
