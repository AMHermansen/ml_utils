import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


class LinearWarmupCosineDecay(LambdaLR):
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int = 1000,
        total_steps: int = 10000,
        final_factor: float = 5e-2,
        initial_factor: float = 1e-2,
    ) -> None:
        """Linear warmup and cosine decay learning rate scheduler.

        Args:
            optimizer: Optimizer to be scheduled.
            warmup_steps: Number of steps for linear warmup.
            total_steps: Total number of steps for cosine decay.
            final_factor: Final learning rate factor after decay.
            initial_factor: Initial learning rate factor during warmup.
        """

        def lr_lambda(steps: int) -> float:
            if steps <= warmup_steps:
                return initial_factor + steps * (1 - initial_factor) / warmup_steps
            if steps >= total_steps:
                return final_factor
            t = (steps - warmup_steps) / (total_steps - warmup_steps) * math.pi
            return (1 + math.cos(t)) * (1 - final_factor) / 2 + final_factor

        super().__init__(optimizer, lr_lambda)
