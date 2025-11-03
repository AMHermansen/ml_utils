# This file includes code adapted from:
#   https://github.com/mattcleigh/mltools
# Copyright (c) Matthew Leigh
# Licensed under the MIT License (See LICENSE file for details).

"""Custom pytorch learning rate schedulers."""

import math
import warnings

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, LRScheduler, OneCycleLR
from typing_extensions import override


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


# This class is adapted from: github.com/mattcleigh/mltools
class LinearWarmupRootDecay(LRScheduler):
    """Learning rate scheduler with a linear wamup and a sqrt decay."""

    def __init__(
        self,
        optimizer: Optimizer,
        dim_model: int = 256,
        warmup_steps: int = 10000,
        last_epoch: int = -1,
        use_max_lr: bool = False,
    ) -> None:
        # For calculating the learning rate profile
        self.dim_model = dim_model
        self.warmup_steps = warmup_steps
        self.num_param_groups = len(optimizer.param_groups)

        # For overwritting the max learning rate (instead of using dim model)
        self.use_max_lr = use_max_lr
        self.max_lr_coef = (self.dim_model * self.warmup_steps) ** (0.5)

        # Super init at end for some reason
        super().__init__(optimizer, last_epoch)

    @override
    def get_lr(self) -> list[float]:
        lr = self.dim_model ** (-0.5) * min(
            self._step_count ** (-0.5), self._step_count * self.warmup_steps ** (-1.5)
        )
        if self.use_max_lr:
            lr *= self.base_lrs[0] * self.max_lr_coef
        return [lr] * self.num_param_groups


# This class is adapted from: github.com/mattcleigh/mltools
class WarmupToConstant(LRScheduler):
    """Gradually warm-up learning rate in optimizer to a constant value."""

    def __init__(self, optimizer: Optimizer, num_steps: int = 100, **_kwargs):
        """args:
        optimizer (Optimizer): Wrapped optimizer.
        num_steps: target learning rate is reached at num_steps.
        """
        self.num_steps = num_steps
        self.finished = False
        super().__init__(optimizer)

    @override
    def get_lr(self):
        if self.last_epoch > self.num_steps:
            return list(self.base_lrs)
        return [
            (base_lr / self.num_steps) * self.last_epoch for base_lr in self.base_lrs
        ]


# This class is adapted from: github.com/mattcleigh/mltools
class CyclicWithWarmup(OneCycleLR):
    """A cyclic scheduler with dedicated warmup periods based on the onecycle LR.

    The only difference is the get_lr method, which resets the scheduler after each
    cycle instead of throwing out an error
    """

    @override
    def get_lr(self):
        """Overloaded method for aquiring new learning rates.

        Only line that is changed from the original method is the step number!
        Also removed the warning that step > length
        """
        if not self._get_lr_called_within_step:  # pylint: disable=no-member
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.",
                UserWarning,
                stacklevel=2,
            )

        lrs = []
        step_num = self.last_epoch % self.total_steps  # Only changed line!!!

        for group in self.optimizer.param_groups:
            try:
                start_step = 0
                for i, phase in enumerate(self._schedule_phases):
                    end_step = phase["end_step"]
                    if step_num <= end_step or i == len(self._schedule_phases) - 1:
                        pct = (step_num - start_step) / (end_step - start_step)
                        # Seems like anneal_func is not the modern way to do this.
                        # TODO(Andreas): Figure out how to do this the modern way.
                        computed_lr = self.anneal_func(  # type: ignore[attr-defined]
                            group[phase["start_lr"]], group[phase["end_lr"]], pct
                        )
                        if self.cycle_momentum:
                            computed_momentum = self.anneal_func(  # type: ignore[attr-defined]
                                group[phase["start_momentum"]],
                                group[phase["end_momentum"]],
                                pct,
                            )
                        break
                    start_step = phase["end_step"]
            except Exception:
                computed_lr = lrs[-1]

            lrs.append(computed_lr)
            if self.cycle_momentum:
                if self.use_beta1:
                    _, beta2 = group["betas"]
                    group["betas"] = (computed_momentum, beta2)
                else:
                    group["momentum"] = computed_momentum

        return lrs
