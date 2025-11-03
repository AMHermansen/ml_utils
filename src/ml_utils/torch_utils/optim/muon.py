# This file includes code adapted from:
#   https://github.com/KellerJordan/Muon
# Copyright (c) 2024 Keller Jordan
# Licensed under the MIT License (See LICENSE file for details).

"""Implementation of the Muon optimizer.

We're not using the pytorch implementation for two reasons:
  1) Muon is implemented in pytorch 2.9+ only, and we want to support earlier versions.
  2) Pytorch's philosophy is to manually delegate different parameter groups to
  different optimizers, and since Muon is only recommended for some parameters, this
  would lead to a more complex code structure. Especially in lightning, where automatic
  optimization is only possible with a single optimizer.

This file is based on the official Muon implementation:
https://github.com/KellerJordan/Muon

This file also includes code based on:
https://github.com/mattcleigh/mltools
"""
from collections.abc import Callable

import torch as th
from torch import nn
from typing_extensions import override

from ml_utils.torch_utils.misc import ParameterNoWeightDecay


# Function adapted from: https://github.com/KellerJordan/Muon
@th.compile
def zeropower_via_newtonschulz5(G, steps: int):
    """Newton-Schulz iteration to compute the zeroth power / orthogonalization of G.
    We opt to use a quintic iteration whose coefficients are selected to maximize the
    slope at zero. For the purpose of minimizing steps, it turns out to be empirically
    effective to keep increasing the slope at zero even beyond the point where the
    iteration no longer converges all the way to one everywhere on the interval. This
    iteration therefore does not produce UV^T but rather something like US'V^T where
    S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert (
        G.ndim >= 2
    )  # batched Muon implementation by @scottjmaddox, and put into practice in the record by @YouJiacheng
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.mT
        B = (
            b * A + c * A @ A
        )  # quintic computation strategy adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


# Function adapted from: https://github.com/KellerJordan/Muon
def muon_update(grad, momentum, beta=0.95, ns_steps=5, nesterov=True):
    """Muon update computation."""
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta) if nesterov else momentum
    if update.ndim == 4:  # for the case of conv filters
        update = update.view(len(update), -1)
    update = zeropower_via_newtonschulz5(update, steps=ns_steps)
    update *= max(1, grad.size(-2) / grad.size(-1)) ** 0.5
    return update


# Function adapted from: https://github.com/KellerJordan/Muon
def adam_update(grad, buf1, buf2, step, betas, eps):
    buf1.lerp_(grad, 1 - betas[0])
    buf2.lerp_(grad.square(), 1 - betas[1])
    buf1c = buf1 / (1 - betas[0] ** step)
    buf2c = buf2 / (1 - betas[1] ** step)
    return buf1c / (buf2c.sqrt() + eps)


# Heuristic function to make using Muon easier.
# It might be a good idea to wrap everything except input/output layers into a single
# Module and then this function can be used to decide whether to use Muon or not for
# those. While the remaining should not use Muon no matter what.
# I will have to experiment with this, to figure out how to best use Muon in practice.
def suitable_for_muon(param: nn.Parameter) -> bool:
    """A heuristic to determine whether a parameter is suitable for Muon optimization.

    Muon is not recommended for the embedding/first layer or output layers, as well.
    This cannot be determined here.

    Args:
        param: The parameter to check.

    Returns:
        bool: Whether the parameter is suitable for Muon optimization.
    """
    if param.dim() < 2:
        return False
    return not isinstance(param, ParameterNoWeightDecay)


# Class adapted from: https://github.com/KellerJordan/Muon
class SingleDeviceMuonWithAuxAdam(th.optim.Optimizer):
    """Non-distributed variant of MuonWithAuxAdam."""

    def __init__(self, param_groups):
        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                # defaults
                group["lr"] = group.get("lr", 0.02)
                group["momentum"] = group.get("momentum", 0.95)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == {
                    "params",
                    "lr",
                    "momentum",
                    "weight_decay",
                    "use_muon",
                }
            else:
                # defaults
                group["lr"] = group.get("lr", 3e-4)
                group["betas"] = group.get("betas", (0.9, 0.95))
                group["eps"] = group.get("eps", 1e-10)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == {
                    "params",
                    "lr",
                    "betas",
                    "eps",
                    "weight_decay",
                    "use_muon",
                }
        super().__init__(param_groups, {})

    @th.no_grad()
    @override
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        loss = None
        if closure is not None:
            with th.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                for p in group["params"]:
                    if p.grad is None:
                        # continue
                        p.grad = th.zeros_like(p)  # Force synchronization
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = th.zeros_like(p)
                    update = muon_update(
                        p.grad, state["momentum_buffer"], beta=group["momentum"]
                    )
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update.reshape(p.shape), alpha=-group["lr"])
            else:
                for p in group["params"]:
                    if p.grad is None:
                        # continue
                        p.grad = th.zeros_like(p)  # Force synchronization
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = th.zeros_like(p)
                        state["exp_avg_sq"] = th.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1
                    update = adam_update(
                        p.grad,
                        state["exp_avg"],
                        state["exp_avg_sq"],
                        state["step"],
                        group["betas"],
                        group["eps"],
                    )
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])

        return loss


Muon = SingleDeviceMuonWithAuxAdam  # No need for distributed versions.
