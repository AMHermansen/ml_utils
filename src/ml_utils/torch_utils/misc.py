# This file includes code adapted from:
#   https://github.com/mattcleigh/mltools
# Copyright (c) Matthew Leigh
# Licensed under the MIT License (See LICENSE file for details).

"""Miscellaneous pytorch utility functions."""

import torch as th

from ml_utils.torch_utils.types import CulensTensor


def append_dimensions(x: th.Tensor, target_dimensions: int, dim: int = -1) -> th.Tensor:
    """Appends dimensions of size 1 to a tensor until it reaches the target number of dimensions.

    Implementation based on https://github.com/mattcleigh/mltools/blob/master/mltools/torch_utils.py

    Args:
        x (th.Tensor): The input tensor.
        target_dimensions (int): The target number of dimensions.
        dim (int, optional): The dimension along which to append. Defaults to -1.

    Returns:
        th.Tensor: The reshaped tensor with the target number of dimensions.
    """
    if (dim_diff := target_dimensions - x.ndim) < 0:
        raise ValueError(
            f"Target dimensions {target_dimensions} must be greater than or equal to current dimensions {x.ndim}."
        )

    # Fast path for common cases
    if dim_diff == 0:
        return x
    if dim_diff == 1:
        return x.unsqueeze(dim)
    if dim == -1:
        return x[(...,) + (None,) * dim_diff]
    if dim == 0:
        return x[(None,) * dim_diff + (...,)]

    allow_range = [-x.dim() - 1, x.dim()]
    if not allow_range[0] <= dim <= allow_range[1]:
        raise ValueError(f"Dimension {dim} out of range {allow_range}.")

    if dim < 0:
        dim += x.dim() + 1
    return x.view(*x.shape[:dim], *dim_diff * (1,), *x.shape[dim:])


def is_increasing_sequence(cu_seqlens: CulensTensor) -> bool:
    """Check if cu_seqlens represent an increasing sequence."""
    return th.all(th.diff(cu_seqlens) > 0).item()
