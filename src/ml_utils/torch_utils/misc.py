# This file includes code adapted from:
#   https://github.com/mattcleigh/mltools
# Copyright (c) Matthew Leigh
# Licensed under the MIT License (See LICENSE file for details).

"""Miscellaneous pytorch utility functions."""

from collections.abc import Callable
from typing import Any, cast

import numpy as np
import torch as th
from torch import nn

from ml_utils.torch_utils.types import CulensTensor
from ml_utils.utils import default


# Implementation based on https://github.com/mattcleigh/mltools/blob/master/mltools/torch_utils.py
class ParameterNoWeightDecay(nn.Parameter):
    """A parameter that is excluded from weight decay.

    This only works when used with optimizers from `ml_utils.torch_utils.optimizers`.
    """


# Implementation based on https://github.com/mattcleigh/mltools/blob/master/mltools/torch_utils.py
def append_dimensions(x: th.Tensor, target_dimensions: int, dim: int = -1) -> th.Tensor:
    """Appends dimensions of size 1 to a tensor until it reaches the target number of dimensions.

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
    return cast("bool", th.all(th.diff(cu_seqlens) >= 0).item())


def recurse_and_apply(
    collection: Any,
    func: Callable[[Any], Any],
    target_type: type | tuple[type, ...],
):
    """Recursively applies a function to all elements in a nested collection.

    Args:
        collection (Any): The input collection (can be a list, tuple, dict, or other).
        func (Callable[[Any], Any]): The function to apply to each element.
        target_type (type): The type of elements to which the function should be
            applied.

    Returns:
        Any: The collection with the function applied to each element.
    """
    if isinstance(collection, target_type):
        return func(collection)
    if isinstance(collection, list):
        return [recurse_and_apply(item, func, target_type) for item in collection]
    if isinstance(collection, tuple):
        return tuple(recurse_and_apply(item, func, target_type) for item in collection)
    if isinstance(collection, dict):
        return {
            key: recurse_and_apply(value, func, target_type)
            for key, value in collection.items()
        }
    return collection


def convert_to_torch(
    data_collection: Any,
    device: th.device | str = "cpu",
    dtype: th.dtype | None = None,
):
    """Converts all numpy arrays in a nested collection to PyTorch tensors.

    Args:
        data_collection: Any: A nested collection (list, tuple, dict) containing numpy arrays or PyTorch tensors.
        device: th.device | str: The device to move the tensors to.
        dtype: th.dtype | None: The desired data type of the tensors.
            If None the data type is converted to float32.

    Returns:

    """
    dtype = default(dtype, th.float32)

    def to_torch(x: np.ndarray | th.Tensor) -> th.Tensor:
        if isinstance(x, th.Tensor):
            x = x.to(device=device)
            return x.to(dtype=dtype)
        return th.tensor(x, device=device)

    return recurse_and_apply(data_collection, to_torch, (np.ndarray, th.Tensor))


def convert_to_numpy(data_collection: Any):
    """Converts all PyTorch tensors in a nested collection to numpy arrays.

    This function has to convert bfloat16 tensors to a different dtype, since bfloat16
    isn't support by numpy. The chosen dtype is float32, to capture the full range of
    bfloat16.
    """

    def to_numpy(x: th.Tensor) -> np.ndarray:
        if x.dtype == th.bfloat16:
            x = x.to(th.float32)
        return x.detach().cpu().numpy()

    return recurse_and_apply(data_collection, to_numpy, th.Tensor)
