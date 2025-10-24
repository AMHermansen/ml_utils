from typing import Literal

import jaxtyping as jt
import torch as th
from torch import nn

from ml_utils.torch_utils.misc import append_dimensions
from ml_utils.torch_utils.types import (
    CulensTensor,
    GeneralBatchedTensor,
    GeneralPackedTensor,
)


def get_drop_sample(
    keep_prob: float,
    batch_size: int,
    device: th.device,
    dtype: th.dtype,
):
    """Generates a mask signifying which samples to drop.

    Args:
        keep_prob: Probability of an element to be kept.
        batch_size: Size of the batch.
        device: device to create the mask on.
        dtype: dtype of the mask.

    Returns:
        A tensor of shape (batch_size,) with 1s for kept samples and 0s for dropped
        samples.
    """
    random_tensor = th.rand(batch_size, device=device, dtype=dtype)
    return (random_tensor < keep_prob).to(dtype)


def apply_batched_drop_path(
    output: GeneralBatchedTensor,
    drop_prob: float,
):
    """Applies DropPath to the output tensor.

    Randomly drops entire samples in the batch with probability `drop_prob`.

    Args:
        output: Output tensor of shape (batch_size, ...).
        drop_prob: Probability of an element to be zeroed.

    Returns:
        Tensor after applying DropPath.
    """
    keep_prob = 1 - drop_prob
    dropped = get_drop_sample(keep_prob, output.shape[0], output.device, output.dtype)
    dropped = append_dimensions(dropped, output.dim())
    return output.div(keep_prob) * dropped


# This function technically needs the lengths and not the cumulative lengths,
# but since most other functions / classes use cumulative lengths, we keep it
# consistent, at the cost of a (probably very minor) inefficiency.
# If one aims to optimize this for efficiency, one could store the lengths at beginning
# of forward pass in class, and then refactor this to take lengths directly.
def apply_packed_drop_path(
    output: GeneralPackedTensor,
    cu_seqlens: CulensTensor,
    drop_prob: float,
):
    """Applies DropPath to packed sequences.

    Args:
        output: Output tensor of shape (total_length, ...).
        cu_seqlens: Cumulative sequence lengths tensor. Shape (batch_size + 1,).
        drop_prob: Probability of a sequence to be zeroed.

    Returns:
        Tensor after applying DropPath.
    """
    keep_prob = 1 - drop_prob
    dropped = get_drop_sample(
        keep_prob, cu_seqlens.shape[0] - 1, output.device, output.dtype
    )
    # repeating according to sequence lengths
    dropped = th.repeat_interleave(dropped, th.diff(cu_seqlens), dim=0)
    dropped = append_dimensions(dropped, output.dim())
    return output.div(keep_prob) * dropped


def maybe_apply_dropout(
    x: jt.Float[th.Tensor, " *dimensions"],
    dropout_rate: float,
    training: bool,
) -> jt.Float[th.Tensor, " *dimensions"]:
    """Applies dropout to the input tensor if dropout_rate > 0 and in training mode.

    Args:
        x: Input tensor.
        dropout_rate: Dropout probability.
        training: Whether the model is in training mode.

    Returns:
        Tensor after applying dropout.
    """
    if dropout_rate > 0.0 and training:
        return nn.functional.dropout(x, p=dropout_rate)
    return x


def maybe_apply_stochastic_depth(
    x: jt.Float[th.Tensor, " *dimensions"],
    cu_seqlens: CulensTensor | None,
    drop_path_rate: float,
    training: bool,
    input_format: Literal["packed", "unpacked"],
):
    """Applies stochastic depth (DropPath) to the input tensor.

    Args:
        x: Input tensor.
        cu_seqlens: Cumulative sequence lengths tensor for packed input format.
        drop_path_rate: Probability of an element to be zeroed.
        training: Whether the model is in training mode.
        input_format: Format of the input tensor, either "packed" or "unpacked".

    Returns:
        Tensor after applying stochastic depth.
    """
    if drop_path_rate > 0.0 and training:
        if input_format == "packed" and cu_seqlens is not None:
            return apply_packed_drop_path(x, cu_seqlens, drop_path_rate)
        if input_format == "packed" and cu_seqlens is None:
            raise ValueError(
                "cu_seqlens must be provided for packed input format."
            )
        return apply_batched_drop_path(x, drop_path_rate)
    return x
