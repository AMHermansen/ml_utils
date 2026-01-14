"""Utilities for converting between different tensor layouts used in attention
mechanisms.
"""

from collections.abc import Callable
from typing import TypeAlias

import torch as th
from einops import rearrange

from ml_utils.torch_utils.types import (
    BatchedHeadedMatrixTensor,
    CulensTensor,
    PackedKVTensor,
    PackedQKVTensor,
    PackedTensor,
)

ScoreModSignature: TypeAlias = Callable[
    [th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor], th.Tensor
]
MaskFuncSignature: TypeAlias = Callable[
    [th.Tensor, th.Tensor, th.Tensor, th.Tensor], th.Tensor
]


def convert_to_headed_and_qkvmerged_layout(
    tensor: PackedTensor,
    nheads: int,
) -> PackedQKVTensor:
    """Convert tensor to headed-packed QKV layout."""
    return rearrange(
        tensor,
        pattern="tot_len (n_merge nheads dim) -> tot_len n_merge nheads dim",
        nheads=nheads,
        n_merge=3,  # for QKV
    )


def convert_to_headed_and_kvmerged_layout(
    tensor: PackedTensor,
    nheads: int,
) -> PackedKVTensor:
    """Convert tensor to headed-packed KV layout."""
    return rearrange(
        tensor,
        pattern="tot_len (n_merge nheads dim) -> tot_len n_merge nheads dim",
        nheads=nheads,
        n_merge=2,  # for KV
    )


def convert_to_headed_layout(
    tensor: PackedTensor,
    nheads: int,
) -> PackedTensor:
    """Convert tensor to headed layout."""
    return rearrange(
        tensor,
        pattern="tot_len (nheads dim) -> tot_len nheads dim",
        nheads=nheads,
    )


def convert_from_headed_layout(
    tensor: PackedTensor,
    nheads: int,
) -> PackedTensor:
    """Convert tensor from headed layout to standard layout."""
    return rearrange(
        tensor,
        pattern="tot_len nheads dim -> tot_len (nheads dim)",
        nheads=nheads,
    )


# flex_attention utilities


def create_score_mod(bias_proj: BatchedHeadedMatrixTensor) -> ScoreModSignature:
    def score_mod(
        score: th.Tensor,
        batch: th.Tensor,
        head: th.Tensor,
        q_idx: th.Tensor,
        k_idx: th.Tensor,
    ) -> th.Tensor:
        return score + bias_proj[batch, q_idx, k_idx, head, ]

    return score_mod


def build_seq_idx(offsets: CulensTensor) -> th.Tensor:
    """Build sequence index tensor from offsets.

    Args:
        offsets: Offsets tensor of shape (batch_size + 1,).

    Returns:
        Sequence index tensor of shape (total_seq_len,).
    """
    return th.repeat_interleave(
        th.arange(len(offsets) - 1, device=offsets.device), th.diff(offsets)
    )
