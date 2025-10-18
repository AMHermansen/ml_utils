"""Utilities for converting between different tensor layouts used in attention
mechanisms.
"""
from einops import rearrange

from ml_utils.torch_utils.types import (
    PackedKVTensor,
    PackedQKVTensor,
    PackedTensor,
)


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
