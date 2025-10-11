"""Module for converting between packed and batched representations of data."""

import torch as th

from ml_utils.torch_utils.types import (
    CulensTensor,
    GeneralBatchedTensor,
    GeneralPackedTensor,
    MaskTensor,
)


def pack_tensors(
    mask: MaskTensor,
    *tensors: GeneralBatchedTensor,
) -> tuple[CulensTensor, tuple[GeneralPackedTensor, ...]]:
    """Pack tensors based on the provided mask.

    Args:
        mask: Boolean mask indicating valid entries. Shape (B, N).
        *tensors: Tensors to be packed. Each tensor should have shape (B, N, ..., F).

    Returns:
        cu_seqlens:
            Cumulative sequence lengths. Shape (B+1,).
        packed_tensors:
            Packed tensors with shape (total_valid_entries, ..., F).
    """
    packed_tensors = []
    for tensor in tensors:
        if len(tensor.shape) < 2:
            raise ValueError(
                f"Each tensor must be 2-dimensional, but got one-dimensional shape "
                f"{tensor.shape}."
            )
        batch_size, length, *_ = tensor.shape
        if mask.shape[0] != batch_size or mask.shape[1] != length:
            raise ValueError(
                f"Mask shape {mask.shape} must match tensor shape {tensor.shape} "
                "in the first two dimensions."
            )

        packed_tensor = tensor[mask]
        packed_tensors.append(packed_tensor)
    cu_seqlens = th.cumsum(
        th.cat(
            [th.zeros(1, device=mask.device, dtype=th.int32), mask.sum(dim=1)], dim=0
        ),
        dim=0,
    )
    return cu_seqlens, tuple(packed_tensors)


def unpack_tensors(
    cu_seqlens: CulensTensor,
    *tensors: GeneralPackedTensor,
    max_length: int | None = None,
) -> tuple[MaskTensor, tuple[GeneralBatchedTensor, ...]]:
    """Unpack tensors based on the provided cumulative sequence lengths.

    Args:
        cu_seqlens: Cumulative sequence lengths. Shape (B+1,).
        *tensors: Tensors to be unpacked. Each tensor should have
            shape (total_valid_entries, F).
        max_length: Maximum length to pad the sequences to. If None, uses the
            maximum sequence length in the batch.

    Returns:
        mask:
            Boolean mask indicating valid entries. Shape (B, N).
        batched_tensors:
            Unpacked tensors with shape (B, N, ..., F).

    """
    if max_length is None:
        max_length = th.diff(cu_seqlens).max().item()
    batch_size = len(cu_seqlens) - 1
    mask = th.arange(max_length, device=cu_seqlens.device).unsqueeze(0) < th.diff(
        cu_seqlens
    ).unsqueeze(1)
    batched_tensors = []
    for tensor in tensors:
        feature_dim = tensor.shape[1]
        batched_tensor = th.zeros(
            (batch_size, max_length, feature_dim),
            device=tensor.device,
            dtype=tensor.dtype,
        )
        batched_tensor[mask] = tensor
        batched_tensors.append(batched_tensor)
    return mask, tuple(batched_tensors)
