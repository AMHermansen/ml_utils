"""Module for converting between packed and batched representations of data."""

from typing import cast

import jaxtyping as jt
import torch as th

from ml_utils.torch_utils.misc import is_increasing_sequence
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

    Raises:
        ValueError: If the mask is not of boolean type.
        ValueError: If any tensor is not at least 2-dimensional.
    """
    if mask.dtype != th.bool:
        raise ValueError(f"Mask must be of boolean type, but got {mask.dtype}.")
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
            [
                th.zeros(1, device=mask.device, dtype=th.int32),
                mask.sum(dim=1),
            ],
            dim=0,
        ),
        dim=0,
    )
    # We really, really do not want cu_seqlens to not be increasing.
    # Leads to very hard to debug errors in flash attention.
    assert is_increasing_sequence(cu_seqlens), (
        "cu_seqlens must be an increasing sequence"
    )
    return cu_seqlens, tuple(packed_tensors)


def pack_tensor(
    mask: MaskTensor,
    tensor: GeneralBatchedTensor,
) -> tuple[CulensTensor, GeneralPackedTensor]:
    """Pack a single tensor based on the provided mask.

    Provides a convenient wrapper around `pack_tensors` for a single tensor. To avoid
    unnecessary tuple packing/unpacking.

    Args:
        mask: Boolean mask indicating valid entries. Shape (B, N).
        tensor: Tensor to be packed. Should have shape (B, N, ..., F).

    Returns:
        cu_seqlens:
            Cumulative sequence lengths. Shape (B+1,).
        packed_tensor:
            Packed tensor with shape (total_valid_entries, ..., F).
    """
    cu_seqlens, (packed_tensor,) = pack_tensors(mask, tensor)
    return cu_seqlens, packed_tensor


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
    # We really, really do not want cu_seqlens to not be increasing.
    # Leads to very hard to debug errors in flash attention.
    assert is_increasing_sequence(cu_seqlens), (
        "cu_seqlens must be an increasing sequence"
    )
    if max_length is None:
        # Pyright gets confused about Number and int here.
        max_length = cast("int", th.diff(cu_seqlens).max().item())
        assert isinstance(max_length, int)
    batch_size = len(cu_seqlens) - 1
    mask = th.arange(max_length, device=cu_seqlens.device).unsqueeze(0) < th.diff(
        cu_seqlens
    ).unsqueeze(1)
    batched_tensors = []
    for tensor in tensors:
        feature_like_dims = tensor.shape[1:]
        batched_tensor = th.zeros(
            (batch_size, max_length, *feature_like_dims),
            device=tensor.device,
            dtype=tensor.dtype,
        )
        batched_tensor[mask] = tensor
        batched_tensors.append(batched_tensor)
    return mask, tuple(batched_tensors)


def unpack_tensor(
    cu_seqlens: CulensTensor,
    tensor: GeneralPackedTensor,
    *,
    max_length: int | None = None,
) -> tuple[MaskTensor, GeneralBatchedTensor]:
    """Unpack a single tensor based on the provided cumulative sequence lengths.

    Provides a convenient wrapper around `unpack_tensors` for a single tensor. To avoid
    unnecessary tuple packing/unpacking.

    Args:
        cu_seqlens: Cumulative sequence lengths. Shape (B+1,).
        tensor: Tensor to be unpacked. Should have shape (total_valid_entries, F).
        max_length: Maximum length to pad the sequences to. If None, uses the
            maximum sequence length in the batch. Keyword only to avoid confusion with
            unpack_tensors signature.

    Returns:
        mask:
            Boolean mask indicating valid entries. Shape (B, N).
        batched_tensor:
            Unpacked tensor with shape (B, N, ..., F).
    """
    mask, (batched_tensor,) = unpack_tensors(
        cu_seqlens,
        tensor,
        max_length=max_length
    )
    return mask, batched_tensor


def prepend_tokens_to_packed_tensor(
    packed_tensor: jt.Float[th.Tensor, " total_len *feature_dims"],
    cu_seqlens: CulensTensor,
    tokens: jt.Float[th.Tensor, " n_tokens *feature_dims"] | th.nn.Parameter,
) -> tuple[GeneralPackedTensor, CulensTensor]:
    """Prepends tokens to a packed sequence.

    Args:
        packed_tensor: The input packed sequence. Shape (total_len, *feature_dims).
        cu_seqlens: The cumulative sequence lengths. Shape (batch_size + 1,).
        tokens: The tokens to prepend. Shape (n_tokens, *feature_dims).

    Returns:
        tuple[PackedTensor, CulensTensor]: The new packed sequence and updated cumulative
            sequence lengths.
    """
    assert is_increasing_sequence(cu_seqlens), (
        "cu_seqlens must be an increasing sequence"
    )
    mask, (unpacked_tensors,) = unpack_tensors(cu_seqlens, packed_tensor)
    batch_size = len(cu_seqlens) - 1
    unpacked_shape = unpacked_tensors.shape
    tokens_expanded = tokens.unsqueeze(0).repeat(
        batch_size, 1, *[1] * (len(unpacked_shape) - 2)
    )
    new_unpacked = th.cat([tokens_expanded, unpacked_tensors], dim=1)
    mask_new = th.cat(
        [
            th.ones((batch_size, tokens.shape[0]), dtype=th.bool, device=mask.device),
            mask,
        ],
        dim=1,
    )
    new_cu_seqlens, (new_packed_tensor,) = pack_tensors(mask_new, new_unpacked)
    assert is_increasing_sequence(new_cu_seqlens), (
        "cu_seqlens must be an increasing sequence"
    )
    return new_packed_tensor, new_cu_seqlens


def remove_tokens_from_packed_tensor(
    packed_tensor: jt.Float[th.Tensor, " total_len *feature_dims"],
    cu_seqlens: CulensTensor,
    n_tokens: int,
) -> tuple[GeneralPackedTensor, CulensTensor]:
    """Removes tokens from the start of a packed sequence.

    Args:
        packed_tensor: The input packed sequence. Shape (total_len, *feature_dims).
        cu_seqlens: The cumulative sequence lengths. Shape (batch_size + 1,).
        n_tokens: The number of tokens to remove from the start of each sequence.

    Returns:
        tuple[PackedTensor, CulensTensor]: The new packed sequence and updated cumulative sequence lengths
    """
    assert is_increasing_sequence(cu_seqlens), (
        "cu_seqlens must be an increasing sequence"
    )
    mask, (unpacked_tensors,) = unpack_tensors(cu_seqlens, packed_tensor)
    unpacked_shape = unpacked_tensors.shape
    if n_tokens >= unpacked_shape[1]:
        raise ValueError(
            f"Cannot remove {n_tokens} tokens from sequences of length "
            f"{unpacked_shape[1]}."
        )
    new_unpacked = unpacked_tensors[:, n_tokens:, ...]
    mask_new = mask[:, n_tokens:]
    new_cu_seqlens, (new_packed_tensor,) = pack_tensors(mask_new, new_unpacked)
    assert is_increasing_sequence(new_cu_seqlens), (
        "cu_seqlens must be an increasing sequence"
    )
    return new_packed_tensor, new_cu_seqlens


def get_masked_cu_seqlens(
    cu_seqlens: CulensTensor,
    mask: jt.Bool[th.Tensor, " total_seqlen"]
) -> CulensTensor:
    """Get cumulative sequence lengths after applying a mask.

    Args:
        cu_seqlens: The original cumulative sequence lengths. Shape (B+1,).
        mask: Boolean mask indicating valid entries. Shape (total_seqlen).

    Returns:
        CulensTensor: The new cumulative sequence lengths after applying the mask.
    """
    assert is_increasing_sequence(cu_seqlens), (
        "cu_seqlens must be an increasing sequence"
    )
    cumulative_number_of_masked_elements = th.cat(
        [
            th.zeros(1, device=mask.device, dtype=th.int32),
            th.cumsum(
                mask.to(dtype=th.int32),
                dim=0,
            )
        ]
    )
    return cumulative_number_of_masked_elements[cu_seqlens].to(dtype=th.int32)


def get_packed_mean_loss(
    packed_losses: jt.Float[th.Tensor, " total_len"],
    cu_seqlens: CulensTensor,
) -> th.Tensor:
    """Compute the combined mean loss from packed losses.
    
    Args:
        packed_losses: The packed losses for each valid entry. Shape (total_len,).
        cu_seqlens: The cumulative sequence lengths. Shape (B+1,).
    
    Returns:
        th.Tensor: The combined mean loss as a scalar tensor.
    """
    lengths = th.diff(cu_seqlens)
    lengths_expanded = th.repeat_interleave(lengths, lengths)
    weighted_losses = th.einsum("n...,n->n...", packed_losses, 1.0 / lengths_expanded)
    return th.sum(weighted_losses) / len(lengths)
