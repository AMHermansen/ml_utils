import numpy as np
from typing import cast

# Port of packing utilities from ml_utils.torch_utils.packing to numpy.


def is_increasing_sequence(sequence: np.ndarray) -> bool:
    """Check if a sequence is strictly increasing."""
    return cast("bool", np.all(np.diff(sequence) >= 0))


def pack_arrays(
    mask: np.ndarray,
    *arrays: np.ndarray,
) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    """Pack arrays based on the provided mask.

    Args:
        mask: Boolean mask indicating valid entries. Shape (B, N).
        *arrays: Arrays to be packed. Each array should have shape (B, N, ..., F).

    Returns:
        cu_seqlens:
            Cumulative sequence lengths. Shape (B+1,).
        packed_arrays:
            Packed arrays with shape (total_valid_entries, ..., F).

    Raises:
        ValueError: If the mask is not of boolean type.
        ValueError: If any array is not at least 2-dimensional.
    """
    if mask.dtype != bool:
        raise ValueError(f"Mask must be of boolean type, but got {mask.dtype}.")

    packed_arrays = []
    for array in arrays:
        if len(array.shape) < 2:
            raise ValueError(
                f"Each array must be at least 2-dimensional, but got shape "
                f"{array.shape}."
            )
        batch_size, length, *_ = array.shape
        if mask.shape[0] != batch_size or mask.shape[1] != length:
            raise ValueError(
                f"Mask shape {mask.shape} must match array shape {array.shape} "
                "in the first two dimensions."
            )

        packed_array = array[mask]
        packed_arrays.append(packed_array)

    cu_seqlens = np.cumsum(
        np.concatenate([np.zeros(1, dtype=np.int32), mask.sum(axis=1)]),
        axis=0,
    )
    # We really, really do not want cu_seqlens to not be increasing.
    # Leads to very hard to debug errors in flash attention.
    assert is_increasing_sequence(cu_seqlens), (
        "cu_seqlens must be an increasing sequence"
    )
    return cu_seqlens, tuple(packed_arrays)


def pack_array(
    mask: np.ndarray,
    array: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Pack a single array based on the provided mask.

    Provides a convenient wrapper around `pack_arrays` for a single array. To avoid
    unnecessary tuple packing/unpacking.

    Args:
        mask: Boolean mask indicating valid entries. Shape (B, N).
        array: Tensor to be packed. Should have shape (B, N, ..., F).

    Returns:
        cu_seqlens:
            Cumulative sequence lengths. Shape (B+1,).
        pakced_array:
            Packed array with shape (total_valid_entries, ..., F).
    """
    cu_seqlens, (pakced_array,) = pack_arrays(mask, array)
    return cu_seqlens, pakced_array


def unpack_arrays(
    cu_seqlens: np.ndarray,
    *arrays: np.ndarray,
    max_length: int | None = None,
) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    """Unpack arrays based on the provided cumulative sequence lengths.

    Args:
        cu_seqlens: Cumulative sequence lengths. Shape (B+1,).
        *arrays: Tensors to be unpacked. Each array should have
            shape (total_valid_entries, F).
        max_length: Maximum length to pad the sequences to. If None, uses the
            maximum sequence length in the batch.

    Returns:
        mask:
            Boolean mask indicating valid entries. Shape (B, N).
        batched_arrays:
            Unpacked arrays with shape (B, N, ..., F).

    """
    # We really, really do not want cu_seqlens to not be increasing.
    # Leads to very hard to debug errors in flash attention.
    assert is_increasing_sequence(cu_seqlens), (
        "cu_seqlens must be an increasing sequence"
    )

    if max_length is None:
        max_length = np.diff(cu_seqlens).max().item()

    batch_size = len(cu_seqlens) - 1
    seq_lengths = np.diff(cu_seqlens)

    # Create mask using broadcasting
    mask = np.arange(max_length).reshape(1, -1) < seq_lengths.reshape(-1, 1)

    batched_arrays = []
    for array in arrays:
        feature_like_dims = array.shape[1:]
        batched_array = np.zeros(
            (batch_size, max_length, *feature_like_dims),
            dtype=array.dtype,
        )
        batched_array[mask] = array
        batched_arrays.append(batched_array)

    return mask, tuple(batched_arrays)


def unpack_array(
    cu_seqlens: np.ndarray,
    array: np.ndarray,
    *,
    max_length: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Unpack a single array based on the provided cumulative sequence lengths.

    Provides a convenient wrapper around `unpack_arrays` for a single array. To avoid
    unnecessary tuple packing/unpacking.

    Args:
        cu_seqlens: Cumulative sequence lengths. Shape (B+1,).
        array: Tensor to be unpacked. Should have shape (total_valid_entries, F).
        max_length: Maximum length to pad the sequences to. If None, uses the
            maximum sequence length in the batch. Keyword only to avoid confusion with
            unpack_arrays signature.

    Returns:
        mask:
            Boolean mask indicating valid entries. Shape (B, N).
        batched_array:
            Unpacked array with shape (B, N, ..., F).
    """
    mask, (batched_array,) = unpack_arrays(cu_seqlens, array, max_length=max_length)
    return mask, batched_array
