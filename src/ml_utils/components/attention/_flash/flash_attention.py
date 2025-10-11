from collections.abc import Callable

import torch as th
from flash_attn.flash_attn_interface import (
    flash_attn_varlen_func,
    flash_attn_varlen_kvpacked_func,
    flash_attn_varlen_qkvpacked_func,
)

from ml_utils.components.attention.attention_config import FlashAttentionKWArgs
from ml_utils.torch_utils.types import (
    AllPackedQKVTypes,
    CulensTensor,
    PackedMHATensor,
)


def detect_qkv_structure(
    qkv: AllPackedQKVTypes,
) -> tuple[Callable[..., PackedMHATensor], bool]:
    """Utility function to detect how qkv is structured.

    Finds the correct flash attention implementation for the provided qkv structure.
    Also return a boolean if cross-attention is possible, (and thus different
    cu_seqlens) are required for q / kv.

    Args:
        qkv: The Query, Key, and Value tensors.
            They can either be grouped together as three separate tensors in a tuple,
            or kv can be merged into a single tensor, or all qkv can be merged into a
            single tensor.

            Non-merged tensors should have size: (packed_length nheads dimension).
            Merged tensors should have size: (packed_length n_merge nheads dimension).

    Returns:
        tuple: Where the first element is the proper flash attention implementation.
            The Second element is a boolean deciding if cross-attention is possible.

    """
    if isinstance(qkv, tuple) and len(qkv) == 3:
        # We have 3 separate tensors, one for q, k, and v.
        return flash_attn_varlen_func, True
    if isinstance(qkv, tuple) and len(qkv) == 2:
        # Only two separate tensors (i.e. cross attention with kv merged)
        return flash_attn_varlen_kvpacked_func, True
    if isinstance(qkv, th.Tensor):
        return flash_attn_varlen_qkvpacked_func, False
    raise ValueError(f"detect_qkv_structure received unexpected input: {type(qkv)}")


def get_qkv_dtype(qkv: AllPackedQKVTypes) -> th.dtype:
    """Utility function to get the dtype of the qkv tensors.

    Args:
        qkv: The Query, Key, and Value tensors.
            They can either be grouped together as three separate tensors in a tuple,
            or kv can be merged into a single tensor, or all qkv can be merged into a
            single tensor.

            Non-merged tensors should have size: (packed_length nheads dimension).
            Merged tensors should have size: (packed_length n_merge nheads dimension).

    Returns:
        The dtype of the qkv tensors.

    """
    if isinstance(qkv, tuple):
        return qkv[0].dtype
    if isinstance(qkv, th.Tensor):
        return qkv.dtype
    raise ValueError(f"get_qkv_dtype received unexpected input: {type(qkv)}")


def transform_qkv_to_bfloat16(
    qkv: AllPackedQKVTypes,
) -> AllPackedQKVTypes:
    """Utility function to convert qkv tensors to bfloat16.

    Args:
        qkv: The Query, Key, and Value tensors.
            They can either be grouped together as three separate tensors in a tuple,
            or kv can be merged into a single tensor, or all qkv can be merged into a
            single tensor.

            Non-merged tensors should have size: (packed_length nheads dimension).
            Merged tensors should have size: (packed_length n_merge nheads dimension).

    Returns:
        The qkv tensors converted to bfloat16.

    """
    if isinstance(qkv, tuple):
        return tuple(t.to(th.bfloat16) for t in qkv)
    if isinstance(qkv, th.Tensor):
        return qkv.to(th.bfloat16)
    raise ValueError(
        f"transform_qkv_to_bfloat16 received unexpected input: {type(qkv)}"
    )


def common_flash_attention_interface(
    qkv: AllPackedQKVTypes,
    *,
    cu_seqlens_q: CulensTensor,
    cu_seqlens_k: CulensTensor | None = None,
    max_seqlen_q: int | None = None,
    max_seqlen_k: int | None = None,
    flash_attn_kwargs: FlashAttentionKWArgs | None = None,
) -> PackedMHATensor:
    """Args:
        qkv: The Query, Key, and Value tensors.
            They can either be grouped together as three separate tensors in a tuple,
            or kv can be merged into a single tensor, or all qkv can be merged into a
            single tensor.

            Non-merged tensors should have size: (packed_length nheads dimension).
            Merged tensors should have size: (packed_length n_merge nheads dimension).
        cu_seqlens_q: Cumulative sequence lengths for the queries.
        cu_seqlens_k: Cumulative sequence lengths for the keys/values.
            If None, then cu_seqlens_q is used (i.e., self-attention).
        max_seqlen_q: Maximum sequence length for the queries. If None, it will be
            inferred from cu_seqlens_q.
        max_seqlen_k: Maximum sequence length for the keys/values. If None, it will be
            inferred from cu_seqlens_k (or cu_seqlens_q if cu_seqlens_k is None).
        flash_attn_kwargs: Keyword arguments for the flash attention implementation.

    Returns:
        partial application of the correct flash attention implementation, with
        the provided qkv and cu_seqlens_q / cu_seqlens_k.

    """
    flash_attn_kwargs = (
        flash_attn_kwargs if flash_attn_kwargs is not None else FlashAttentionKWArgs()
    )
    flash_attn_impl, cross_attention_possible = detect_qkv_structure(qkv)
    original_qkv_type = get_qkv_dtype(qkv)
    if original_qkv_type not in {th.float16, th.bfloat16}:
        qkv = transform_qkv_to_bfloat16(qkv)

    if max_seqlen_q is None:
        max_seqlen_q = int(th.max(th.diff(cu_seqlens_q)).item())
    if not cross_attention_possible:
        # We are in merged qkv mode, so attention function interface is different.
        return flash_attn_impl(
            qkv,
            cu_seqlens_q,
            max_seqlen_q,
            dropout_p=flash_attn_kwargs.dropout_p,
            softmax_scale=flash_attn_kwargs.softmax_scale,
            causal=flash_attn_kwargs.causal,
            window_size=flash_attn_kwargs.window_size,
            softcap=flash_attn_kwargs.softcap,
            alibi_slopes=None,  # Not supported in merged qkv mode.
            deterministic=flash_attn_kwargs.deterministic,
        ).to(original_qkv_type)

    if cu_seqlens_k is None:
        cu_seqlens_k = cu_seqlens_q
    if max_seqlen_k is None:
        max_seqlen_k = int(th.max(th.diff(cu_seqlens_k)).item())

    # Directly call flash_attn_impl with the provided arguments.
    return flash_attn_impl(
        *qkv,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p=flash_attn_kwargs.dropout_p,
        softmax_scale=flash_attn_kwargs.softmax_scale,
        causal=flash_attn_kwargs.causal,
        window_size=flash_attn_kwargs.window_size,
        softcap=flash_attn_kwargs.softcap,
        alibi_slopes=None,  # Not supported in merged qkv mode.
        deterministic=flash_attn_kwargs.deterministic,
    ).to(original_qkv_type)
