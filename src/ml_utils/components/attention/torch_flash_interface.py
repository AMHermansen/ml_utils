from functools import partial

import torch as th
from torch.nn.functional import scaled_dot_product_attention

from ml_utils.components.attention.attention_config import FlashAttentionKWArgs
from ml_utils.torch_utils import (
    pack_tensors,
    unpack_tensors,
)
from ml_utils.torch_utils.types import (
    AllPackedQKVTypes,
    CulensTensor,
    PackedMHATensor,
)

from .attention_mask import combine_query_and_key_mask


# Currently bug in pytorch see https://github.com/pytorch/pytorch/issues/149608
def torch_flash_attention_interface(
    qkv: AllPackedQKVTypes,
    *,
    cu_seqlens_q: CulensTensor,
    cu_seqlens_k: CulensTensor | None = None,
    max_seqlen_q: int | None = None,
    max_seqlen_k: int | None = None,
    flash_attn_kwargs: FlashAttentionKWArgs | None = None,
) -> PackedMHATensor:
    """A wrapper around PyTorch's scaled_dot_product_attention to mimic flash attention.

    This function takes packed query, key, and value tensors along with their
    cumulative sequence lengths and applies scaled dot-product attention.
    This is significantly less efficient than the true flash attention implementation.

    This function only exists to provide a fallback, since flash_attention requires
    at least Ampere GPUs.

    Args:
        qkv: A tuple of (q, k, v) or (q, kv) or a single tensor containing q, k, v
            concatenated along the last dimension.
            Shape is gnerally (total_seqlen, (merged_tensor), num_heads, head_dim).
        cu_seqlens_q: Cumulative sequence lengths for queries.
        cu_seqlens_k: Cumulative sequence lengths for keys. If None, assumed to be the
            same as cu_seqlens_q.
        max_seqlen_q: Maximum sequence length for queries. If None, inferred from
            cu_seqlens_q.
        max_seqlen_k: Maximum sequence length for keys. If None, inferred from
            cu_seqlens_k.
        flash_attn_kwargs: Additional keyword arguments to pass to
            scaled_dot_product_attention.

    Returns:
        The output tensor after applying scaled dot-product attention.
    """
    del max_seqlen_q, max_seqlen_k  # Unused, only needed to match flash attention API.
    swap_length_and_head_dim = partial(th.permute, dims=(0, 2, 1, 3))
    if cu_seqlens_k is None:
        cu_seqlens_k = cu_seqlens_q

    flash_attn_kwargs = (
        flash_attn_kwargs if flash_attn_kwargs is not None else FlashAttentionKWArgs()
    )

    if isinstance(qkv, tuple):
        if len(qkv) == 3:
            q, k, v = qkv
        elif len(qkv) == 2:
            q, kv = qkv
            k, v = th.chunk(kv, 2, dim=1)
        else:
            raise ValueError("qkv must be a tuple of (q, k, v) or (q, kv)")
    else:
        qkv = th.chunk(qkv, 3, dim=1)
        q, k, v = qkv
    q = q.squeeze(1)
    k = k.squeeze(1)
    v = v.squeeze(1)

    q_mask, (q_unpacked,) = unpack_tensors(cu_seqlens_q, q)
    kv_mask, (k_unpacked, v_unpacked) = unpack_tensors(cu_seqlens_k, k, v)

    attention_mask = combine_query_and_key_mask(q_mask, kv_mask)

    unpacked_out = scaled_dot_product_attention(
        swap_length_and_head_dim(q_unpacked),
        swap_length_and_head_dim(k_unpacked),
        swap_length_and_head_dim(v_unpacked),
        attn_mask=attention_mask,
        dropout_p=flash_attn_kwargs.dropout_p,
        is_causal=flash_attn_kwargs.causal,
        scale=flash_attn_kwargs.softmax_scale,
    )
    _, (packed_out,) = pack_tensors(kv_mask, swap_length_and_head_dim(unpacked_out))
    return packed_out


# Slow version that works around the bug in pytorch.
def torch_flash_attention_interface(
    qkv: AllPackedQKVTypes,
    *,
    cu_seqlens_q: CulensTensor,
    cu_seqlens_k: CulensTensor | None = None,
    max_seqlen_q: int | None = None,
    max_seqlen_k: int | None = None,
    flash_attn_kwargs: FlashAttentionKWArgs | None = None,
) -> PackedMHATensor:
    """A wrapper around PyTorch's scaled_dot_product_attention to mimic flash attention.

    This function takes packed query, key, and value tensors along with their
    cumulative sequence lengths and applies scaled dot-product attention.
    This is significantly less efficient than the true flash attention implementation.

    This function only exists to provide a fallback, since flash_attention requires
    at least Ampere GPUs.

    Args:
        qkv: A tuple of (q, k, v) or (q, kv) or a single tensor containing q, k, v
            concatenated along the last dimension.
            Shape is gnerally (total_seqlen, (merged_tensor), num_heads, head_dim).
        cu_seqlens_q: Cumulative sequence lengths for queries.
        cu_seqlens_k: Cumulative sequence lengths for keys. If None, assumed to be the
            same as cu_seqlens_q.
        max_seqlen_q: Maximum sequence length for queries. If None, inferred from
            cu_seqlens_q.
        max_seqlen_k: Maximum sequence length for keys. If None, inferred from
            cu_seqlens_k.
        flash_attn_kwargs: Additional keyword arguments to pass to
            scaled_dot_product_attention.

    Returns:
        The output tensor after applying scaled dot-product attention.
    """
    del max_seqlen_q, max_seqlen_k  # Unused, only needed to match flash attention API.
    swap_length_and_head_dim = partial(th.permute, dims=(0, 2, 1, 3))
    if cu_seqlens_k is None:
        cu_seqlens_k = cu_seqlens_q

    flash_attn_kwargs = (
        flash_attn_kwargs if flash_attn_kwargs is not None else FlashAttentionKWArgs()
    )

    if isinstance(qkv, tuple):
        if len(qkv) == 3:
            q, k, v = qkv
        elif len(qkv) == 2:
            q, kv = qkv
            k, v = th.chunk(kv, 2, dim=1)
        else:
            raise ValueError("qkv must be a tuple of (q, k, v) or (q, kv)")
    else:
        qkv = th.chunk(qkv, 3, dim=1)
        q, k, v = qkv
    q = q.squeeze(1)
    k = k.squeeze(1)
    v = v.squeeze(1)

    q_mask, (q_unpacked,) = unpack_tensors(cu_seqlens_q, q)
    kv_mask, (k_unpacked, v_unpacked) = unpack_tensors(cu_seqlens_k, k, v)

    attention_mask = combine_query_and_key_mask(q_mask, kv_mask)

    unpacked_out = th.stack(
        [
            scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attn_mask,
                dropout_p=flash_attn_kwargs.dropout_p,
                is_causal=flash_attn_kwargs.causal,
                scale=flash_attn_kwargs.softmax_scale,
            )
            for query, key, value, attn_mask in zip(
                swap_length_and_head_dim(q_unpacked),
                swap_length_and_head_dim(k_unpacked),
                swap_length_and_head_dim(v_unpacked),
                attention_mask,
            )
        ]
    )
    _, (packed_out,) = pack_tensors(kv_mask, swap_length_and_head_dim(unpacked_out))
    return packed_out
