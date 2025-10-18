from dataclasses import dataclass

import jaxtyping as jt
import torch as th


@dataclass(frozen=True)
class FlashAttentionKWArgs:
    """Structure of keyword arguments to pass into varlen_attn functions.

    These are the common arguments that will generally stay fixed during training.

    Attributes:
        dropout_p: float. Dropout probability.
        softmax_scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim).
        causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive
            modeling).
        window_size: (left, right). If not (-1, -1), implements sliding window local
            attention.
        softcap: float. Anything > 0 activates softcapping attention.
        deterministic: bool. Whether to use the deterministic implementation of the
            backward pass, which is slightly slower and uses more memory. The forward
            pass is always deterministic.

    """

    dropout_p: float = 0.0
    softmax_scale: float | None = None
    causal: bool = False
    window_size: tuple[int, int] = (-1, -1)  # -1 means infinite context window
    softcap: float = 0.0  # 0.0 means deactivated
    deterministic: bool = False


def combine_query_and_key_mask(
    query_mask: jt.Bool[th.Tensor, "batch query_len"],
    key_mask: jt.Bool[th.Tensor, "batch key_len"],
) -> jt.Float[th.Tensor, "batch query_len key_len"]:
    """Combines a query mask and a key mask into a single attention mask.

    Args:
        query_mask: A boolean tensor of shape (batch, query_len) indicating valid query
            positions.
        key_mask: A boolean tensor of shape (batch, key_len) indicating valid key
            positions.

    Returns:
        A boolean tensor of shape (batch, query_len, key_len) where True indicates that
        the corresponding query-key pair is valid (i.e., both query and key positions
        are valid).
    """
    query_mask = query_mask.unsqueeze(2)
    key_mask = key_mask.unsqueeze(1)
    return query_mask & key_mask
