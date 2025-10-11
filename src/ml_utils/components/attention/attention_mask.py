import jaxtyping as jt
import torch as th


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
