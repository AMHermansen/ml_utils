from dataclasses import dataclass


@dataclass
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
        alibi_slopes: (nheads,) or (batch_size, nheads), fp32. A bias of
            (-alibi_slope * |i + seqlen_k - seqlen_q - j|)
            is added to the attention score of query i and key j.
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
