from dataclasses import dataclass, field
from typing import Literal

from ._backends import FlashAttentionKWArgs


@dataclass(frozen=True)
class SelfAttentionConfig:
    """Configuration for self-attention modules.

    Args:
        nheads: Number of attention heads.
        split_qkv: Whether to use separate linear layers for Q, K, V projections.
            This might be advantageous for Muon optimizer.
        qk_norm_type: Type of normalization to apply to Q and K. Can be "layer",
            "rms", or None. Default is "rms". See `QKNorm` for details.
        qkv_bias: Whether to include bias in the QKV linear layer.
        flash_attention_kwargs: Additional keyword arguments for flash attention.
        use_flash_attention: Whether to use flash attention or standard attention.
    """

    nheads: int = 8
    qkv_bias: bool = False
    split_qkv: bool = False
    qk_norm_type: Literal["layer", "rms"] | None = "rms"
    flash_attention_kwargs: FlashAttentionKWArgs = field(
        default_factory=FlashAttentionKWArgs
    )
    use_flash_attention: bool = True


@dataclass(frozen=True)
class CrossAttentionConfig:
    """Configuration for cross-attention modules.

    Args:
        nheads: Number of attention heads.
        kv_in_dim: Dimension of the key/value input. If None, defaults to the query
            input dimension.
        q_bias: Whether to include bias in the query linear layer.
        kv_bias: Whether to include bias in the key/value linear layer.
        qk_norm_type: Type of normalization to apply to Q and K. Can be "layer",
            "rms", or None. Default is "rms". See `QKNorm` for details.
        flash_attention_kwargs: Additional keyword arguments for flash attention.
        use_flash_attention: Whether to use flash attention or standard attention.
    """

    nheads: int = 8
    kv_in_dim: int | None = None
    q_bias: bool = False
    kv_bias: bool = False
    qk_norm_type: Literal["layer", "rms"] | None = "rms"
    flash_attention_kwargs: FlashAttentionKWArgs = field(
        default_factory=FlashAttentionKWArgs
    )
    use_flash_attention: bool = True


@dataclass(frozen=True)
class BiasAttentionConfig:
    """Configuration for bias attention modules.

    Args:
        nheads: Number of attention heads.
        head_dim: Dimension of each attention head.
        pre_norm_type: Type of normalization to apply before attention. Can be "layer",
            "rms", or None. Default is "rms".
        pre_norm_bias_type: Type of normalization to apply to bias before attention.
        qkv_bias: Whether to include bias in the QKV linear layer.
        out_bias: Whether to include bias in the output linear layer.
        bias_bias: Whether to include bias in the bias linear layer.
        gate_bias: Whether to include bias in the gate linear layer.
    """

    nheads: int = 8
    head_dim: int = 32
    pre_norm_type: Literal["layer", "rms"] | None = "rms"
    pre_norm_bias_type: Literal["layer", "rms"] | None = "rms"
    qkv_bias: bool = False
    out_bias: bool = False
    bias_bias: bool = False
    gate_bias: bool = False
