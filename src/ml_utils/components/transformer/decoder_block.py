from dataclasses import dataclass, field

from typing_extensions import override

from ml_utils.components.attention import (
    CrossAttentionConfig,
    PackedCrossAttention,
    PackedSelfAttention,
    SelfAttentionConfig,
)
from ml_utils.components.base import BaseComponent
from ml_utils.components.swiglu import SwiGLUMLP
from ml_utils.components.wrappers import Residual, ResidualConfig
from ml_utils.torch_utils.types import CulensTensor, PackedTensor


@dataclass(frozen=True)
class TransformerDecoderBlockConfig:
    """Configuration for a Transformer block.

    Configuration for encoder transformer block consisting of
    self-attention, cross-attention, and SwiGLU MLP.

    Args:
        self_attention_config: Configuration for the self-attention
            layer. See `SelfAttentionConfig` for details.
        cross_attention_config: Configuration for the cross-attention
            layer. See `CrossAttentionConfig` for details.
        residual_config: Configuration for the residual connections.
            See `ResidualConfig` for details.
    """
    self_attention_config: SelfAttentionConfig = field(
        default_factory=SelfAttentionConfig
    )
    cross_attention_config: CrossAttentionConfig = field(
        default_factory=CrossAttentionConfig
    )
    residual_config: ResidualConfig = field(default_factory=ResidualConfig)

    do_self_attention_before_cross_attention: bool = True


class TransformerDecoderBlock(BaseComponent):
    """Transformer block with self-attention and SwiGLU MLP.

    This is a Transformer decoder block, that includes both self-attention
    and cross-attention mechanisms, followed by a SwiGLU MLP. The order of
    self-attention and cross-attention can be configured.

    It is important to note that this block is designed for encoder-decoder setups,
    and is NOT the block that should be used for a purely autoregressive model.

    Args:
        in_features: Number of input features.
        config: Configuration for the Transformer block.

    """
    def __init__(
        self,
        in_features: int,
        config: TransformerDecoderBlockConfig
    ):
        """Transformer block consisting of self-attention and SwiGLU MLP.

        Args:
            in_features: Number of input features.
            config: Configuration for the Transformer block.
        """
        super().__init__()
        self._config = config

        self.self_attention = Residual(
            PackedSelfAttention(
                in_features=in_features,
                config=config.self_attention_config
            ),
            config.residual_config,
        )
        self.cross_attention = Residual(
            PackedCrossAttention(
                in_features=in_features,
                config=config.cross_attention_config
            ),
            config.residual_config,
        )
        self._feed_forward = Residual(
            SwiGLUMLP(in_features=in_features),
            config.residual_config,
        )

    def forward(
        self,
        q_sequence: PackedTensor,
        kv_sequence: PackedTensor,
        cu_seqlens_q: CulensTensor,
        cu_seqlens_kv: CulensTensor,
        max_seqlen_q: int | None = None,
        max_seqlen_kv: int | None = None,
    ) -> PackedTensor:
        """Forward pass through the Transformer block.

        Args:
            q_sequence: The primary input sequence tensor of shape (total_tokens, in_features).
            kv_sequence: The conditioning input sequence tensor of shape (
            total_tokens, kv_features).
            cu_seqlens_q: Cumulative sequence lengths for the primary sequence.
            cu_seqlens_kv: Cumulative sequence lengths for the conditioning sequence.
            max_seqlen_q: Maximum sequence length for the primary sequence.
            max_seqlen_kv: Maximum sequence length for the conditioning sequence.

        Returns:
            Output tensor of shape (total_tokens, in_features).
        """
        if self._config.do_self_attention_before_cross_attention:
            x = self.self_attention(
                q_sequence,
                cu_seqlens_q,
                max_seqlen_q,
            )
            x = self.cross_attention(
                x,
                kv_sequence,
                cu_seqlens_q,
                cu_seqlens_kv,
                max_seqlen_q,
                max_seqlen_kv,
            )
        else:
            x = self.cross_attention(
                q_sequence,
                kv_sequence,
                cu_seqlens_q,
                cu_seqlens_kv,
                max_seqlen_q,
                max_seqlen_kv,
            )
            x = self.self_attention(
                x,
                cu_seqlens_q,
                max_seqlen_q,
            )
        return self._feed_forward(x)

    @property
    def use_flash_attention(self) -> bool | None:
        """Whether flash attention is used in the self-attention layer."""
        if (
            self.self_attention.use_flash_attention
            != self.cross_attention.use_flash_attention
        ):
            return None
        return self.attention.use_flash_attention

    @use_flash_attention.setter
    def use_flash_attention(self, value: bool):
        self.self_attention.use_flash_attention = value
        self.cross_attention.use_flash_attention = value

    @override
    @property
    def in_features(self) -> int:
        return self.attention.in_features

    @override
    @property
    def out_features(self) -> int:
        return self.attention.out_features
