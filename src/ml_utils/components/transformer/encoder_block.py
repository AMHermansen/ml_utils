from dataclasses import dataclass, field
from typing import override

from ml_utils.components.attention import PackedSelfAttention, SelfAttentionConfig
from ml_utils.components.base import BaseComponent
from ml_utils.components.swiglu import SwiGLUMLP
from ml_utils.components.wrappers import Residual, ResidualConfig
from ml_utils.torch_utils.types import CulensTensor, PackedTensor


@dataclass(frozen=True)
class TransformerEncoderBlockConfig:
    """Configuration for a Transformer block.

    Configuration for encoder transformer block consisting of
    self-attention followed by SwiGLU MLP.

    Args:
        attention_config: Configuration for the self-attention
            layer. See `SelfAttentionConfig` for details.
        residual_config: Configuration for the residual connections.
            See `ResidualConfig` for details.
    """
    attention_config: SelfAttentionConfig = field(default_factory=SelfAttentionConfig)
    residual_config: ResidualConfig = field(default_factory=ResidualConfig)


class TransformerEncoder(BaseComponent):
    """Transformer block with self-attention and SwiGLU MLP.

    This block consists of a self-attention layer followed by a SwiGLU MLP.
    This can be used both autoregressively and non-autoregressively, depending on
    the `causal` flag in the attention configuration.
    Sometimes autoregressive models are confusingly called "decoder" models.

    Args:
        in_features: Number of input features.
        config: Configuration for the Transformer block.

    """
    def __init__(
        self,
        in_features: int,
        config: TransformerEncoderBlockConfig
    ):
        """Transformer block consisting of self-attention and SwiGLU MLP.

        Args:
            in_features: Number of input features.
            config: Configuration for the Transformer block.
        """
        super().__init__()
        self._config = config

        self.attention = Residual(
            PackedSelfAttention(
                in_features=in_features,
                config=config.attention_config
            ),
            config.residual_config,
        )
        self._feed_forward = Residual(
            SwiGLUMLP(in_features=in_features),
            config.residual_config,
        )

    def forward(
        self,
        x: PackedTensor,
        cu_seqlens: CulensTensor,
        max_seqlen: int | None
    ) -> PackedTensor:
        """Forward pass through the Transformer block.

        Args:
            x: Input tensor of shape (total_tokens, in_features).
            cu_seqlens: Cumulative sequence lengths tensor.
            max_seqlen: Maximum sequence length.

        Returns:
            Output tensor of shape (total_tokens, in_features).
        """
        x = self.attention(x, cu_seqlens, max_seqlen)
        return self._feed_forward(x)

    @override
    @property
    def in_features(self) -> int:
        return self.attention.in_features

    @override
    @property
    def out_features(self) -> int:
        return self.attention.out_features
