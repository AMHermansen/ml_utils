from dataclasses import dataclass, field
from functools import partial
from typing import Literal, TypeAlias

import jaxtyping as jt
import torch as th
from typing_extensions import override

from ml_utils.components.attention import PackedSelfAttention, SelfAttentionConfig
from ml_utils.components.base import BaseComponent
from ml_utils.components.swiglu import SwiGLUMLP
from ml_utils.components.wrapper import Residual, ResidualConfig, ResidualWithContext
from ml_utils.torch_utils.types import CulensTensor, PackedTensor
from ml_utils.utils import exists

PackedContextTensor: TypeAlias = jt.Float[th.Tensor, " packed_length context_dim"]
BatchedContextTensor: TypeAlias = jt.Float[th.Tensor, " batch_size context_dim"]


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
        swiglu_mode: Activation function to use in the SwiGLU MLP.
            Options are "swish", "mp", "gelu", and "silu". See `SwiGLUMLP` for details.
        swiglu_upscale_factor: Factor to upscale the hidden dimension
            in the SwiGLU MLP. Default is 2.0.
    """

    attention_config: SelfAttentionConfig = field(default_factory=SelfAttentionConfig)
    residual_config: ResidualConfig = field(default_factory=ResidualConfig)
    swiglu_mode: Literal["swish", "mp", "gelu", "silu"] = "swish"
    swiglu_upscale_factor: float = 2.0


class TransformerEncoderBlock(BaseComponent):
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
        config: TransformerEncoderBlockConfig,
        context_dim: int = 0,
    ) -> None:
        """Transformer block consisting of self-attention and SwiGLU MLP.

        Context dimension changes how the residual connections are handled. If context
        is provided, the residual connections will use adaptive normalization to
        provide conditioning for the sequence. This was the most successful way of
         doing conditioning for DiffusionTransformers(DiT).
        See https://arxiv.org/pdf/2212.09748

        Args:
            in_features: Number of input features.
            config: Configuration for the Transformer block.
            context_dim: Dimension of the context vector for conditioning.
        """
        super().__init__()
        config = config if exists(config) else TransformerEncoderBlockConfig()
        self._config = config
        self._context_dim = context_dim
        residual_wrapper = (
            partial(
                ResidualWithContext,
                context_dim=context_dim,
                config=config.residual_config,
            )
            if context_dim
            else partial(Residual, config=config.residual_config)
        )

        self.attention = residual_wrapper(
            PackedSelfAttention(
                in_features=in_features, config=config.attention_config
            ),
        )
        self._feed_forward = residual_wrapper(
            SwiGLUMLP(
                in_features=in_features,
                mode=config.swiglu_mode,
                upscale_factor=config.swiglu_upscale_factor,
            ),
        )

    def forward(
        self,
        x: PackedTensor,
        cu_seqlens: CulensTensor,
        max_seqlen: int | None = None,
        *,
        context: BatchedContextTensor | PackedContextTensor | None = None,
    ) -> PackedTensor:
        """Forward pass through the Transformer block.

        Args:
            x: Input tensor of shape (total_tokens, in_features).
            cu_seqlens: Cumulative sequence lengths tensor.
            max_seqlen: Maximum sequence length.
            context: Optional context tensor for conditioning. If provided, it should
                either have shape (batch_size, context_dim) or
                (total_tokens, context_dim).


        Returns:
            Output tensor of shape (total_tokens, in_features).

        Raises:
            ValueError: If context is provided but the block is not configured to use
                context or vice versa. Also raises if the context dimension does not
                match.
        """
        if self.using_context != exists(context):
            raise ValueError(
                f"Context dimension mismatch. Expected context: {self.using_context}, "
                f"but got context: {exists(context)}"
            )

        # If not using context, proceed with standard attention and feed-forward
        if not self.using_context:
            x = self.attention(x, cu_seqlens, max_seqlen)
            return self._feed_forward(x)

        # Validate provided context
        assert exists(context), "Context must be provided when context_dim > 0"
        if context.shape[-1] != self.context_dim:
            raise ValueError(
                f"Context dimension mismatch. Expected: {self.context_dim}, "
                f"but got: {context.shape[-1]}"
            )

        # Expand context if needed
        if len(context) == len(cu_seqlens) - 1:
            context = th.repeat_interleave(context, th.diff(cu_seqlens), dim=0)

        x = self.attention(x, cu_seqlens, max_seqlen, context=context)
        return self._feed_forward(x, context=context)

    @property
    def use_flash_attention(self) -> bool:
        """Whether flash attention is used in the self-attention layer."""
        return self.attention.use_flash_attention

    @use_flash_attention.setter
    def use_flash_attention(self, value: bool):
        self.attention.use_flash_attention = value

    @property
    @override
    def in_features(self) -> int:
        return self.attention.in_features

    @property
    @override
    def out_features(self) -> int:
        return self.attention.out_features

    @property
    def context_dim(self) -> int:
        """Dimension of the context vector."""
        return self._context_dim

    @property
    def using_context(self) -> bool:
        """Whether the encoder is using context vectors."""
        return self._context_dim > 0
