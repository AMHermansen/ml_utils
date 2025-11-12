from dataclasses import dataclass, field
from functools import partial
from typing import Literal, TypeAlias

import jaxtyping as jt
import torch as th
from typing_extensions import override

from ml_utils.components.attention import (
    CrossAttentionConfig,
    PackedCrossAttention,
    PackedSelfAttention,
    SelfAttentionConfig,
)
from ml_utils.components.base import BaseComponent
from ml_utils.components.swiglu import SwiGLUMLP
from ml_utils.components.wrapper import Residual, ResidualConfig, ResidualWithContext
from ml_utils.torch_utils.types import CulensTensor, PackedTensor
from ml_utils.utils import exists

PackedContextTensor: TypeAlias = jt.Float[th.Tensor, " packed_length context_dim"]
BatchedContextTensor: TypeAlias = jt.Float[th.Tensor, " batch_size context_dim"]


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
        swiglu_mode: Activation function to use in the SwiGLU MLP.
            Options are "swish", "mp", "gelu", and "silu". See `SwiGLUMLP` for details.
        swiglu_upscale_factor: Factor to upscale the hidden dimension
            in the SwiGLU MLP. Default is 2.0.
    """

    self_attention_config: SelfAttentionConfig = field(
        default_factory=SelfAttentionConfig
    )
    cross_attention_config: CrossAttentionConfig = field(
        default_factory=CrossAttentionConfig
    )
    residual_config: ResidualConfig = field(default_factory=ResidualConfig)

    do_self_attention_before_cross_attention: bool = True
    swiglu_mode: Literal["swish", "mp", "gelu", "silu"] = "swish"
    swiglu_upscale_factor: float = 2.0


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
        config: TransformerDecoderBlockConfig,
        context_dim: int = 0,
    ):
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
        config = config if exists(config) else TransformerDecoderBlockConfig()
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

        self.self_attention = residual_wrapper(
            PackedSelfAttention(
                in_features=in_features, config=config.self_attention_config
            ),
        )
        self.cross_attention = residual_wrapper(
            PackedCrossAttention(
                in_features=in_features, config=config.cross_attention_config
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
        q_sequence: PackedTensor,
        kv_sequence: PackedTensor,
        cu_seqlens_q: CulensTensor,
        cu_seqlens_kv: CulensTensor,
        max_seqlen_q: int | None = None,
        max_seqlen_kv: int | None = None,
        *,
        context: BatchedContextTensor | PackedContextTensor | None = None,
    ) -> PackedTensor:
        """Forward pass through the Transformer block.

        Args:
            q_sequence: The primary input sequence tensor of shape (total_tokens,
                in_features).
            kv_sequence: The conditioning input sequence tensor of shape (
            total_tokens, kv_features).
            cu_seqlens_q: Cumulative sequence lengths for the primary sequence.
            cu_seqlens_kv: Cumulative sequence lengths for the conditioning sequence.
            max_seqlen_q: Maximum sequence length for the primary sequence.
            max_seqlen_kv: Maximum sequence length for the conditioning sequence.
            context: Optional context tensor for conditioning, either batched or packed.

        Returns:
            Output tensor of shape (total_tokens, in_features).
        """
        if self.using_context != exists(context):
            raise ValueError(
                f"Context dimension mismatch. Expected context: {self.using_context}, "
                f"but got context: {exists(context)}"
            )
        if exists(context) and context.shape[-1] != self.context_dim:
            raise ValueError(
                f"Context dimension mismatch. Expected: {self.context_dim}, "
                f"but got: {context.shape[-1]}"
            )
        if exists(context) and len(context) != len(q_sequence):
            context = th.repeat_interleave(context, th.diff(cu_seqlens_q), dim=0)

        if self._config.do_self_attention_before_cross_attention:
            x = self._apply_self_attention(
                q_sequence,
                cu_seqlens_q,
                max_seqlen_q,
                context=context,
            )
            x = self._apply_cross_attention(
                x,
                kv_sequence,
                cu_seqlens_q,
                cu_seqlens_kv,
                max_seqlen_q,
                max_seqlen_kv,
                context=context,
            )
        else:
            x = self._apply_cross_attention(
                q_sequence,
                kv_sequence,
                cu_seqlens_q,
                cu_seqlens_kv,
                max_seqlen_q,
                max_seqlen_kv,
                context=context,
            )
            x = self._apply_self_attention(
                x,
                cu_seqlens_q,
                max_seqlen_q,
                context=context,
            )
        return self._apply_feed_forward(x, context=context)

    def _apply_self_attention(
        self,
        x: PackedTensor,
        cu_seqlens: CulensTensor,
        max_seqlen: int | None,
        context: PackedContextTensor | None = None,
    ) -> PackedTensor:
        """Helper to handle optional context."""
        if self.using_context:
            return self.self_attention(
                x,
                cu_seqlens,
                max_seqlen,
                context=context,
            )
        return self.self_attention(
            x,
            cu_seqlens,
            max_seqlen,
        )

    def _apply_cross_attention(
        self,
        x: PackedTensor,
        kv_sequence: PackedTensor,
        cu_seqlens_q: CulensTensor,
        cu_seqlens_kv: CulensTensor,
        max_seqlen_q: int | None,
        max_seqlen_kv: int | None,
        context: PackedContextTensor | None = None,
    ) -> PackedTensor:
        """Helper to handle optional context."""
        if self.using_context:
            return self.cross_attention(
                x,
                kv_sequence,
                cu_seqlens_q,
                cu_seqlens_kv,
                max_seqlen_q,
                max_seqlen_kv,
                context=context,
            )
        return self.cross_attention(
            x,
            kv_sequence,
            cu_seqlens_q,
            cu_seqlens_kv,
            max_seqlen_q,
            max_seqlen_kv,
        )

    def _apply_feed_forward(
        self,
        x: PackedTensor,
        context: PackedContextTensor | None = None,
    ) -> PackedTensor:
        """Helper to handle optional context."""
        if self.using_context:
            return self._feed_forward(
                x,
                context=context,
            )
        return self._feed_forward(
            x,
        )

    @property
    def use_flash_attention(self) -> bool | None:
        """Whether flash attention is used in the self-attention layer."""
        if (
            self.self_attention.use_flash_attention
            != self.cross_attention.use_flash_attention
        ):
            return None
        assert isinstance(self.self_attention.use_flash_attention, bool)
        return self.self_attention.use_flash_attention

    @use_flash_attention.setter
    def use_flash_attention(self, value: bool):
        self.self_attention.use_flash_attention = value
        self.cross_attention.use_flash_attention = value

    @property
    @override
    def in_features(self) -> int:
        return self.self_attention.in_features

    @property
    @override
    def out_features(self) -> int:
        return self.self_attention.out_features

    @property
    def context_dim(self) -> int:
        """Dimension of the context vector."""
        return self._context_dim

    @property
    def using_context(self) -> bool:
        """Whether the encoder is using context vectors."""
        return self._context_dim > 0
