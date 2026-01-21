from dataclasses import dataclass, field
from functools import partial
from typing import Literal

import torch as th

from ml_utils.components import SwiGLUMLP
from ml_utils.components.wrapper import ResidualConfig, Residual
from ml_utils.components.attention import SelfAttentionBias, BiasAttentionConfig
from ml_utils.components.base import BaseComponent
from ml_utils.torch_utils.types import BatchedMatrixTensor, BatchSequenceTensor


@dataclass(frozen=True)
class BiasTransformerEncoderBlockConfig:
    """Configuration for a transformer encoder block with bias attention.

    Attributes:
        attention_config: Configuration for the bias attention mechanism.
        residual_config: Configuration for the residual connections.
        swiglu_mode: Activation mode for the SwiGLU feedforward network.
        swiglu_upscale_factor: Upscale factor for the SwiGLU feedforward network
    """

    attention_config: BiasAttentionConfig = field(default_factory=BiasAttentionConfig)
    residual_config: ResidualConfig = field(default_factory=ResidualConfig)
    swiglu_mode: Literal["swish", "mp", "gelu", "silu"] = "swish"
    swiglu_upscale_factor: float = 2.0


class BiasTransformerEncoderBlock(BaseComponent):
    """Transformer encoder block with bias attention.

    This block consists of a bias attention mechanism followed by a feedforward network,
    both wrapped in residual connections.

    Args:
        config: Configuration for the encoder block.
    """

    def __init__(
        self,
        in_features: int,
        bias_features: int,
        config: BiasTransformerEncoderBlockConfig,
    ):
        super().__init__()
        self._in_features = in_features
        self._bias_features = bias_features
        self._config = config
        residual_wrapper = (
            partial(
                Residual,
                config=self._config.residual_config
            )
        )
        self._attention = residual_wrapper(
            SelfAttentionBias(
                in_features=in_features,
                bias_features=bias_features,
                config=self._config.attention_config,
            )
        )
        self._feed_forward = residual_wrapper(
            SwiGLUMLP(
                in_features=in_features,
                mode=self._config.swiglu_mode,
                upscale_factor=self._config.swiglu_upscale_factor,
            )
        )

    def forward(
        self,
        x: BatchSequenceTensor,
        bias: BatchedMatrixTensor,
        mask: th.Tensor | None = None,
    ) -> BatchSequenceTensor:
        """Forward pass of the encoder block.

        Args:
            x: Input tensor of shape (batch_size, seq_length, in_features).
            bias: Bias tensor of shape (batch_size, seq_length, bias_features).
            mask: Optional attention mask of shape (batch_size, seq_length).

        Returns:
            Output tensor of shape (batch_size, seq_length, in_features).
        """
        x = self._attention(x, bias, mask=mask)
        x = self._feed_forward(x)
        return x

    @property
    def in_features(self) -> int:
        return self._in_features

    @property
    def out_features(self) -> int:
        return self._in_features
    
    @property
    def in_bias_features(self) -> int:
        return self._bias_features