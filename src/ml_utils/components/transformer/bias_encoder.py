from dataclasses import dataclass, field

from torch import nn
import torch as th

from .bias_encoder_block import BiasTransformerEncoderBlock, BiasTransformerEncoderBlockConfig
from ml_utils.components.base import BaseComponent
from ml_utils.torch_utils.types import BatchedMatrixTensor, BatchSequenceTensor


@dataclass(frozen=True)
class BiasTransformerEncoderConfig:
    """Configuration for a full Bias Transformer Encoder.

    Args:

    """
    num_layers: int = 6
    transformer_config: BiasTransformerEncoderBlockConfig = field(
        default_factory=BiasTransformerEncoderBlockConfig
    )


class BiasTransformerEncoder(BaseComponent):
    """Transformer Encoder composed of multiple Bias Transformer Encoder Blocks.

    Args:
        in_features: Input feature dimension.
        bias_features: Bias feature dimension.
        config: Configuration for the Bias Transformer Encoder.
    """

    def __init__(
        self,
        in_features: int,
        bias_features: int,
        config: BiasTransformerEncoderConfig,
    ):
        super().__init__()
        self._config = config
        self._in_features = in_features
        self._bias_features = bias_features

        self._layers = nn.ModuleList([
            BiasTransformerEncoderBlock(
                in_features=in_features,
                bias_features=bias_features,
                config=config.transformer_config,
            )
            for _ in range(config.num_layers)
        ])

    @property
    def out_features(self) -> int:
        return self._in_features

    @property
    def in_features(self) -> int | None:
        return self._in_features

    def forward(
        self,
        x: BatchSequenceTensor,
        bias: BatchedMatrixTensor,
        mask: th.Tensor | None = None,
    ) -> BatchSequenceTensor:
        """Forward pass through the Bias Transformer Encoder.

        Args:
            x: Input tensor of shape (batch_size, seq_length, in_features).
            bias: Bias tensor of shape (batch_size, seq_length, seq_length, bias_features).
            mask: Optional mask tensor of shape (batch_size, seq_length).

        Returns:

        """
        for layer in self._layers:
            x = layer(x, bias, mask=mask)
        return x
