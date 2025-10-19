from dataclasses import dataclass, field

import torch as th
from torch import nn
from typing_extensions import override

from ml_utils.components.base import BaseComponent
from ml_utils.torch_utils.misc import ParameterNoWeightDecay
from ml_utils.torch_utils.packing import (
    prepend_tokens_to_packed_tensor,
    remove_tokens_from_packed_tensor,
)
from ml_utils.torch_utils.types import CulensTensor, PackedTensor
from ml_utils.utils import exists, maybe_add, maybe_subtract

from .encoder_block import TransformerEncoderBlock, TransformerEncoderBlockConfig


@dataclass
class TransformerEncoderConfig:
    """Configuration for the full Transformer Encoder.

    Args:
        num_layers: Number of Transformer Encoder Blocks.
        num_registers: Number of learnable register tokens to prepend to the input.
            0 means no registers are used. Default is 0. Registers are similar to
            class tokens. But they are not part of the returned output.
        num_class_tokens: Number of class tokens to prepend to the input. Class
            tokens are returned as part of the output. Default is 0.
            If num_class_tokens > 0, then the returned output will contain more
            tokens than the input (by num_class_tokens * batch_size).
        transformer_config: Configuration for each Transformer Encoder Block.
            If None, default configuration is used.
            See `TransformerEncoderBlockConfig` for details.
    """

    num_layers: int = 6
    num_registers: int = 0
    num_class_tokens: int = 0
    transformer_config: TransformerEncoderBlockConfig = field(default_factory=TransformerEncoderBlockConfig)


class TransformerEncoder(BaseComponent):
    """Transformer Encoder consisting of multiple Transformer Encoder Blocks.

    Args:
        in_features: Input feature dimension.
        config: Configuration for the Transformer Encoder. See
            `TransformerEncoderConfig`, for details.
    """
    def __init__(
        self,
        in_features: int,
        config: TransformerEncoderConfig
    ):
        """Transformer Encoder consisting of multiple TransformerEncoderBlocks.

        Args:
            in_features: Input feature dimension.
            config: Configuration for the Transformer Encoder. See
                `TransformerEncoderConfig`, for details.
        """
        super().__init__()
        config = config if exists(config) else TransformerEncoderConfig()
        self._config = config
        self._in_features = in_features
        self._num_layers = config.num_layers
        self._num_registers = config.num_registers
        self._num_class_tokens = config.num_class_tokens
        self._transformer_config = config.transformer_config

        self._layers = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    in_features=self._in_features,
                    config=self._transformer_config,
                )
                for _ in range(self._num_layers)
            ]
        )

        self._registers = ParameterNoWeightDecay(
            th.randn(self._num_registers, self._in_features)
        ) if self._num_registers > 0 else None
        self._class_tokens = ParameterNoWeightDecay(
            th.randn(self._num_class_tokens, self._in_features)
        ) if self._num_class_tokens > 0 else None

    def forward(
        self,
        x: PackedTensor,
        cu_seqlens: CulensTensor,
        max_seqlen: int | None = None
    ) -> tuple[PackedTensor, CulensTensor, int | None]:
        """Forward pass through the Transformer Encoder.

        Args:
            x: Packed input tensor of shape (total_seq_len, in_features).
            cu_seqlens: Cumulative sequence lengths tensor.
                Shape: (batch_size + 1,).
            max_seqlen: Optional maximum sequence length.

        Returns:
            tuple containing:
                - Packed output tensor of
                    shape (total_seq_len + num_class_tokens * batch_size, in_features).
                - Updated cumulative sequence lengths tensor.
                - Updated maximum sequence length.
        """
        x, cu_seqlens, max_seqlen = self.maybe_add_class_tokens(
            x,
            cu_seqlens,
            max_seqlen,
        )
        x, cu_seqlens, max_seqlen = self.maybe_add_registers(
            x,
            cu_seqlens,
            max_seqlen,
        )
        if not exists(max_seqlen):
            max_seqlen = th.diff(cu_seqlens).max().item()

        for layer in self._layers:
            x = layer(x, cu_seqlens, max_seqlen)

        return self.maybe_remove_registers(
            x,
            cu_seqlens,
            max_seqlen
        )

    def maybe_add_class_tokens(
        self,
        x: PackedTensor,
        cu_seqlens: CulensTensor,
        max_seqlen: int | None = None,
    ) -> tuple[PackedTensor, CulensTensor, int | None]:
        """Prepend class tokens to the input if configured to do so.

        Args:
            x: Packed input tensor of shape (total_seq_len, in_features).
            cu_seqlens: Cumulative sequence lengths tensor.
             Shape: (batch_size + 1,).
            max_seqlen: Optional maximum sequence length.

        Returns:
            tuple containing:
                - Packed output tensor with class tokens prepended.
                - Updated cumulative sequence lengths tensor.
                - Updated maximum sequence length.
        """
        if self.has_cls_tokens:
            x, cu_seqlens = prepend_tokens_to_packed_tensor(
                x,
                cu_seqlens,
                self._class_tokens,
            )
        return x, cu_seqlens, maybe_add(max_seqlen, self._num_class_tokens)

    def maybe_add_registers(
        self,
        x: PackedTensor,
        cu_seqlens: CulensTensor,
        max_seqlen: int | None = None,
    ) -> tuple[PackedTensor, CulensTensor, int | None]:
        """Prepend register tokens to the input if configured to do so.

        Args:
            x: Packed input tensor of shape (total_seq_len, in_features).
            cu_seqlens: Cumulative sequence lengths tensor.
                Shape: (batch_size + 1,).
            max_seqlen: Optional maximum sequence length.

        Returns:
            tuple containing:
                - Packed output tensor with register tokens prepended.
                - Updated cumulative sequence lengths tensor.
                - Updated maximum sequence length.
        """
        if self.has_registers:
            x, cu_seqlens = prepend_tokens_to_packed_tensor(
                x,
                cu_seqlens,
                self._registers,
            )
        return x, cu_seqlens, maybe_add(max_seqlen, self._num_registers)

    def maybe_remove_registers(
        self,
        x: PackedTensor,
        cu_seqlens: CulensTensor,
        max_seqlen: int | None,
    ) -> tuple[PackedTensor, CulensTensor, int | None]:
        """Remove register tokens from the input if configured to do so.

        Args:
            x: Packed input tensor of shape (total_seq_len, in_features).
            cu_seqlens: Cumulative sequence lengths tensor.
                Shape: (batch_size + 1,).
            max_seqlen: Optional maximum sequence length.

        Returns:
            tuple containing:
                - Packed output tensor with register tokens removed.
                - Updated cumulative sequence lengths tensor.
                - Updated maximum sequence length.
        """
        if self.has_registers:
            x, cu_seqlens = remove_tokens_from_packed_tensor(
                x,
                cu_seqlens,
                self._num_registers,
            )
        return x, cu_seqlens, maybe_subtract(max_seqlen, self._num_registers)

    @property
    def use_flash_attention(self) -> bool:
        """Whether any of the encoder blocks use flash attention."""
        assert all(
            layer.use_flash_attention == self._layers[0].use_flash_attention
            for layer in self._layers
        )
        # Type checker only assumes that self._layers is a ModuleList[Module].
        # And gets confused that non-Module attributes may not exist.
        return self._layers[0].use_flash_attention  # type: ignore

    @use_flash_attention.setter
    def use_flash_attention(self, value: bool):
        """Set whether to use flash attention in all encoder blocks."""
        for layer in self._layers:
            layer.use_flash_attention = value

    @property
    def has_registers(self) -> bool:
        """Whether the encoder has register tokens."""
        return self._num_registers > 0

    @property
    def has_cls_tokens(self) -> bool:
        """Whether the encoder has class tokens."""
        return self._num_class_tokens > 0

    @override
    @property
    def in_features(self) -> int | None:
        """Input feature dimension."""
        return self._in_features

    @override
    @property
    def out_features(self) -> int | None:
        """Output feature dimension."""
        return self._in_features
