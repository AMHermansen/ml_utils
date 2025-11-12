from dataclasses import dataclass, field
from typing import TypeAlias, cast, overload

import jaxtyping as jt
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

from .decoder_block import TransformerDecoderBlock, TransformerDecoderBlockConfig

PackedContextTensor: TypeAlias = jt.Float[th.Tensor, " packed_length context_dim"]
BatchedContextTensor: TypeAlias = jt.Float[th.Tensor, " batch_size context_dim"]


@dataclass
class TransformerDecoderConfig:
    """Configuration for the full Transformer Decoder.

    Args:
        num_layers: Number of TransformerDecoderBlocks.
        num_registers: Number of register tokens to prepend to the input.
        transformer_config: Configuration for each TransformerDecoderBlock.
    """

    num_layers: int = 6
    num_registers: int = 0
    transformer_config: TransformerDecoderBlockConfig = field(
        default_factory=TransformerDecoderBlockConfig
    )


class TransformerDecoder(BaseComponent):
    """Transformer Decoder consisting of multiple TransformerDecoderBlocks.

    This is not an autoregressive transformer. This module takes in a
    primary input sequence (q_sequence) and a conditioning sequence
    (kv_sequence) and processes them through multiple decoder blocks.

    Args:
        in_features: Input feature dimension.
        config: Configuration for the Transformer Decoder.
    """

    def __init__(
        self,
        in_features: int,
        config: TransformerDecoderConfig,
        context_dim: int = 0,
    ):
        """Transformer Decoder consisting of multiple TransformerDecoderBlocks.

        Args:
            in_features: Input feature dimension.
            config: Configuration for the Transformer Decoder.
            context_dim: Dimension of the context vector. If 0, no context is used.
        """
        config = config if exists(config) else TransformerDecoderConfig()
        super().__init__()
        self._config = config
        self._context_dim = context_dim
        self._in_features = in_features
        self._num_layers = config.num_layers
        self._num_registers = config.num_registers
        self._transformer_config = config.transformer_config

        self._layers = nn.ModuleList([
            TransformerDecoderBlock(
                in_features=in_features,
                config=config.transformer_config,
                context_dim=context_dim,
            )
            for _ in range(config.num_layers)
        ])

        self._registers = (
            ParameterNoWeightDecay(th.randn(self._num_registers, self._in_features))
            if self._num_registers > 0
            else None
        )

    def forward(
        self,
        q_sequence: PackedTensor,
        kv_sequence: PackedTensor,
        cu_seqlens_q: CulensTensor,
        cu_seqlens_kv: CulensTensor,
        max_seqlen_q: int | None = None,
        max_seqlen_kv: int | None = None,
        context: BatchedContextTensor | PackedContextTensor | None = None,
    ) -> tuple[PackedTensor, CulensTensor, int | None]:
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

        Returns:
            Output tensor of shape (total_tokens, in_features).
        """
        q_sequence, cu_seqlens_q, max_seqlen_q = self.maybe_add_registers(
            q_sequence,
            cu_seqlens_q,
            max_seqlen_q,
        )
        context = self.maybe_expand_context(context, cu_seqlens_q)
        if not exists(max_seqlen_q):
            max_seqlen_q = cast("int", th.diff(cu_seqlens_q).max().item())
            assert isinstance(max_seqlen_q, int)
        if not exists(max_seqlen_kv):
            max_seqlen_kv = cast("int", th.diff(cu_seqlens_kv).max().item())
            assert isinstance(max_seqlen_kv, int)

        for layer in self._layers:
            q_sequence = layer(
                q_sequence=q_sequence,
                kv_sequence=kv_sequence,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_kv=cu_seqlens_kv,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_kv=max_seqlen_kv,
                context=context,
            )
        return self.maybe_remove_registers(q_sequence, cu_seqlens_q, max_seqlen_q)

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
            assert self._registers is not None
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

    @overload
    def maybe_expand_context(
        self,
        context: None,
        cu_seqlens: CulensTensor,
    ) -> None: ...

    @overload
    def maybe_expand_context(
        self,
        context: BatchedContextTensor | PackedContextTensor,
        cu_seqlens: CulensTensor,
    ) -> jt.Float[th.Tensor, "new_total_length context_dim"]: ...

    def maybe_expand_context(
        self,
        context: BatchedContextTensor | PackedContextTensor | None,
        cu_seqlens: CulensTensor,
    ) -> jt.Float[th.Tensor, "new_total_length context_dim"] | None:
        """Expand context to match the packed input if necessary.

        Args:
            context: Context tensor of shape
                (batch_size, context_dim) or
                (total_seq_len, context_dim).
            cu_seqlens: Cumulative sequence lengths tensor.
                Shape: (batch_size + 1,).

        Returns:
            Expanded context tensor of shape
                (new_total_length, context_dim) or None.

        Raises:
            ValueError: If context is packed but registers or class tokens are used.
        """
        if not exists(context):
            return None

        if len(context) == len(cu_seqlens) - 1:
            # Context is batched, expand to packed
            return th.repeat_interleave(
                context,
                th.diff(cu_seqlens),
                dim=0,
            )

        if self.has_registers:
            raise ValueError("Cannot use per-token context when using register.")
        return context

    @property
    def use_flash_attention(self) -> bool:
        """Whether any of the encoder blocks use flash attention."""
        assert all(
            layer.use_flash_attention == self._layers[0].use_flash_attention  # type: ignore
            for layer in self._layers
        )
        # Type checker only assumes that self._layers is a ModuleList[Module].
        # And gets confused that non-Module attributes may not exist.
        return self._layers[0].use_flash_attention  # type: ignore

    @use_flash_attention.setter
    def use_flash_attention(self, value: bool):
        """Set whether to use flash attention in all encoder blocks."""
        for layer in self._layers:
            if hasattr(layer, "use_flash_attention"):
                layer.use_flash_attention = value  # type: ignore

    @property
    def has_registers(self) -> bool:
        """Whether the encoder has register tokens."""
        return self._num_registers > 0

    @property
    @override
    def in_features(self) -> int | None:
        """Input feature dimension."""
        return self._in_features

    @property
    @override
    def out_features(self) -> int:
        """Output feature dimension."""
        return self._in_features

    @property
    def context_dim(self) -> int:
        """Dimension of the context vector."""
        return self._context_dim

    @property
    def using_context(self) -> bool:
        """Whether the encoder is using context vectors."""
        return self._context_dim > 0
