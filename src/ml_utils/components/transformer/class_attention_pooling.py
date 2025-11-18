from typing import TypeAlias

import jaxtyping as jt
import torch as th
from torch.nn import RMSNorm
from typing_extensions import override

from ml_utils.components.base import BaseComponent
from ml_utils.torch_utils import unpack_tensor
from ml_utils.torch_utils.types import CulensTensor, PackedTensor
from ml_utils.utils import default, exists

from .encoder import TransformerEncoder, TransformerEncoderConfig
from .encoder_block import TransformerEncoderBlockConfig

BatchedContextTensor: TypeAlias = jt.Float[th.Tensor, " batch_size context_dim"]


class ClassAttentionPooling(BaseComponent):
    """Class Attention Pooling module.

    This module implements a pooling mechanism using class attention via a transformer
    encoder. It projects input features into a latent space, processes them through
    multiple transformer layers, and extracts a class token as the pooled output.

    Args:
        in_features (int): Dimension of the input features.
        out_features (int): Dimension of the output features.
        latent_dim: Latent feature dimension for internal projections.
                If None, defaults to in_features.
        context_dim (int): Dimension of the context vector. If 0, no context is used.
        num_layers (int): Number of transformer encoder layers to use.
        block_config (TransformerEncoderBlockConfig | None): Configuration for the
            transformer encoder blocks. If None, defaults to standard configuration.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        latent_dim: int | None = None,
        context_dim: int = 0,
        num_layers: int = 1,
        block_config: TransformerEncoderBlockConfig | None = None,
    ):
        """Initializes the ClassAttentionPooling module.

        Args:
            in_features (int): Dimension of the input features.
            out_features (int): Dimension of the output features.
            latent_dim: Latent feature dimension for internal projections.
                If None, defaults to in_features.
            context_dim (int): Dimension of the context vector. If 0, no context is used.
            num_layers (int): Number of transformer encoder layers to use.
            block_config (TransformerEncoderBlockConfig | None): Configuration for the
                transformer encoder blocks. If None, defaults to standard configuration.
        """
        super().__init__()
        self._in_features = in_features
        self._out_features = out_features
        self._block_config = default(block_config, TransformerEncoderBlockConfig())
        self._full_encoder_config = TransformerEncoderConfig(
            num_layers=num_layers,
            num_registers=0,
            num_class_tokens=1,
            transformer_config=self._block_config,
        )
        self._context_dim = context_dim
        self._latent_dim = default(latent_dim, in_features)

        self._encoder = TransformerEncoder(
            in_features=self._latent_dim,
            config=self._full_encoder_config,
            context_dim=context_dim,
        )

        self._conditional_layer_setup(
            in_features=in_features,
            latent_dim=latent_dim,
            out_features=out_features,
        )

    def _conditional_layer_setup(
        self,
        in_features: int,
        latent_dim: int | None,
        out_features: int,
    ):
        self._in_projection = (
            th.nn.Linear(in_features=in_features, out_features=latent_dim)
            if exists(latent_dim) and latent_dim != in_features
            else th.nn.Identity()
        )
        self._out_projection = (
            th.nn.Linear(self._latent_dim, out_features=out_features)
            if self._latent_dim != out_features
            else th.nn.Identity()
        )
        self._maybe_out_norm = (
            RMSNorm(self._latent_dim)
            if self._latent_dim != out_features
            else th.nn.Identity()
        )

    @property
    @override
    def out_features(self) -> int:
        return self._out_features

    @property
    @override
    def in_features(self) -> int:
        return self._in_features

    def forward(
        self,
        x: PackedTensor,
        cu_seqlens: CulensTensor,
        max_seqlen: int | None = None,
        *,
        context: BatchedContextTensor | None = None,
    ) -> jt.Float[th.Tensor, " batch_size out_features"]:
        """Forward pass of the ClassAttentionPooling module.

        Args:
            x: Input packed tensor of shape (total_seq_len, in_features).
            cu_seqlens: Cumulative sequence lengths tensor of shape (batch_size + 1).
            max_seqlen: Maximum sequence length in the batch. If None, it will be
                inferred.
            context: Optional context tensor of shape (batch_size, context_dim).

        Returns:
            Pooled output tensor of shape (batch_size, out_features).
        """
        encoded_features, new_cu_seqlens, _ = self._encoder(
            self._in_projection(x),
            cu_seqlens,
            max_seqlen,
            context=context,
        )
        _, unpacked_features = unpack_tensor(new_cu_seqlens, encoded_features)
        class_token = unpacked_features[
            :, 0, ...
        ]  # Class token is first token in sequence
        return self._out_projection(self._maybe_out_norm(class_token))
