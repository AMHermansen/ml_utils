from collections.abc import Callable
from typing import TypeAlias

import torch as th
from torch import nn
from torch.nn.attention.flex_attention import (
    flex_attention,
)
from typing_extensions import override

from ml_utils.components.base import BaseComponent
from ml_utils.components.utils import instantiate_norm_layer
from ml_utils.torch_utils.types import (
    BatchedMatrixTensor,
)
from ml_utils.utils import default

from ._utils import (
    create_score_mod,
)
from .attention_config import BiasAttentionConfig

ScoreModSignature: TypeAlias = Callable[
    [th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor], th.Tensor
]
MaskFuncSignature: TypeAlias = Callable[
    [th.Tensor, th.Tensor, th.Tensor, th.Tensor], th.Tensor
]


class PackedSelfAttentionBias(BaseComponent):
    """Packed Self-Attention with Biases.

    This class implements a packed self-attention mechanism that uses attention bias.
    As such it processes both input features and bias features to compute attention
    scores.

    Args:
        in_features: Number of input features.
        bias_features: Number of bias features.
        config: Configuration for the attention mechanism.

    """

    def __init__(
        self,
        in_features: int,
        bias_features: int,
        config: BiasAttentionConfig | None = None,
    ):
        """Initialize PackedSelfAttentionBias.

        Args:
            in_features: Number of input features.
            bias_features: Number of bias features.
            config: Configuration for the attention mechanism.
        """
        super().__init__()
        config = default(config, BiasAttentionConfig())

        self._in_features = in_features
        self._bias_features = bias_features
        self._config = config

        total_head_dim = self._config.nheads * self._config.head_dim

        self._qkv_proj = nn.Linear(
            self._in_features,
            3 * total_head_dim,
            bias=self._config.qkv_bias,
        )
        self._out_proj = nn.Linear(
            total_head_dim,
            self._in_features,
            bias=self._config.out_bias,
        )
        self._gate = nn.Sequential(
            nn.Linear(self._in_features, total_head_dim),
            nn.Sigmoid(),
        )
        self._bias_proj = nn.Linear(
            self._bias_features,
            self._config.nheads,
            bias=self._config.bias_bias,
        )

        self._pre_norm = instantiate_norm_layer(
            self._config.pre_norm_type, self._in_features
        )
        self._pre_norm_bias = instantiate_norm_layer(
            self._config.pre_norm_bias_type, self._bias_features
        )

    @property
    @override
    def in_features(self) -> int:
        return self._in_features

    @property
    @override
    def out_features(self) -> int:
        return self._in_features

    def forward(
        self,
        x: th.Tensor,
        bias: BatchedMatrixTensor,
    ):
        """Forward pass of the PackedSelfAttentionBias.

        Args:
            x: Jagged tensor.
            bias: Bias tensor of shape (batch_size, seq_len, seq_len, bias_features).

        Returns:
            Output tensor.
        """
        norm_x = self._pre_norm(x)
        qkv = self._qkv_proj(norm_x)
        q, k, v = th.chunk(qkv, 3, dim=-1)
        q = q.unflatten(-1, (self._config.nheads, self._config.head_dim)).transpose(
            1, 2
        )
        k = k.unflatten(-1, (self._config.nheads, self._config.head_dim)).transpose(
            1, 2
        )
        v = v.unflatten(-1, (self._config.nheads, self._config.head_dim)).transpose(
            1, 2
        )
        gate_proj = self._gate(norm_x)

        bias_proj = self._bias_proj(self._pre_norm_bias(bias))
        score_mod = create_score_mod(bias_proj)
        attn_output = flex_attention(
            q,
            k,
            v,
            score_mod=score_mod,
        )
        assert isinstance(attn_output, th.Tensor)
        attn_output = attn_output.transpose(1, 2).flatten(-2)
        attn_output = attn_output * gate_proj
        return self._out_proj(attn_output)
