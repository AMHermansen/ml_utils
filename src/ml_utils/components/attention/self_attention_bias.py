from collections.abc import Callable
from functools import partial
from typing import TypeAlias

import einx
import torch as th
from torch import nn
from torch.nn.attention.flex_attention import (
    flex_attention,
)
from typing_extensions import override
from einops.layers.torch import Rearrange

from ml_utils.components.base import BaseComponent
from ml_utils.components.utils import instantiate_norm_layer
from ml_utils.torch_utils.types import (
    BatchedMatrixTensor,
)
from ml_utils.utils import default, exists

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

# From a quick benchmark, flex_attention seems to be quite slower than math based
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


class SelfAttentionBias(BaseComponent):
    """Self-Attention with Biases.

    This class implements a self-attention mechanism that uses attention bias.
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
        self._to_headed_layout = Rearrange(
            "b s (h d) -> b h s d", h=self._config.nheads
        )
        self._from_headed_layout = Rearrange(
            "b h s d -> b s (h d)"
        )
        self._to_headed_bias_layout = Rearrange("b s1 s2 h -> b h s1 s2")

        self._compute_scores = partial(th.einsum, "b h i d, b h j d -> b h i j")
        self._compute_out = partial(th.einsum, "b h i j, b h j d -> b h i d")

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
        mask: th.Tensor | None = None,
    ):
        """Forward pass of the SelfAttentionBias.

        Args:
            x: Tensor of shape (batch_size, seq_len, single_features).
            bias: Bias tensor of shape (batch_size, seq_len, seq_len, bias_features).
            mask: Optional attention mask of shape (batch_size, seq_len, seq_len).

        Returns:
            Output tensor.
        """
        norm_x = self._pre_norm(x)
        qkv = self._qkv_proj(norm_x)
        q, k, v = th.chunk(qkv, 3, dim=-1)
        q = self._to_headed_layout(q)
        k = self._to_headed_layout(k)
        v = self._to_headed_layout(v)

        gate_proj = self._gate(norm_x)
        bias_proj = self._bias_proj(self._pre_norm_bias(bias))

        scale = self._config.head_dim ** (-0.5)
        attn_scores = self._compute_scores(q, k) * scale
        attn_scores = attn_scores + self._to_headed_bias_layout(bias_proj)

        if exists(mask):
            attn_scores = einx.where(
                "b j, b h i j, -> b h i j",
                mask, attn_scores, - th.finfo(attn_scores.dtype).max
            )
        attn_weights = th.softmax(attn_scores, dim=-1)
        attn_output_headed = self._compute_out(attn_weights, v)
        attn_output = self._from_headed_layout(attn_output_headed)
        return self._out_proj(attn_output * gate_proj)
