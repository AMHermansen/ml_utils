from typing import Literal

import torch as th
from torch import nn
from torch.nn.attention.flex_attention import (
    BlockMask,
    create_block_mask,
    flex_attention,
)

from ml_utils.components.attention._utils import create_score_mod
from ml_utils.components.utils import instantiate_norm_layer
from ml_utils.torch_utils.types import BatchedMatrixTensor
from ml_utils.utils import default

from .utils import TriangleAttentionConfig


def flatten_function(x: BatchedMatrixTensor) -> th.Tensor:
    batch_size, seq_length, seq_length, num_features = x.shape
    return x.reshape(batch_size * seq_length, seq_length, num_features)


def unflatten_function(
    x: th.Tensor, batch_size: int, seq_length: int
) -> BatchedMatrixTensor:
    return x.reshape(batch_size, seq_length, seq_length, -1)


def to_headed_shape(x: th.Tensor, num_heads: int) -> th.Tensor:
    batch_size, seq_length, num_features = x.shape
    head_dim = num_features // num_heads
    return x.unflatten(-1, (num_heads, head_dim)).transpose(1, 2)  # (B, H, N, D)


def from_headed_shape(x: th.Tensor) -> th.Tensor:
    return x.transpose(1, 2).flatten(-2)  # (B, N, C)


class TriangleAttention(nn.Module):
    """Triangle Attention module as described in AlphaFold3.

    Args:
        in_features: Number of input (pair) features
        config: Configuration for Triangle Attention
    """

    def __init__(
        self,
        in_features: int,
        direction: Literal["starting", "ending"] = "starting",
        config: TriangleAttentionConfig | None = None,
    ):
        """Instantiates Triangle Attention module.

        Args:
            in_features: Number of input (pair) features
            direction: Direction of attention ("starting" or "ending" node).
                Follows naming from AlphaFold3.
            config: Configuration for Triangle Attention
        """
        config = default(config, TriangleAttentionConfig())
        self._direction = direction
        super().__init__()
        self._in_features = in_features
        self._config = config

        self._norm = instantiate_norm_layer(config.norm_type, in_features)
        self._qkv_proj = nn.Linear(in_features, 3 * in_features, bias=config.use_bias)
        self._out_proj = nn.Linear(in_features, in_features, bias=config.use_bias)

        self._bias_proj = nn.Linear(in_features, config.num_heads, bias=config.use_bias)
        self._gate = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.Sigmoid(),
        )

        self._flatten_func = flatten_function
        self._unflatten_func = unflatten_function

        self._to_headed_shape = to_headed_shape
        self._from_headed_shape = from_headed_shape

    # Properties
    @property
    def should_flip_coordinates(self) -> bool:
        return self._direction == "ending"

    @property
    def use_flex_attention(self) -> bool:
        # Somewhat surprisingly flex attention seems to be slower across various sizes.
        # Likely due to overheads in re-shuffling dimensions.
        return self._config.use_flex_attention

    def forward(
        self,
        x: BatchedMatrixTensor,
        seq_lens: th.Tensor,
    ):
        """Forward pass of TriangleAttention.

        Args:
            x: BatchedMatrixTensor: Input tensor of shape (B, N, N, C)
            seq_lens: th.Tensor: Sequence lengths tensor of shape (B,)

        Returns:
            BatchedMatrixTensor: Output tensor of shape (B, N, N, C)
        """
        if self._config.use_flex_attention:
            return self._flex_attention_forward(x, seq_lens)
        return self._math_attention_forward(x, seq_lens)

    def _flex_attention_forward(
        self, x: BatchedMatrixTensor, seq_lens: th.Tensor
    ) -> BatchedMatrixTensor:
        """Flex-attention forward pass.

        Args:
            x: BatchedMatrixTensor: Input tensor of shape (B, N, N, C)
            seq_lens: th.Tensor: Sequence lengths tensor of shape (B,)

        Returns:
            BatchedMatrixTensor: Output tensor of shape (B, N, N, C)
        """
        x = self._norm(x)
        batch_size, max_seq_len, *_ = x.shape
        if self.should_flip_coordinates:
            x = x.transpose(1, 2)

        q, k, v = th.chunk(self._qkv_proj(x), 3, dim=-1)
        bias_proj = (
            self._bias_proj(x)
            .unsqueeze(1)
            .repeat(1, max_seq_len, 1, 1, 1)
            .reshape(
                batch_size * max_seq_len,
                max_seq_len,
                max_seq_len,
                self._config.num_heads,
            )
        )  # (B*N, N, N, H)
        score_mod = create_score_mod(bias_proj)
        gate = self._gate(x)

        q_flat = self._to_headed_shape(
            self._flatten_func(q), self._config.num_heads
        )  # (B*N, H, N, C)
        k_flat = self._to_headed_shape(self._flatten_func(k), self._config.num_heads)
        v_flat = self._to_headed_shape(self._flatten_func(v), self._config.num_heads)

        mask = self._construct_padding_mask(seq_lens, batch_size, max_seq_len)
        attn_output = flex_attention(
            q_flat,
            k_flat,
            v_flat,
            score_mod=score_mod,
            block_mask=mask,
        )
        assert isinstance(attn_output, th.Tensor)
        attn_output_normal = self._unflatten_func(
            self._from_headed_shape(
                attn_output,
            ),
            batch_size,
            max_seq_len,
        )
        output = self._out_proj(attn_output_normal * gate)
        if self.should_flip_coordinates:
            output = output.transpose(1, 2)
        return output

    # Attention implementations
    def _math_attention_forward(
        self,
        x: BatchedMatrixTensor,
        seq_lens: th.Tensor,
    ):
        """Alternative forward pass using mathematical definition."""
        x_norm = self._norm(x)
        _batch_size, max_seq_len, *_ = x_norm.shape

        q, k, v = th.chunk(self._qkv_proj(x_norm), 3, dim=-1)
        bias = self._bias_proj(x_norm)
        bias = bias.permute(0, 3, 1, 2)  # (B, H, N, N)
        gate = self._gate(x_norm)

        q_h = q.unflatten(-1, (self._config.num_heads, -1)).permute(
            0, 3, 1, 2, 4
        )  # (B, H, N, N, D)
        k_h = k.unflatten(-1, (self._config.num_heads, -1)).permute(
            0, 3, 1, 2, 4
        )  # (B, H, N, N, D)
        v_h = v.unflatten(-1, (self._config.num_heads, -1)).permute(
            0, 3, 1, 2, 4
        )  # (B, H, N, N, D)

        score_transformer_string = (
            "bhijc,bhikc->bhijk"
            if not self.should_flip_coordinates
            else "bhijc,bhkjc->bhijk"
        )
        attn_out_string = (
            "bhijk,bhikc->bhijc"
            if not self.should_flip_coordinates
            else "bhijk,bhkjc->bhijc"
        )
        # Adjust broadcasting dimension of bias based on direction.
        # "Ending" broadcast over "j" (unsqueeze dim 3) and swap from "ik" to "ki".
        # "Starting" broadcast over "i" (unsqueeze dim 2)
        bias = (
            bias.unsqueeze(3).transpose(2, 4)
            if self.should_flip_coordinates
            else bias.unsqueeze(2)
        )

        scores = th.einsum(
            score_transformer_string,
            q_h,
            k_h,
        ) / th.sqrt(th.tensor(q_h.shape[-1], device=x_norm.device, dtype=x_norm.dtype))
        scores_with_bias = scores + bias
        mask = (
            (
                th.arange(
                    max_seq_len, device=x_norm.device, dtype=x_norm.dtype
                ).unsqueeze(0)
                >= seq_lens.unsqueeze(1)
            )
            .unsqueeze(1)
            .unsqueeze(1)
            .unsqueeze(1)
            .repeat(1, self._config.num_heads, max_seq_len, max_seq_len, 1)
        )
        scores_with_bias = scores_with_bias.masked_fill(mask, float("-inf"))
        attn_weights = th.softmax(scores_with_bias, dim=-1)
        attn_output = th.einsum(
            attn_out_string,
            attn_weights,
            v_h,
        )  # (B, H, N, N, D)
        attn_output_normal = attn_output.permute(0, 2, 3, 1, 4).flatten(-2)
        return self._out_proj(attn_output_normal * gate)

    # Private utility methods.
    def _construct_padding_mask(
        self,
        seq_lens: th.Tensor,
        batch_size: int,
        max_seq_len: int,
    ) -> BlockMask:
        """Constructs a padding mask for variable-length sequences.

        Args:
            seq_lens: th.Tensor: Sequence lengths tensor of shape (B,)
            batch_size: int: Batch size
            max_seq_len: int: Maximum sequence length

        Returns:
            BlockMask: Padding mask for attention mechanism
        """
        extended_seq_lens = seq_lens.repeat_interleave(max_seq_len)

        def mask_modifier(
            batch_idx: th.Tensor,
            _head_idx: th.Tensor,
            _query_idx: th.Tensor,
            kv_idx: th.Tensor,
        ) -> th.Tensor:
            length = extended_seq_lens[batch_idx]
            return kv_idx < length

        return create_block_mask(
            mask_modifier,
            batch_size * max_seq_len,
            self._config.num_heads,
            max_seq_len,
            max_seq_len,
            device=seq_lens.device,
        )
