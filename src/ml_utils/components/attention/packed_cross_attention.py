from dataclasses import replace
from functools import partial

from torch import nn

from ml_utils.components.attention._utils import (
    convert_from_headed_layout,
    convert_to_headed_and_kvmerged_layout,
    convert_to_headed_layout,
)
from ml_utils.torch_utils.types import (
    AllPackedQKVTypes,
    CulensTensor,
    PackedTensor,
)
from ml_utils.utils import exists

from .attention_config import CrossAttentionConfig, FlashAttentionKWArgs
from .qk_norm import QKNorm
from .torch_flash_interface import torch_flash_attention_interface

# Import flash attention can fail if no CUDA
try:
    from ._flash import (
        common_flash_attention_interface,
    )
except ImportError:
    common_flash_attention_interface = torch_flash_attention_interface


class PackedCrossAttention(nn.Module):
    def __init__(
        self,
        dimension: int,
        q_in_dim: int,
        config: CrossAttentionConfig | None = None,
    ):
        """Packed Cross-Attention module supporting flash attention and QK normalization.

        Args:
            dimension: The embedding dimension of the attention.
            q_in_dim: The input dimension for the query.
            config: Configuration for the cross-attention module.
        """
        config = config if exists(config) else CrossAttentionConfig()
        super().__init__()
        if dimension % config.nheads != 0:
            raise ValueError("dimension must be divisible by nheads")

        self._config = config
        self._nheads = config.nheads
        self._dimension = dimension
        self._head_dim = dimension // config.nheads
        self._q_in_dim = q_in_dim
        self._kv_in_dim = config.kv_in_dim if exists(config.kv_in_dim) else q_in_dim
        self._q_bias = config.q_bias
        self._kv_bias = config.kv_bias
        self._use_qk_norm = exists(config.qk_norm_type)
        self._qk_norm_type = config.qk_norm_type
        self._use_flash_attention = config.use_flash_attention

        self._attention_function = (
            common_flash_attention_interface
            if config.use_flash_attention
            else torch_flash_attention_interface
        )

        # Store train/eval flash attention kwargs for convenience
        self._train_flash_attention_kwargs = (
            config.flash_attention_kwargs
            if exists(config.flash_attention_kwargs)
            else FlashAttentionKWArgs()
        )
        self._eval_flash_attention_kwargs = replace(
            self._train_flash_attention_kwargs,
            dropout_p=0.0,  # No dropout during evaluation
        )

        # Utilities to convert to and from headed layouts
        self._convert_to_headed_and_kvmerged_layout = partial(
            convert_to_headed_and_kvmerged_layout,
            nheads=config.nheads,
        )
        self._convert_to_headed_layout = partial(
            convert_to_headed_layout,
            nheads=config.nheads,
        )
        self._convert_from_headed_layout = partial(
            convert_from_headed_layout,
            nheads=config.nheads,
        )

        # QKNorm only gets initialized if self._qk_norm_type is not None
        # Somehow type checker fails to realize that so disabling the check here
        self._qk_norm = (
            QKNorm(dimension // config.nheads, norm_type=self._qk_norm_type)  # type: ignore
            if self._include_qk_norm
            else None
        )

        # Projection layers
        self._q_proj = nn.Linear(
            in_features=self._q_in_dim,
            out_features=dimension,
            bias=config.q_bias,
        )
        self._kv_proj = nn.Linear(
            in_features=self._kv_in_dim,
            out_features=2 * dimension,
            bias=config.kv_bias,
        )

        self._out_proj = nn.Linear(dimension, dimension)

    def forward(
        self,
        q_sequence: PackedTensor,
        kv_sequence: PackedTensor,
        cu_seqlens_q: CulensTensor,
        cu_seqlens_kv: CulensTensor,
        max_seqlen_q: int | None = None,
        max_seqlen_kv: int | None = None,
    ):
        flash_attention_kwargs = (
            self._train_flash_attention_kwargs
            if self.training
            else self._eval_flash_attention_kwargs
        )
        q_embedded = self._q_proj(q_sequence)
        kv_embedded = self._kv_proj(kv_sequence)

        qkv = self._maybe_apply_qknorm(
            q_embedded=q_embedded,
            kv_embedded=kv_embedded,
        )

        attn_out = self._attention_function(
            qkv=qkv,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_kv,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_kv,
            flash_attn_kwargs=flash_attention_kwargs,
        )

        attn_out_unheaded = self._convert_from_headed_layout(attn_out)
        return self._out_proj(attn_out_unheaded)

    def _maybe_apply_qknorm(
        self,
        q_embedded: PackedTensor,
        kv_embedded: PackedTensor
    ) -> AllPackedQKVTypes:
        """Applies QKNorm if enabled. Otherwise, just converts KV to headed layout.

        Args:
            q_embedded: The query tensor in headed layout.
            kv_embedded: The key-value tensor in embedded layout.

        Returns:
            A tuple of (q_headed, k_headed, v_headed) if QKNorm is applied,
            otherwise (q_headed, kv_headed) with KV merged.
        """
        # We have two cases: with or without QKNorm
        # If no QKNorm, we can keep KV merged for efficiency.
        q_headed = self._convert_to_headed_layout(q_embedded)
        if self._include_qk_norm:
            k_embedded, v_embedded = kv_embedded.chunk(2, dim=-1)
            k_headed = self._convert_to_headed_layout(k_embedded)
            v_headed = self._convert_to_headed_layout(v_embedded)
            q_headed, k_headed = self._qk_norm(q_headed, k_headed)
            return q_headed, k_headed, v_headed
        kv_headed = self._convert_to_headed_and_kvmerged_layout(kv_embedded)
        return q_headed, kv_headed

    @property
    def use_flash_attention(self) -> bool:
        """Get whether flash attention is used."""
        return self._use_flash_attention

    @use_flash_attention.setter
    def use_flash_attention(self, value: bool) -> None:
        """Set whether to use flash attention or standard attention."""
        self._use_flash_attention = value
        self._attention_function = (
            common_flash_attention_interface
            if value
            else torch_flash_attention_interface
        )

    @property
    def in_features(self) -> int:
        return self._q_in_dim

    @property
    def kv_in_features(self) -> int:
        return self._kv_in_dim

    @property
    def out_features(self) -> int:
        return self._dimension
