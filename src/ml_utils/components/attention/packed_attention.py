from dataclasses import replace
from functools import partial

from einops import rearrange
from torch import nn
from typing_extensions import override

from ml_utils.components.base import BaseComponent
from ml_utils.torch_utils.types import CulensTensor, PackedTensor

from .attention_config import FlashAttentionKWArgs
from .torch_flash_interface import torch_flash_attention_interface

# Import flash attention can fail if no CUDA
try:
    from ._flash import (
        common_flash_attention_interface,
    )
except ImportError:
    common_flash_attention_interface = torch_flash_attention_interface


class PackedSelfAttention(BaseComponent):
    """Multi-head self-attention layer for packed sequences.

    Args:
        dimension: Dimension of the input and output features.
        nheads: Number of attention heads.
        flash_attention_kwargs: Additional keyword arguments for flash attention.
        use_flash_attention: Whether to use flash attention or standard attention.
    """

    def __init__(
        self,
        dimension: int,
        nheads: int,
        flash_attention_kwargs: FlashAttentionKWArgs | None = None,
        use_flash_attention: bool = True,
    ):
        """Constructor for PackedSelfAttention.

        Args:
            dimension: Dimension of the input and output features.
            nheads: Number of attention heads.
            flash_attention_kwargs: Additional keyword arguments for flash attention.
            use_flash_attention: Whether to use flash attention or standard attention.

        Raises:
            ValueError: If dimension is not divisible by nheads.
        """
        super().__init__()
        if dimension % nheads != 0:
            raise ValueError("dimension must be divisible by nheads")
        self._nheads = nheads
        self._dimension = dimension
        self._train_flash_attention_kwargs = (
            flash_attention_kwargs
            if flash_attention_kwargs is not None
            else FlashAttentionKWArgs()
        )
        self._eval_flash_attention_kwargs = replace(
            self._train_flash_attention_kwargs, dropout_p=0.0
        )
        self._use_flash_attention = use_flash_attention
        self._convert_to_headed_layout = partial(
            rearrange,
            pattern="tot_len (n_merge nheads dim) -> tot_len n_merge nheads dim",
            nheads=nheads,
            n_merge=3,  # for QKV
        )
        self._convert_from_headed_layout = partial(
            rearrange,
            pattern="packed_length nheads dim -> packed_length (nheads dim)",
            nheads=nheads,
        )
        self._attention_function = (
            common_flash_attention_interface
            if use_flash_attention
            else torch_flash_attention_interface
        )
        self._to_qkv = nn.Linear(dimension, dimension * 3, bias=False)
        self._out_proj = nn.Linear(dimension, dimension)

    def forward(
        self, x: PackedTensor, culens: CulensTensor, max_seqlen: int | None
    ) -> PackedTensor:
        """Forward pass of the packed self-attention layer.

        Args:
            x: Packed input tensor of shape (packed_length, dimension)
            culens: Cumulative lengths tensor of shape (batch_size + 1)
            max_seqlen: Maximum sequence length in the batch. If None, it will be
                inferred from `culens`.

        Returns:
            Packed output tensor of shape (packed_length, dimension)
        """
        flash_attention_kwargs = (
            self._train_flash_attention_kwargs
            if self.training
            else self._eval_flash_attention_kwargs
        )
        qkv = self._convert_to_headed_layout(self._to_qkv(x))
        out = self._attention_function(
            qkv,
            cu_seqlens_q=culens,
            max_seqlen_q=max_seqlen,
            flash_attn_kwargs=flash_attention_kwargs,
        )
        out = self._convert_from_headed_layout(out)
        return self._out_proj(out)

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

    @override
    @property
    def in_dim(self) -> int:
        return self._dimension

    @override
    @property
    def out_dim(self) -> int:
        return self._dimension
