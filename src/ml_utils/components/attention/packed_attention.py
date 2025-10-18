from dataclasses import replace
from functools import partial
from typing import Literal

import torch as th
from einops import rearrange
from torch import nn
from typing_extensions import override

from ml_utils.components.base import BaseComponent
from ml_utils.torch_utils.types import (
    CulensTensor,
    PackedMHATensor,
    PackedQKVTensor,
    PackedTensor,
)
from ml_utils.utils import exists

from .attention_config import FlashAttentionKWArgs
from .qk_norm import QKNorm
from .qkv_linear import QKVLinear
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
        qkv_bias: bool = False,
        split_qkv: bool = False,
        qk_norm_type: Literal["layer", "rms"] | None = "rms",
        flash_attention_kwargs: FlashAttentionKWArgs | None = None,
        use_flash_attention: bool = True,
    ):
        """Constructor for PackedSelfAttention.

        Args:
            dimension: Dimension of the input and output features.
            nheads: Number of attention heads.
            split_qkv: Whether to use separate linear layers for Q, K, V projections.
                This might be advantageous for Muon optimizer.
            qk_norm_type: Type of normalization to apply to Q and K. Can be "layer",
                "rms", or None. Default is "rms".
            qkv_bias: Whether to include bias in the QKV linear layer.
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
        self._qkv_bias = qkv_bias
        self._qk_norm_type = qk_norm_type
        self._include_qk_norm = exists(qk_norm_type)
        self._train_flash_attention_kwargs = (
            flash_attention_kwargs
            if flash_attention_kwargs is not None
            else FlashAttentionKWArgs()
        )
        self._eval_flash_attention_kwargs = replace(
            self._train_flash_attention_kwargs,
            dropout_p=0.0,  # No dropout during evaluation
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

        # QKNorm only gets initialized if self._qk_norm_type is not None
        # Somehow type checker fails to realize that so disabling the check here
        self._qk_norm = (
            QKNorm(dimension // nheads, norm_type=self._qk_norm_type)  # type: ignore
            if self._include_qk_norm
            else None
        )
        self._qkv_proj = QKVLinear(dimension, bias=qkv_bias, split_qkv=split_qkv)
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
        qkv = self._extract_qkv(x)

        out = self._attention_function(
            qkv,
            cu_seqlens_q=culens,
            max_seqlen_q=max_seqlen,
            flash_attn_kwargs=flash_attention_kwargs,
        )
        out = self._convert_from_headed_layout(out)
        return self._out_proj(out)

    def _extract_qkv(self, x: PackedTensor) -> PackedMHATensor | PackedQKVTensor:
        if self._qkv_proj.split_qkv:
            # Separate projections for Q, K, V
            q, k, v = (self._convert_to_headed_layout(x) for x in self._qkv_proj(x))
            if self._include_qk_norm:
                q, k = self._qk_norm(q, k)
            qkv = q, k, v
        else:
            qkv = self._convert_to_headed_layout(self._qkv_proj(x))
            # Combined projection for Q, K, V
            if self._include_qk_norm:
                q, k, v = th.unbind(qkv, dim=1)
                q, k = self._qk_norm(q, k)
                qkv = q, k, v
        return qkv

    def init_weights(
        self,
        init_qkv_std: float | None = None,
        init_proj_std: float | None = None,
        factor: float = 1.0,
    ):
        """Initialize the weights of the attention layer.

        Args:
            init_qkv_std: Standard deviation for initializing attention weights.
                If None, defaults to 0.02.
            init_proj_std: Standard deviation for initializing projection weights.
                If None, defaults to 0.02.
            factor: Scaling factor for initialization standard deviations.
        """
        init_qkv_std = (
            init_qkv_std
            if init_qkv_std is not None
            else self.in_features**-0.5
        )
        proj_std = (
            init_proj_std
            if init_proj_std is not None
            else init_qkv_std * factor
        )

        self._qkv_proj.init_weights(
            init_std=init_qkv_std,
        )
        nn.init.normal_(self._out_proj.weight, std=proj_std)
        if self._out_proj.bias is not None:
            nn.init.zeros_(self._out_proj.bias)

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
    def in_features(self) -> int:
        return self._dimension

    @override
    @property
    def out_features(self) -> int:
        return self._dimension
