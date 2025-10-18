from typing import Literal

import jaxtyping as jt
import torch as th
from torch import nn


class QKNorm(nn.Module):
    def __init__(self, dim: int, norm_type: Literal["layer", "rms"] = "rms") -> None:
        """QK Normalization Module.

        Args:
            dim: Dimension of the input tensors.
            norm_type: Type of normalization to apply ("layer" or "rms").
        """
        super().__init__()
        self._type = norm_type

        if norm_type == "layer":
            self.q_norm = nn.LayerNorm(dim)
            self.k_norm = nn.LayerNorm(dim)
        elif norm_type == "rms":
            self.q_norm = nn.RMSNorm(dim)
            self.k_norm = nn.LayerNorm(dim)
        else:
            raise ValueError(f"Unsupported normalization type: {norm_type}")

    def forward(
        self,
        query: jt.Float[th.Tensor, "*batch dim"],
        key: jt.Float[th.Tensor, "*batch dim"]) -> tuple[
        jt.Float[th.Tensor, "*batch dim"],
        jt.Float[th.Tensor, "*batch dim"]
    ]:
        """Forward pass for QK normalization.

        Args:
            query: Input query tensor.
            key: Input key tensor.

        Returns:
            A tuple containing the normalized query and key tensors.
        """
        normed_query = self.q_norm(query)
        normed_key = self.k_norm(key)
        return normed_query, normed_key
