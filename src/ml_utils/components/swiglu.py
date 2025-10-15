import jaxtyping as jt
import torch as th
from torch import nn
from torch.nn.functional import silu
from typing_extensions import override

from .base import BaseComponent


def swish_gated_linear_unit(
    tensor: jt.Float[th.Tensor, "... dim"],
) -> jt.Float[th.Tensor, "... dim // 2"]:
    """SwiGLU activation function.

    Args:
        tensor (jt.Float[th.Tensor, "... dim"]): Input tensor of shape (..., dim)

    Returns:
        jt.Float[th.Tensor, "... dim // 2"]: Output tensor of shape (..., dim // 2)
    """
    x, gate = tensor.chunk(2, dim=-1)
    return x * silu(gate)


class SwiGLU(nn.Module):
    def __init__(self):
        """SwiGLU activation."""
        super().__init__()

    def forward(
        self, tensor: jt.Float[th.Tensor, "... dim"]
    ) -> jt.Float[th.Tensor, "... (dim//2)"]:  # noqa: F821
        return swish_gated_linear_unit(tensor)


class SwiGLUMLP(BaseComponent):
    def __init__(self, dim: int):
        """MLP with SwiGLU activation.

        Args:
            dim (int): Input dimension.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 2),
            SwiGLU(),
            nn.Linear(dim, dim),
        )
        self._dim = dim

    @property
    @override
    def out_features(self) -> int:
        return self._dim

    @property
    @override
    def in_features(self) -> int:
        return self._dim

    def forward(
        self, tensor: jt.Float[th.Tensor, "... dim"]
    ) -> jt.Float[th.Tensor, "... dim"]:
        return self.net(tensor)
