import jaxtyping as jt
import torch as th
from torch import nn
from torch.nn.functional import silu
from typing_extensions import override

from .base import BaseComponent


def swish_gated_linear_unit(
    tensor: jt.Float[th.Tensor, "... in_features"],
) -> jt.Float[th.Tensor, "... in_features // 2"]:
    """SwiGLU activation function.

    Args:
        tensor (jt.Float[th.Tensor, "... in_features"]): Input tensor of shape (..., in_features)

    Returns:
        jt.Float[th.Tensor, "... in_features // 2"]: Output tensor of shape (..., in_features // 2)
    """
    x, gate = tensor.chunk(2, dim=-1)
    return x * silu(gate)


class SwiGLU(nn.Module):
    def __init__(self):
        """SwiGLU activation."""
        super().__init__()

    def forward(
        self, tensor: jt.Float[th.Tensor, "... in_features"]
    ) -> jt.Float[th.Tensor, "... (in_features//2)"]:  # noqa: F821
        return swish_gated_linear_unit(tensor)


class SwiGLUMLP(BaseComponent):
    def __init__(self, in_features: int):
        """MLP with SwiGLU activation.

        Args:
            in_features (int): Number of input features.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, in_features * 2),
            SwiGLU(),
            nn.Linear(in_features, in_features),
        )
        self._features = in_features

    @property
    @override
    def out_features(self) -> int:
        return self._features

    @property
    @override
    def in_features(self) -> int:
        return self._features

    def forward(
        self, tensor: jt.Float[th.Tensor, "... in_features"]
    ) -> jt.Float[th.Tensor, "... in_features"]:
        return self.net(tensor)
