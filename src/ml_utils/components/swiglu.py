from typing import Literal

import jaxtyping as jt
import torch as th
from torch import nn
from torch.nn.functional import silu
from typing_extensions import override

from .base import BaseComponent


def swish_gated_linear_unit(
    tensor: jt.Float[th.Tensor, "... in_features"],
    inner_factor: float = 1.0,
    outer_factor: float = 1.0,
) -> jt.Float[th.Tensor, "... in_features // 2"]:
    """SwiGLU activation function.

    Applies the SwiGLU activation function to the input tensor.
    The input tensor is split into two halves along the last dimension.

    If the scaling factors are equal it corresponds to changing the multiplicative
    factor inside sigmoid alone of silu. If they are different it corresponds to a more
    general form of SwiGLU.

    Args:
        tensor (jt.Float[th.Tensor, "... in_features"]): Input tensor of shape (..., in_features)
        inner_factor (float, optional): Scaling factor applied inside silu. Defaults to 1.0.
        outer_factor (float, optional): Scaling factor applied outside silu. Defaults to 1.0.

    Returns:
        jt.Float[th.Tensor, "... in_features // 2"]: Output tensor of shape (..., in_features // 2)
    """
    x, gate = tensor.chunk(2, dim=-1)
    return x * silu(inner_factor * gate) / outer_factor


class SwiGLU(nn.Module):
    def __init__(self, mode: Literal["swish", "mp", "gelu", "silu"] = "swish"):
        """SwiGLU activation.

        Args:
            mode: Literal["swish", "mp", "gelu", "silu"]: Mode of SwiGLU activation.
                If "swish", uses standard SwiGLU.
                If "mp", uses the Magnitude-Preserving SwiGLU variant from EDM2.
                If "gelu", uses the GeLU approximation using swish.
        """
        super().__init__()
        if mode in {"swish", "silu"}:
            self.inner_factor, self.outer_factor = 1., 1.
        elif mode == "mp":
            self.inner_factor, self.outer_factor = 1.0, 0.596
            # See https://arxiv.org/pdf/2312.02696 Eq: 80
        elif mode == "gelu":
            self.inner_factor, self.outer_factor = 1.702, 1.702
            # See https://arxiv.org/pdf/1606.08415 Section 2
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def forward(
        self, tensor: jt.Float[th.Tensor, "... in_features"]
    ) -> jt.Float[th.Tensor, "... (in_features//2)"]:  # noqa: F821
        return swish_gated_linear_unit(
            tensor,
            inner_factor=self.inner_factor,
            outer_factor=self.outer_factor
        )


class SwiGLUMLP(BaseComponent):
    def __init__(
        self,
        in_features: int,
        mode: Literal["swish", "mp", "gelu", "silu"] = "swish",
    ):
        """MLP with SwiGLU activation.

        Args:
            in_features (int): Number of input features.
            mode: Literal["swish", "mp", "gelu", "silu"]: Mode of SwiGLU activation.
                If "swish"/"silu", uses standard SwiGLU.
                If "mp", uses the Magnitude-Preserving SwiGLU variant from EDM2.
                If "gelu", uses the GeLU approximation using swish.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, in_features * 2),
            SwiGLU(mode=mode),
            nn.Linear(in_features, in_features),
        )
        self._features = in_features
        self._mode = mode

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
