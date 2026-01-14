from typing import Literal

import torch as th
from torch import nn
from typing_extensions import override

from ml_utils.components.base import BaseComponent
from ml_utils.components.swiglu import gated_linear_unit
from ml_utils.components.utils import instantiate_norm_layer
from ml_utils.torch_utils.types import BatchedMatrixTensor
from ml_utils.utils import default

from .utils import TriangleMultiplicationConfig


def prepare_einsum_equation(direction: Literal["outgoing", "incoming"]):
    if direction == "outgoing":
        return "b i k c, b j k c -> b i j c"
    if direction == "incoming":
        return "b k i c, b k j c -> b i j c"
    raise ValueError(f"Invalid direction: {direction}")


class TriangleMultiplication(BaseComponent):
    """A component that performs triangle multiplication on input tensor.

    Args:
        in_features: Number of input features.
        direction: Direction of triangle multiplication,
            either "incoming" or "outgoing".
        config: Configuration for the TriangleMultiplication component.
    """

    def __init__(
        self,
        in_features: int,
        direction: Literal["incoming", "outgoing"] = "incoming",
        config: TriangleMultiplicationConfig | None = None,
    ):
        """Initializes the TriangleMultiplication component.

        Args:
            in_features: Number of input features.
            direction: Direction of triangle multiplication,
                either "incoming" or "outgoing".
            config: Configuration for the TriangleMultiplication component.
        """
        config = default(config, TriangleMultiplicationConfig())
        super().__init__()
        self._in_features = in_features
        self._direction = direction
        self._norm = instantiate_norm_layer(config.norm_type, in_features)

        self._a_proj = nn.Linear(in_features, 2 * in_features, bias=config.use_bias)
        self._b_proj = nn.Linear(in_features, 2 * in_features, bias=config.use_bias)
        self._gate_proj = nn.Sequential(
            nn.Sigmoid(), nn.Linear(in_features, in_features)
        )
        self._triangle_eq = prepare_einsum_equation(direction)

    @property
    @override
    def in_features(self) -> int:
        return self._in_features

    @property
    @override
    def out_features(self) -> int:
        return self._in_features

    def forward(self, x: BatchedMatrixTensor, seq_lens: th.Tensor):
        """Forward pass of the TriangleMultiplication component.

        Args:
            x (BatchedMatrixTensor): Input tensor of shape
                (batch_size, N, N, in_features).
            seq_lens (th.Tensor): Sequence lengths tensor of shape (batch_size,).

        Returns:
            BatchedMatrixTensor: Output tensor of shape (batch_size, N, N, in_features).
        """
        x = self._norm(x)
        a = gated_linear_unit(self._a_proj(x))
        b = gated_linear_unit(self._b_proj(x))
        gate = self._gate_proj(x)

        mask = self.create_mask(seq_lens, x)
        a = self.apply_mask(a, mask)
        b = self.apply_mask(b, mask)
        return gate * th.einsum(self._triangle_eq, a, b)

    @staticmethod
    def create_mask(seq_lens: th.Tensor, x: BatchedMatrixTensor) -> th.Tensor:
        """Creates a mask based on sequence lengths.

        Args:
            seq_lens: Tensor containing sequence lengths.
            x: Input tensor, shape (batch_size, N, N, in_features).

        Returns:
            A boolean mask tensor of shape (batch_size, N, N).
        """
        mask = th.arange(x.size(1), device=seq_lens.device).unsqueeze(
            0
        ) < seq_lens.unsqueeze(1)
        mask = mask.unsqueeze(1) & mask.unsqueeze(2)  # (batch_size, N, N)
        return mask

    @staticmethod
    def apply_mask(x: BatchedMatrixTensor, mask: th.Tensor) -> BatchedMatrixTensor:
        """Applies the mask to the input tensor.

        Args:
            x: Input tensor, shape (batch_size, N, N, in_features).
            mask: Boolean mask tensor of shape (batch_size, N, N).

        Returns:
            Masked input tensor.
        """
        return x * mask.unsqueeze(-1).to(x.dtype)
