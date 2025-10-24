import math
from typing import Literal

import jaxtyping as jt
import torch as th
from typing_extensions import override

from .base import BaseComponent


class FourierEmbedding(BaseComponent):
    """Fourier feature embedding module.
    Projects input coordinates into a higher-dimensional space using random Fourier features.
    """

    def __init__(self, num_frequencies: int):
        """Fourier feature embedding.

        Args:
            num_frequencies: Number of Fourier features to use.
        """
        super().__init__()
        self.num_frequencies = num_frequencies
        self.register_buffer("freqs", 2 * math.pi * th.randn(num_frequencies))
        self.register_buffer("phases", 2 * math.pi * th.rand(num_frequencies))

    @property
    @override
    def out_features(self) -> int:
        return self.num_frequencies

    @property
    @override
    def in_features(self) -> int | None:
        return None

    def forward(
        self, x: jt.Float[th.Tensor, "..."]
    ) -> jt.Float[th.Tensor, "... num_frequencies"]:
        """Embed input coordinates with Fourier features.

        Args:
            x: (..., 1) tensor of input coordinates
        Returns:
            (..., N) tensor of Fourier features
        """
        return (th.einsum("..., ... d -> ... d", x, self.freqs) + self.phases).cos().to(  # type: ignore
            x.dtype
        ) + math.sqrt(2)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(num_frequencies={self.num_frequencies})"


class CosineEmbedding(BaseComponent):
    def __init__(
        self,
        out_dim: int,
        scheme: Literal[
            "linear", "lin", "exponential", "exp", "power", "pow"
        ] = "linear",
        min_value: float = 0.0,
        max_value: float = 1.0,
        do_sin: bool = False,
    ):
        """Cosine embedding module.
        Projects input coordinates into a higher-dimensional space using cosine functions.

        Args:
            out_dim: Output dimension of the embedding.
            scheme: Scheme for frequency selection, either "linear", "exponential",
                "power".
            min_value: Minimum value of the input coordinates.
            max_value: Maximum value of the input coordinates.
            do_sin: Whether to include sine functions in the embedding.
        """
        super().__init__()
        self.register_buffer("min_value", th.tensor(min_value))
        self.register_buffer("max_value", th.tensor(max_value))
        self.register_buffer("range", self.max_value - self.min_value)  # type: ignore

        if do_sin and out_dim % 2 != 0:
            raise ValueError("out_dim must be even when do_sin is True.")

        freq_dim = out_dim // 2 if do_sin else out_dim
        self._freq_dim = freq_dim

        freqs = th.arange(freq_dim).float()
        if scheme in {"exp", "exponential"}:
            freqs = th.exp(freqs)
        elif scheme in {"pow", "power"}:
            freqs = 2**freqs
        elif scheme in {"lin", "linear"}:
            freqs += 1
        else:
            raise ValueError(f"Unrecognised frequency scaling: {scheme}")
        self.register_buffer("freqs", freqs)
        self.do_sin = do_sin

    @property
    @override
    def out_features(self) -> int:
        return self._freq_dim * (2 if self.do_sin else 1)

    @property
    @override
    def in_features(self) -> int | None:
        return None

    def _check_bounds(self, x: jt.Float[th.Tensor, "..."], epsilon=1e-6):
        assert th.all(
            x >= self.min_value - epsilon  # type: ignore
        ), f"Input {x} below min value {self.min_value}"
        assert th.all(
            x <= self.max_value + epsilon  # type: ignore
        ), f"Input {x} above max value {self.max_value}"

    def forward(
        self, x: jt.Float[th.Tensor, "..."]
    ) -> jt.Float[th.Tensor, "... out_dim"]:
        """Embed input coordinates with cosine features.

        Args:
            x: (..., 1) tensor of input coordinates
        Returns:
            (..., out_dim) tensor of cosine features
        """
        self._check_bounds(x)
        # Normalize to [0, 1]
        x = (x - self.min_value) / self.range  # type: ignore  
        x = th.einsum("..., d -> ... d", x, self.freqs)  # (..., D)

        if self.do_sin:
            return th.cat([x.cos(), x.sin()], dim=-1).to(x.dtype)
        return x.cos().to(x.dtype)

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(out_dim={self.out_features})"

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(out_dim={self.out_features}, "
            f"min_value={self.min_value.item()}, max_value={self.max_value.item()}, "  # type: ignore
            f"do_sin={self.do_sin})"
        )
