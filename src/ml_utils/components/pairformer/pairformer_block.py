from typing import Literal

import torch as nn
import torch as th
from torch import nn

from ml_utils.components.attention.self_attention_bias import SelfAttentionBias
from ml_utils.components.mlp import MLP
from ml_utils.torch_utils.types import BatchedMatrixTensor
from ml_utils.utils import default, exists

from .triangle_attention import TriangleAttention
from .triangle_multiplication import TriangleMultiplication
from .utils import PairFormerBlockConfig


class Dropout(nn.Module):
    """Row and column dropout module for the PairFormer architecture.

    Args:
        probability: Dropout probability.
        axis: Axis along which to apply dropout. "row" or "column". If None, reverts to
            standard dropout.

    """

    def __init__(
        self,
        probability: float = 0.25,
        *,
        axis: Literal["row", "column"] | None = None,
    ):
        super().__init__()
        self._dropout = nn.Dropout(probability)
        self._axis = axis

    def forward(
        self,
        x: th.Tensor,
    ) -> th.Tensor:
        self._verify_valid_axis(x)

        if not exists(self._axis):
            # Standard dropout. We are not dropping entire rows or columns.
            return self._dropout(x)

        ones = self._construct_ones_tensor(x)
        return x * self._dropout(ones)

    def _verify_valid_axis(self, x: th.Tensor):
        if self._axis in {"row", "column"} and x.ndim != 4:
            raise ValueError(
                f"Input tensor must be 4-dimensional when dropping rows or columns, but got {x.ndim} dimensions."
            )

    def _construct_ones_tensor(self, x: th.Tensor) -> th.Tensor:
        batch_size, n_rows, n_cols, features = x.shape

        if self._axis == "row":
            return th.ones((batch_size, 1, n_cols, features), device=x.device)

        if self._axis == "column":
            return th.ones((batch_size, n_rows, 1, features), device=x.device)

        raise ValueError(f"Invalid axis: {self._axis}")


class PairFormerBlock(nn.Module):
    """Implements a single block of the PairFormer architecture.

    Args:
        single_features: Number of features in the single representations.
        pair_features: Number of features in the pair representations.
        config: Configuration for the PairFormer block. If None, default configuration is used.
    """

    def __init__(
        self,
        single_features: int,
        pair_features: int,
        config: PairFormerBlockConfig | None = None,
    ):
        """Initializes the PairFormer block.

        Args:
            single_features: Number of features in the single representations.
            pair_features: Number of features in the pair representations.
            config: Configuration for the PairFormer block. If None, default configuration is used.
        """
        config = default(config, PairFormerBlockConfig())
        super().__init__()
        self._single_features = single_features
        self._pair_features = pair_features
        self._config = config

        self.triangle_multiplication_outgoing = TriangleMultiplication(
            in_features=pair_features,
            direction="outgoing",
            config=config.triangle_multiplication_config,
        )
        self.triangle_multiplication_incoming = TriangleMultiplication(
            in_features=pair_features,
            direction="incoming",
            config=config.triangle_multiplication_config,
        )
        self.triangle_attention_starting = TriangleAttention(
            in_features=pair_features,
            direction="starting",
            config=config.triangle_attention_config,
        )
        self.triangle_attention_ending = TriangleAttention(
            in_features=pair_features,
            direction="ending",
            config=config.triangle_attention_config,
        )
        self.pair_mlp = MLP(
            in_features=pair_features,
            out_features=pair_features,
            config=config.pair_mlp_config,
        )
        self.dropout_row1 = Dropout(
            probability=config.dropout_probability,
            axis="row",
        )
        self.dropout_row2 = Dropout(
            probability=config.dropout_probability,
            axis="row",
        )
        self.dropout_row3 = Dropout(
            probability=config.dropout_probability,
            axis="row",
        )
        self.dropout_col1 = Dropout(
            probability=config.dropout_probability,
            axis="column",
        )

        self.single_attention = SelfAttentionBias(
            in_features=single_features,
            bias_features=pair_features,
            config=config.single_attention_config,
        )
        self.single_mlp = MLP(
            in_features=single_features,
            out_features=single_features,
            config=config.single_mlp_config,
        )
        if self._config.use_pre_mlp_norm:
            self._pre_mlp_norm = nn.RMSNorm(single_features)
        else:
            self._pre_mlp_norm = nn.Identity()
        if self._config.compile_modules:
            self._compile_modules()

    def forward(
        self,
        single_features: th.Tensor,
        pair_features: BatchedMatrixTensor,
        seq_lens: th.Tensor,
        mask: th.Tensor,
    ) -> tuple[th.Tensor, th.Tensor]:
        """Forward pass of the PairFormer module.

        Args:
            single_features: Single representations. Shape (batch_size,
            pair_features: Bias tensor of shape (batch_size, seq_len, seq_len, pair_features).
            seq_lens: Sequence lengths for each batch element. Shape (batch_size,).
            mask: Attention mask. Shape (batch_size, seq_len).

        Returns:
            Updated single and pair representations.
        """
        # Even though seq_lens and mask contain the same information, we require both
        # because some modules use seq_lens, while others use mask...
        # The interface is cleaned up in the PairFormer module.
        pair_features = pair_features + self.dropout_row1(
            self.triangle_multiplication_outgoing(pair_features, seq_lens)
        )
        pair_features = pair_features + self.dropout_row2(
            self.triangle_multiplication_incoming(pair_features, seq_lens)
        )
        pair_features = pair_features + self.dropout_row3(
            self.triangle_attention_starting(pair_features, seq_lens)
        )
        pair_features = pair_features + self.dropout_col1(
            self.triangle_attention_ending(pair_features, seq_lens)
        )
        pair_features = pair_features + self.pair_mlp(pair_features)

        single_features = single_features + self.single_attention(
            single_features,
            bias=pair_features,
            mask=mask,
        )
        single_features = single_features + self.single_mlp(single_features)
        return single_features, pair_features

    def _compile_modules(self):
        # Increase cache limit test.
        th._dynamo.config.cache_size_limit = 1000
        self.triangle_multiplication_outgoing = th.compile(
            self.triangle_multiplication_outgoing,
            dynamic=True,
            fullgraph=True,
        )
        self.triangle_multiplication_incoming = th.compile(
            self.triangle_multiplication_incoming,
            dynamic=True,
            fullgraph=True,
        )
        self.triangle_attention_starting = th.compile(
            self.triangle_attention_starting,
            dynamic=True,
            fullgraph=True,
        )
        self.triangle_attention_ending = th.compile(
            self.triangle_attention_ending,
            dynamic=True,
            fullgraph=True,
        )
        self.pair_mlp = th.compile(
            self.pair_mlp,
            dynamic=True,
            fullgraph=True,
        )
        # Single attention has issues with compilation.
        self.single_mlp = th.compile(
            self.single_mlp,
            dynamic=True,
            fullgraph=True,
        )
