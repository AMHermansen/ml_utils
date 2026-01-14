import torch as th
from torch import nn

from ml_utils.utils import default

from .pairformer_block import PairFormerBlock
from .utils import PairFormerConfig


class PairFormer(nn.Module):
    def __init__(
        self,
        single_features: int,
        pair_features: int,
        config: PairFormerConfig | None = None,
    ):
        config = default(config, PairFormerConfig())
        super().__init__()
        self._single_features = single_features
        self._pair_features = pair_features
        self._config = config

        self._blocks = nn.ModuleList([
            PairFormerBlock(
                single_features=single_features,
                pair_features=pair_features,
                config=config.block_config,
            )
            for _ in range(config.num_blocks)
        ])

    def forward(
        self,
        single_features: th.Tensor,
        pair_features: th.Tensor,
        seq_lens: th.Tensor,
    ) -> tuple[th.Tensor, th.Tensor]:
        """Forward pass of PairFormer.

        Args:
            single_features: Single representation. Jagged tensor.
            pair_features: Pair representation.
                Shape (batch_size, seq_len, seq_len, pair_features)
            seq_lens: Sequence lengths. Shape (batch_size,).

        Returns:
            Updated single and pair representations.
        """
        for block in self._blocks:
            single_features, pair_features = block(
                single_features,
                pair_features,
                seq_lens,
            )
        return single_features, pair_features
