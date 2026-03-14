import torch as th
from torch import nn

from ml_utils.utils import default, exists

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
            for _ in range(config.num_layers)
        ])

    def forward(
        self,
        single_features: th.Tensor,
        pair_features: th.Tensor,
        seq_lens: th.Tensor | None = None,
        mask: th.Tensor | None = None,
    ) -> tuple[th.Tensor, th.Tensor]:
        """Forward pass of PairFormer.

        At least one of seq_lens or mask must be provided.

        Args:
            single_features: Single representation.
                Shape (batch_size, seq_len, single_features)
            pair_features: Pair representation.
                Shape (batch_size, seq_len, seq_len, pair_features)
            seq_lens: Optional Sequence lengths. Shape (batch_size,).
                If None, seq_lens will be inferred from mask.
            mask: Optional mask. Shape (batch_size, seq_len)
                If None, mask will be inferred from seq_lens.

        Returns:
            Tuple of (single_features, pair_features) after processing through the blocks.
                single_features: Shape (batch_size, seq_len, single_features)
                pair_features: Shape (batch_size, seq_len, seq_len, pair_features)

        Raises:
            ValueError: If both seq_lens and mask are None.
        """
        if not exists(seq_lens) and not exists(mask):
            raise ValueError("At least one of seq_lens or mask must be provided.")

        if not exists(mask):
            assert exists(seq_lens)  # Should be guaranteed by previous check
            mask = (
                th.arange(seq_lens.max().item()).unsqueeze(0).to(seq_lens.device)
                < seq_lens.unsqueeze(1)
            )
        if not exists(seq_lens):
            assert exists(mask)  # Should be guaranteed by previous check
            seq_lens = mask.sum(dim=1)

        for block in self._blocks:
            single_features, pair_features = block(
                single_features,
                pair_features,
                seq_lens,
                mask,
            )
        return single_features, pair_features
