from typing import Any

import torch as th
from torch.utils.data import default_collate


def check_variable_in_list_of_batches(
    variable_name: str, batches: list[dict[str, any]]
) -> bool:
    return all(variable_name in batch for batch in batches)


class CumulativeSeqlengthCollator:
    """Collate function that effectively applies Sequence Packing to a batch of batches.

    Attributes:
        seqlen_variable_names (list[str]): List of variable names that are sequences
            and should be concatenated.
        seqlen_key (str): Key in the batch dictionaries that contains the cumulative
            sequence lengths.

    Examples:
        >>> collator = CumulativeSeqlengthCollator(seqlen_variable_names=["features"])
        >>> batch1 = {"features": th.randn(3, 5), "cu_seqlen": th.tensor([3]), "label": th.tensor(0)}
        >>> batch2 = {"features": th.randn(2, 5), "cu_seqlen": th.tensor([2]), "label": th.tensor(1)}
        >>> combined_batch = collator([batch1, batch2])
        >>> print(combined_batch["features"].shape)  # torch.Size([5, 5])
        >>> print(combined_batch["cu_seqlen"])  # tensor([0, 3, 5])
        >>> print(combined_batch["label"])  # tensor([0, 1])
    """

    def __init__(
        self,
        seqlen_variable_names: list[str] | str,
        seqlen_keys: list[str] | str = "cu_seqlen",
    ):
        """Constructor for CumulativeSeqlengthCollator.

        Args:
            seqlen_variable_names: List of variable names that are sequences and should be concatenated.
            seqlen_keys: Key(s) in the batch dictionaries that contains the cumulative sequence lengths.
        """
        self.seqlen_variable_names = (
            seqlen_variable_names
            if isinstance(seqlen_variable_names, list)
            else [seqlen_variable_names]
        )
        self.seqlen_key = (
            seqlen_keys if isinstance(seqlen_keys, list) else [seqlen_keys]
        )

    @property
    def names_used_for_seq_len(self) -> set[str]:
        """Set of variable names used for sequence length calculations."""
        return set(self.seqlen_variable_names + self.seqlen_key)

    def __call__(self, batches: list[dict[str, Any]]) -> dict[str, Any]:
        for var_name in self.seqlen_variable_names:
            if not check_variable_in_list_of_batches(var_name, batches):
                raise ValueError(f"All batches must contain the key '{var_name}'")
        for var_name in self.seqlen_key:
            if not check_variable_in_list_of_batches(var_name, batches):
                raise ValueError(f"All batches must contain the key '{var_name}'")

        if len(batches) == 1:
            cumulative_seqlengths = {
                seqlen_key: th.tensor([0, batches[0][seqlen_key]], dtype=th.int64)
                for seqlen_key in self.seqlen_key
            }
        else:
            cumulative_seqlengths = {
                seqlen_key: th.tensor(
                    [0] + [batch[seqlen_key] for batch in batches], dtype=th.int64
                ).cumsum(dim=0)
                for seqlen_key in self.seqlen_key
            }

        seq_len_vars = {var_name: th.cat([batch[var_name] for batch in batches], dim=0)}
        non_seq_len_vars = default_collate(
            [
                {k: v for k, v in batch.items() if k not in self.names_used_for_seq_len}
                for batch in batches
            ]
        )
        return {
            **seq_len_vars,
            **non_seq_len_vars,
            **cumulative_seqlengths,
        }
