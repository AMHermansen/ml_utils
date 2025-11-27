from itertools import pairwise
from random import shuffle
from typing import Any

import torch as th
from torch.utils.data import default_collate, Sampler


def check_variable_in_list_of_batches(
    variable_name: str, batches: list[dict[str, Any]]
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
        cu_seqlen_names: list[str] | str = "cu_seqlen",
    ):
        """Constructor for CumulativeSeqlengthCollator.

        Args:
            seqlen_variable_names: List of variable names that are sequences and should be concatenated.
            cu_seqlen_names: Key(s) in the batch dictionaries that contains the cumulative sequence lengths.
        """
        self.seqlen_variable_names = (
            seqlen_variable_names
            if isinstance(seqlen_variable_names, list)
            else [seqlen_variable_names]
        )
        self.seqlen_key = (
            cu_seqlen_names if isinstance(cu_seqlen_names, list) else [cu_seqlen_names]
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

        seq_len_vars = {
            var_name: th.cat([batch[var_name] for batch in batches], dim=0)
            for var_name in self.seqlen_variable_names
        }
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


class SequenceBucketingSampler(Sampler):
    def __init__(
        self,
        lengths: th.Tensor,
        batch_size: int,
        length_splits: list[int],
        drop_exceeding: bool = False,
        shuffle: bool = False,
    ):
        self._lengths = lengths
        self._batch_size = batch_size
        self._length_splits = length_splits
        self._shuffle = shuffle
        self._drop_exceeding = drop_exceeding

        self._create_buckets()

    def _create_buckets(self):
        indices = th.arange(len(self._lengths))
        first_bucket = indices[self._lengths < self._length_splits[0]]
        buckets = [first_bucket]
        for lower_bound, upper_bound in pairwise(self._length_splits):
            bucket_indices = indices[
                (self._lengths >= lower_bound) & (self._lengths < upper_bound)
            ]
            buckets.append(bucket_indices)
        buckets.append(indices[self._lengths >= self._length_splits[-1]])
        self._buckets = buckets

    def shuffle_buckets(self, buckets: list[th.Tensor]) -> list[th.Tensor]:
        """Shuffle indices within each bucket and the bucket order if enabled."""
        if not self._shuffle:
            return buckets

        shuffled_buckets: list[th.Tensor] = []
        for b in buckets:
            if len(b) == 0:
                shuffled_buckets.append(b)
                continue
            perm = th.randperm(len(b))
            shuffled_buckets.append(b[perm])

        if len(shuffled_buckets) > 1:
            order = th.randperm(len(shuffled_buckets)).tolist()
            shuffled_buckets = [shuffled_buckets[i] for i in order]

        return shuffled_buckets

    def batchify_buckets(self, buckets: list[th.Tensor]) -> list[list[th.Tensor]]:
        """Split each bucket into a list of batch tensors (per-bucket)."""
        per_bucket_batches: list[list[th.Tensor]] = []
        for b in buckets:
            n = len(b)
            bucket_batches: list[th.Tensor] = []
            for i in range(0, n, self._batch_size):
                chunk = b[i : i + self._batch_size]
                if len(chunk) < self._batch_size and self._drop_exceeding:
                    continue
                bucket_batches.append(chunk)
            per_bucket_batches.append(bucket_batches)
        return per_bucket_batches

    def flatten_buckets(self, batched_buckets: list[list[th.Tensor]]) -> list[th.Tensor]:
        """Flatten the per-bucket lists of batches into a single list of batches."""
        return [batch for bucket in batched_buckets for batch in bucket]

    def __iter__(self):
        # work on copies to avoid mutating internal state
        buckets_copy = [b.clone() for b in self._buckets]
        shuffled = self.shuffle_buckets(buckets_copy)
        per_bucket = self.batchify_buckets(shuffled)
        flat_batches = self.flatten_buckets(per_bucket)
        shuffle(flat_batches)

        for batch in flat_batches:
            yield batch.tolist()
