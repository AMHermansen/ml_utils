"""Type aliases for various tensor shapes used in the project."""

from typing import TypeAlias

import jaxtyping as jt
import torch as th

GeneralBatchedTensor: TypeAlias = jt.Float[th.Tensor, " batch_size *features"]
GeneralPackedTensor: TypeAlias = jt.Float[th.Tensor, " total_valid_entries *features"]
GeneralUnpackedTensor: TypeAlias = jt.Float[
    th.Tensor, " batch_size max_length *features"
]

PackedTensor: TypeAlias = jt.Float[th.Tensor, " packed_length features"]
BatchSequenceTensor: TypeAlias = jt.Float[th.Tensor, " batch_size length features"]
BatchedMatrixTensor: TypeAlias = jt.Float[th.Tensor, " batch_size length length dim"]

CulensTensor: TypeAlias = jt.Int[th.Tensor, " batch_size+1"]
MaskTensor: TypeAlias = jt.Bool[th.Tensor, "batch_size length"]


# Multi-head attention-specific types
PackedMHATensor: TypeAlias = jt.Float[th.Tensor, "packed_length nheads features"]
PackedKVTensor: TypeAlias = jt.Float[th.Tensor, "packed_length 2 nheads features"]
PackedQKVTensor: TypeAlias = jt.Float[th.Tensor, "packed_length 3 nheads features"]

BatchedHeadedMatrixTensor: TypeAlias = jt.Float[
    th.Tensor, "batch_size length length dim"
]

AllPackedQKVTypes: TypeAlias = (
    tuple[PackedMHATensor, PackedMHATensor, PackedMHATensor]
    | tuple[PackedMHATensor, PackedKVTensor]
    | PackedQKVTensor
)
