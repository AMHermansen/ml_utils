from typing import TypeAlias

import jaxtyping as jt
import torch as th

GeneralBatchedTensor: TypeAlias = jt.Float[th.Tensor, "batch_size length ... dimension"]
GeneralPackedTensor: TypeAlias = jt.Float[th.Tensor, "total_valid_entries ... dimension"]

PackedTensor: TypeAlias = jt.Float[th.Tensor, "packed_length dimension"]
BatchedTensor: TypeAlias = jt.Float[th.Tensor, "batch_size length dimension"]

CulensTensor: TypeAlias = jt.Int[th.Tensor, " batch_size+1"]
MaskTensor: TypeAlias = jt.Bool[th.Tensor, "batch_size length"]


# Multi-head attention-specific types
PackedMHATensor: TypeAlias = jt.Float[th.Tensor, "packed_length nheads dimension"]
PackedKVTensor: TypeAlias = jt.Float[th.Tensor, "packed_length 2 nheads dimension"]
PackedQKVTensor: TypeAlias = jt.Float[th.Tensor, "packed_length 3 nheads dimension"]

AllPackedQKVTypes: TypeAlias = (
    tuple[PackedMHATensor, PackedMHATensor, PackedMHATensor]
    | tuple[PackedMHATensor, PackedKVTensor]
    | PackedQKVTensor
)
