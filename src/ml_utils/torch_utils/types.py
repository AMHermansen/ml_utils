import jaxtyping as jt
import torch as th

type GeneralBatchedTensor = jt.Float[th.Tensor, "batch_size length ... dimension"]
type GeneralPackedTensor = jt.Float[th.Tensor, "total_valid_entries ... dimension"]

type PackedTensor = jt.Float[th.Tensor, "packed_length dimension"]
type BatchedTensor = jt.Float[th.Tensor, "batch_size length dimension"]

type CulensTensor = jt.Int[th.Tensor, " batch_size+1"]
type MaskTensor = jt.Bool[th.Tensor, "batch_size length"]


# Multi-head attention-specific types
type PackedMHATensor = jt.Float[th.Tensor, "packed_length nheads dimension"]
type PackedKVTensor = jt.Float[th.Tensor, "packed_length 2 nheads dimension"]
type PackedQKVTensor = jt.Float[th.Tensor, "packed_length 3 nheads dimension"]

type AllPackedQKVTypes = (
    tuple[PackedMHATensor, PackedMHATensor, PackedMHATensor]
    | tuple[PackedMHATensor, PackedKVTensor]
    | PackedQKVTensor
)
