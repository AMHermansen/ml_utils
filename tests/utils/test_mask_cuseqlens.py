import hypothesis.strategies as st
import torch as th
from hypothesis import given

from ml_utils.torch_utils.packing import get_masked_cu_seqlens
from ml_utils.torch_utils.types import CulensTensor


def get_masked_cu_seqlens_naive_correct(
    cu_seqlens: CulensTensor,
    mask: th.Tensor,
) -> CulensTensor:
    """Definitely correct naive implementation using loops"""
    result = []
    for i in range(len(cu_seqlens)):
        end_idx = cu_seqlens[i]
        # Count number of True values in mask[0:end_idx]
        count = th.sum(mask[:end_idx].to(th.int32)).item()
        result.append(count)
    return th.tensor(result, dtype=th.int32, device=cu_seqlens.device)

@given(
    batch_size=st.integers(min_value=1, max_value=20),
    max_seq_len=st.integers(min_value=1, max_value=50),
    mask_ratio=st.floats(min_value=0.0, max_value=1.0),
)
def test_against_naive_implementation(batch_size: int, max_seq_len: int, mask_ratio: float):
    """Compare the optimized implementation with the definitely correct naive one."""
    # Generate random sequence lengths
    seq_lens = th.randint(1, max_seq_len + 1, (batch_size,))
    total_seqlen = seq_lens.sum().item()

    # Create cumulative sequence lengths
    cu_seqlens = th.cat([th.zeros(1, dtype=th.int32), th.cumsum(seq_lens, dim=0)])

    # Create random mask
    mask = th.rand(total_seqlen) > mask_ratio  # ~70% True values

    # Get results from both implementations
    result_optimized = get_masked_cu_seqlens(cu_seqlens, mask)
    result_naive = get_masked_cu_seqlens_naive_correct(cu_seqlens, mask)

    # They should be exactly equal
    assert th.allclose(result_optimized, result_naive), \
        f"Optimized: {result_optimized}, Naive: {result_naive}"
