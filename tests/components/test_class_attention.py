import random

import hypothesis.strategies as st
import pytest
import torch as th
from hypothesis import given, settings
from torch import Tensor

from ml_utils.components import ClassAttentionPooling


# Strategies for generating valid inputs
@st.composite
def model_configs(draw):
    dimension_multiple_base = 16
    in_features = draw(st.integers(min_value=1, max_value=4))
    out_features = draw(st.integers(min_value=1, max_value=4))
    latent_dim = draw(st.one_of(st.none(), st.integers(min_value=1, max_value=4)))
    context_dim = draw(st.integers(min_value=0, max_value=32))
    num_layers = draw(st.integers(min_value=1, max_value=2))
    return (
        dimension_multiple_base * in_features,
        dimension_multiple_base * out_features,
        dimension_multiple_base * latent_dim if latent_dim is not None else None,
        context_dim,
        num_layers,
    )


def packed_sequences(
    batch_size: int, max_seqlen: int, in_features: int
) -> tuple[Tensor, Tensor]:
    # Generate random sequence lengths and create a valid packed sequence
    seq_lens = [random.randint(1, max_seqlen) for _ in range(batch_size)]
    cu_seqlens = [0]
    for len_ in seq_lens:
        cu_seqlens.append(cu_seqlens[-1] + len_)
    total_seq_len = cu_seqlens[-1]
    x = th.randn(total_seq_len, in_features)
    return x, th.tensor(cu_seqlens, dtype=th.int32)


# Main test function
@given(
    model_configs(),
    st.integers(min_value=1, max_value=8),  # batch_size
    st.integers(min_value=1, max_value=16),
)
@settings(max_examples=10, deadline=10_000)
def test_class_attention_pooling_shapes(model_config, batch_size, max_seqlen):
    if not th.cuda.is_available():
        pytest.skip("CUDA is not available")
    device = th.device("cuda")
    in_features, out_features, latent_dim, context_dim, num_layers = model_config

    # Skip invalid configurations where latent_dim < out_features might break RMSNorm
    if latent_dim is not None and latent_dim < out_features:
        return

    model = ClassAttentionPooling(
        in_features=in_features,
        out_features=out_features,
        latent_dim=latent_dim,
        context_dim=context_dim,
        num_layers=num_layers,
    ).to(device)

    # Generate input sequence
    x, cu_seqlens = packed_sequences(
        batch_size=batch_size, max_seqlen=max_seqlen, in_features=in_features
    )
    x = x.to(device)
    cu_seqlens = cu_seqlens.to(device)

    # Prepare context if needed
    context = None
    if context_dim > 0:
        context = th.randn(batch_size, context_dim).to(device)

    # Forward pass
    output = model(x, cu_seqlens, context=context)

    # Verify output shape
    assert output.shape == (batch_size, out_features)
    assert isinstance(output, Tensor)
