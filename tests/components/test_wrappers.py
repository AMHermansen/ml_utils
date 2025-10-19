from typing import Literal

import torch as th
from hypothesis import given, settings
from hypothesis import strategies as st

from ml_utils.components import (
    Residual,
    ResidualConfig,
    ResidualWithContext,
    SwiGLUMLP,
)

batch_ndims_strategy = st.integers(min_value=1, max_value=3)
batch_size_strategy = st.integers(min_value=1, max_value=8)
feature_dim_strategy = st.integers(min_value=1, max_value=32)
context_dim_strategy = st.integers(min_value=1, max_value=16)


def random_shape_strategy():
    return (
        st.tuples(
            batch_ndims_strategy,
            feature_dim_strategy,
        )
        .flatmap(
            lambda args: st.tuples(
                st.lists(batch_size_strategy, min_size=args[0], max_size=args[0]),
                st.just(args[1]),
            )
        )
        .map(lambda tup: (*tuple(tup[0]), tup[1]))
    )


@given(shape=random_shape_strategy())
@settings(deadline=10000)
def test_residual_forward_shape(shape):
    tensor = th.randn(*shape)
    dim = shape[-1]
    mlp = SwiGLUMLP(dim)
    wrapper = Residual(mlp)
    out = wrapper(tensor)
    assert out.shape == tensor.shape


@given(
    shape=random_shape_strategy(),
    norm_name=st.sampled_from(["layer", "rms"]),
)
@settings(deadline=10000)
def test_prenorm_residual_forward_shape(shape, norm_name: Literal["layer", "rms"]):
    tensor = th.randn(*shape)
    dim = shape[-1]
    mlp = SwiGLUMLP(dim)
    wrapper = Residual(mlp, ResidualConfig(norm_name=norm_name))
    out = wrapper(tensor)
    assert out.shape == tensor.shape


@given(
    shape=random_shape_strategy(),
    context_dim=context_dim_strategy,
)
@settings(deadline=10000)
def test_residual_with_context_forward_shape(shape, context_dim):
    tensor = th.randn(*shape)
    context = th.randn(shape[0], context_dim)
    dim = shape[-1]
    mlp = SwiGLUMLP(dim)
    wrapper = ResidualWithContext(mlp, context_dim=context_dim)
    out = wrapper(tensor, context=context)
    assert out.shape == tensor.shape
