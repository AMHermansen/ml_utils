import torch as th
from hypothesis import given, settings
from hypothesis import strategies as st

from ml_utils.components.swiglu import SwiGLU, SwiGLUMLP


@given(
    batch=st.integers(min_value=1, max_value=8),
    dim=st.integers(min_value=2, max_value=128).filter(
        lambda x: x % 2 == 0
    ),  # Ensure dim is even
)
@settings(deadline=10_000)  # This test can be slow sometimes.
def test_swiglu_forward_shape(batch, dim):
    x = th.randn(batch, dim)
    act = SwiGLU()
    out = act(x)
    assert out.shape == (batch, dim // 2)


@given(
    batch=st.integers(min_value=1, max_value=8),
    dim=st.integers(min_value=2, max_value=128),
)
@settings(deadline=10_000)  # This test can be slow sometimes.
def test_mlpswiglu_forward_shape(batch, dim):
    x = th.randn(batch, dim)
    mlp = SwiGLUMLP(dim)
    out = mlp(x)
    assert out.shape == (batch, dim)
