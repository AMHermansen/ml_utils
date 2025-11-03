import math

import pytest
import torch as th
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from ml_utils.components import (
    CosineEmbedding,
    FourierEmbedding,
)


class TestFourierEmbedding:
    """Test cases for FourierEmbedding class."""

    @given(
        num_frequencies=st.integers(min_value=1, max_value=100),
        batch_size=st.integers(min_value=1, max_value=10),
        input_dim=st.integers(min_value=1, max_value=5),
    )
    @settings(max_examples=100, deadline=10_000)
    def test_output_shape(self, num_frequencies, batch_size, input_dim):
        """Test that output has correct shape."""
        embedding = FourierEmbedding(num_frequencies=num_frequencies)

        # Test with various input shapes
        input_tensor = th.randn(batch_size, input_dim)
        output = embedding(input_tensor)

        assert output.shape == (batch_size, input_dim, num_frequencies)
        assert output.dtype == input_tensor.dtype

    @given(num_frequencies=st.integers(min_value=1, max_value=50))
    @settings(max_examples=100, deadline=10_000)
    def test_properties(self, num_frequencies):
        """Test in_features and out_features properties."""
        embedding = FourierEmbedding(num_frequencies=num_frequencies)

        assert embedding.out_features == num_frequencies
        assert embedding.in_features is None

    @given(
        num_frequencies=st.integers(min_value=1, max_value=20),
        x_value=st.floats(min_value=-10.0, max_value=10.0),
    )
    @settings(max_examples=50, deadline=10_000)
    def test_forward_values(self, num_frequencies, x_value):
        """Test that forward produces expected cosine values."""
        embedding = FourierEmbedding(num_frequencies=num_frequencies)

        x = th.tensor([[x_value]])
        output = embedding(x)

        # Check that values are in expected range [sqrt(2)-1, sqrt(2)+1]
        expected_min = math.sqrt(2) - 1 - 1e-6  # Allow for numerical error
        expected_max = math.sqrt(2) + 1 + 1e-6

        assert th.all(output >= expected_min)
        assert th.all(output <= expected_max)

        # Check specific computation for scalar input
        expected = (x_value * embedding.freqs + embedding.phases).cos() + math.sqrt(2)
        assert th.allclose(output.squeeze(), expected, atol=1e-6)

    @given(num_frequencies=st.integers(min_value=1, max_value=10))
    @settings(max_examples=20, deadline=10_000)
    def test_different_dtypes(self, num_frequencies):
        """Test with different input dtypes."""
        embedding = FourierEmbedding(num_frequencies=num_frequencies)

        for dtype in [th.float32, th.float64]:
            x = th.randn(2, 3, dtype=dtype)
            output = embedding(x)

            assert output.dtype == dtype

    @given(num_frequencies=st.integers(min_value=1, max_value=10))
    @settings(max_examples=10, deadline=10_000)
    def test_gradients(self, num_frequencies):
        """Test that gradients flow through the embedding."""
        embedding = FourierEmbedding(num_frequencies=num_frequencies)
        x = th.randn(3, 2, requires_grad=True)

        output = embedding(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert not th.allclose(x.grad, th.zeros_like(x.grad))

    @given(
        num_frequencies=st.integers(min_value=1, max_value=10),
        shape=st.tuples(
            st.integers(min_value=1, max_value=3),
            st.integers(min_value=1, max_value=3),
            st.integers(min_value=1, max_value=3),
        ),
    )
    @settings(max_examples=20, deadline=10_000)
    def test_arbitrary_shapes(self, num_frequencies, shape):
        """Test with arbitrary input shapes."""
        embedding = FourierEmbedding(num_frequencies=num_frequencies)
        x = th.randn(*shape)

        output = embedding(x)
        expected_shape = shape + (num_frequencies,)
        assert output.shape == expected_shape


class TestCosineEmbedding:
    """Test cases for CosineEmbedding class."""

    @given(
        out_dim=st.integers(min_value=1, max_value=100),
        scheme=st.sampled_from(["linear", "lin", "exponential", "exp", "power", "pow"]),
        do_sin=st.booleans(),
        batch_size=st.integers(min_value=1, max_value=10),
    )
    @settings(max_examples=100, deadline=10_000)
    def test_output_shape(self, out_dim, scheme, do_sin, batch_size):
        """Test that output has correct shape."""
        if do_sin and out_dim % 2 != 0:
            # When do_sin is True, out_dim must be even
            out_dim += 1
        embedding = CosineEmbedding(
            out_dim=out_dim,
            scheme=scheme,
            min_value=0.0,
            max_value=1.0,
            do_sin=do_sin,
        )

        x = th.rand(batch_size, 1)  # Input in [0, 1] range
        output = embedding(x)

        expected_out_dim = out_dim
        assert output.shape == (*x.shape, expected_out_dim)
        assert output.dtype == x.dtype

    @given(
        out_dim=st.integers(min_value=1, max_value=50),
        scheme=st.sampled_from(["linear", "exponential", "power"]),
        do_sin=st.booleans(),
    )
    @settings(max_examples=50, deadline=10_000)
    def test_properties(self, out_dim, scheme, do_sin):
        """Test in_features and out_features properties."""
        if do_sin and out_dim % 2 != 0:
            out_dim += 1
        embedding = CosineEmbedding(
            out_dim=out_dim,
            scheme=scheme,
            min_value=0.0,
            max_value=1.0,
            do_sin=do_sin,
        )

        assert embedding.out_features == out_dim
        assert embedding.in_features is None

    @given(
        out_dim=st.integers(min_value=2, max_value=20),
        scheme=st.sampled_from(["linear", "exponential", "power"]),
        do_sin=st.booleans(),
        x_value=st.floats(min_value=0.0, max_value=1.0),
    )
    @settings(max_examples=50, deadline=10_000)
    def test_forward_values(self, out_dim, scheme, do_sin, x_value):
        """Test that forward produces expected cosine values."""
        if do_sin and out_dim % 2 != 0:
            out_dim += 1
        embedding = CosineEmbedding(
            out_dim=out_dim,
            scheme=scheme,
            min_value=0.0,
            max_value=1.0,
            do_sin=do_sin,
        )

        x = th.tensor([[x_value]])
        output = embedding(x)

        # Check that values are in expected range [-1, 1]
        assert th.all(output >= -1.0 - 1e-6)
        assert th.all(output <= 1.0 + 1e-6)

        # Check output dimension based on do_sin
        if do_sin:
            assert output.shape[-1] == out_dim
        else:
            assert output.shape[-1] == out_dim

    @given(
        out_dim=st.integers(min_value=1, max_value=10),
        min_value=st.floats(min_value=-10.0, max_value=0.0),
        max_value=st.floats(min_value=0.1, max_value=10.0),
        x_value=st.floats(min_value=-5.0, max_value=5.0),
    )
    @settings(max_examples=30, deadline=10_000)
    def test_normalization(self, out_dim, min_value, max_value, x_value):
        """Test input normalization to [0, 1] range."""
        assume(min_value < max_value)
        assume(min_value <= x_value <= max_value)

        embedding = CosineEmbedding(
            out_dim=out_dim,
            scheme="linear",
            min_value=min_value,
            max_value=max_value,
            do_sin=False,
        )

        x = th.tensor([[x_value]])
        output = embedding(x)

        # Should not raise assertion error from _check_bounds
        assert output is not None

    @given(
        out_dim=st.integers(min_value=1, max_value=10),
        scheme=st.sampled_from(["linear", "exponential", "power"]),
    )
    @settings(max_examples=20, deadline=10_000)
    def test_frequency_schemes(self, out_dim, scheme):
        """Test different frequency schemes."""
        embedding = CosineEmbedding(
            out_dim=out_dim,
            scheme=scheme,
            min_value=0.0,
            max_value=1.0,
            do_sin=False,
        )

        x = th.tensor([[0.5]])
        output = embedding(x)

        # Check that frequencies are computed correctly
        if scheme in {"linear", "lin"}:
            expected_freqs = th.arange(out_dim).float() + 1
        elif scheme in {"exponential", "exp"}:
            expected_freqs = th.exp(th.arange(out_dim).float())
        elif scheme in {"power", "pow"}:
            expected_freqs = 2 ** th.arange(out_dim).float()

        assert th.allclose(embedding.freqs, expected_freqs)

    @given(out_dim=st.integers(min_value=1, max_value=10))
    @settings(max_examples=10, deadline=10_000)
    def test_invalid_scheme(self, out_dim):
        """Test that invalid scheme raises ValueError."""
        with pytest.raises(ValueError):
            CosineEmbedding(
                out_dim=out_dim,
                scheme="invalid_scheme",
                min_value=0.0,
                max_value=1.0,
                do_sin=False,
            )

    @given(out_dim=st.integers(min_value=1, max_value=10))
    @settings(max_examples=10, deadline=10_000)
    def test_different_dtypes(self, out_dim):
        """Test with different input dtypes."""
        embedding = CosineEmbedding(
            out_dim=out_dim,
            scheme="linear",
            min_value=0.0,
            max_value=1.0,
            do_sin=False,
        )

        for dtype in [th.float32, th.float64]:
            x = th.rand(2, 1, dtype=dtype)
            output = embedding(x)

            assert output.dtype == dtype

    @given(out_dim=st.integers(min_value=1, max_value=10))
    @settings(max_examples=10, deadline=10_000)
    def test_gradients(self, out_dim):
        """Test that gradients flow through the embedding."""
        embedding = CosineEmbedding(
            out_dim=out_dim,
            scheme="linear",
            min_value=0.0,
            max_value=1.0,
            do_sin=False,
        )

        x = th.rand(3, 1, requires_grad=True)
        output = embedding(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert not th.allclose(x.grad, th.zeros_like(x.grad))

    @given(
        out_dim=st.integers(min_value=1, max_value=10),
        shape=st.tuples(
            st.integers(min_value=1, max_value=3),
            st.integers(min_value=1, max_value=3),
            st.integers(min_value=1, max_value=3),
        ),
    )
    @settings(max_examples=20, deadline=10_000)
    def test_arbitrary_shapes(self, out_dim, shape):
        """Test with arbitrary input shapes."""
        embedding = CosineEmbedding(
            out_dim=out_dim,
            scheme="linear",
            min_value=0.0,
            max_value=1.0,
            do_sin=False,
        )

        x = th.rand(*shape)
        output = embedding(x)

        expected_shape = (*shape, out_dim)
        assert output.shape == expected_shape
