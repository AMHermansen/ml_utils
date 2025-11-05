
import hypothesis.strategies as st
import pytest
import torch as th
from hypothesis import HealthCheck, given, settings

from ml_utils.components.attention.qkv_linear import (
    QKVLinear,
)


class TestQKVLinear:
    # Hypothesis strategies
    batch_sizes = st.tuples(st.integers(1, 4), st.integers(1, 4))
    in_features = st.integers(1, 32)
    bias_flags = st.booleans()
    split_flags = st.booleans()

    @given(in_features=in_features, bias=bias_flags, split_qkv=split_flags)
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_initialization(self, in_features: int, bias: bool, split_qkv: bool):
        """Test that QKVLinear initializes correctly with different configurations."""
        module = QKVLinear(in_features=in_features, bias=bias, split_qkv=split_qkv)

        assert module._in_features == in_features
        assert module._bias == bias
        assert module._split_qkv == split_qkv

        if split_qkv:
            assert module._q_linear is not None
            assert module._k_linear is not None
            assert module._v_linear is not None
            assert module._qkv_linear is None

            assert module._q_linear.in_features == in_features
            assert module._q_linear.out_features == in_features
            assert (module._q_linear.bias is not None) == bias
        else:
            assert module._q_linear is None
            assert module._k_linear is None
            assert module._v_linear is None
            assert module._qkv_linear is not None

            assert module._qkv_linear.in_features == in_features
            assert module._qkv_linear.out_features == in_features * 3
            assert (module._qkv_linear.bias is not None) == bias

    @given(
        in_features=in_features,
        bias=bias_flags,
        split_qkv=split_flags,
        batch_dims=batch_sizes
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_forward_shape(self, in_features: int, bias: bool, split_qkv: bool, batch_dims: tuple):
        """Test that forward pass returns correct shapes."""
        module = QKVLinear(in_features=in_features, bias=bias, split_qkv=split_qkv)
        batch_shape = (*batch_dims, in_features)
        x = th.randn(*batch_shape)

        result = module(x)

        if split_qkv:
            assert isinstance(result, tuple)
            assert len(result) == 3
            for tensor in result:
                assert tensor.shape == batch_shape
        else:
            assert isinstance(result, th.Tensor)
            assert result.shape == (*batch_dims, in_features * 3)

    @given(in_features=in_features, bias=bias_flags)
    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    def test_set_split_qkv_merge_to_split(self, in_features: int, bias: bool):
        """Test converting from merged to split QKV projections."""
        module = QKVLinear(in_features=in_features, bias=bias, split_qkv=False)

        # Initialize with some weights
        if bias:
            module._qkv_linear.bias.data.uniform_(-0.1, 0.1)
        module._qkv_linear.weight.data.uniform_(-0.1, 0.1)

        original_weight = module._qkv_linear.weight.data.clone()
        original_bias = module._qkv_linear.bias.data.clone() if bias else None

        # Convert to split
        module.set_split_qkv(True)

        assert module.split_qkv
        assert module._q_linear is not None
        assert module._k_linear is not None
        assert module._v_linear is not None
        assert module._qkv_linear is None

        # Check that weights were correctly split
        out_each = in_features
        expected_w_q = original_weight[:out_each]
        expected_w_k = original_weight[out_each:out_each*2]
        expected_w_v = original_weight[out_each*2:]

        th.testing.assert_close(module._q_linear.weight.data, expected_w_q)
        th.testing.assert_close(module._k_linear.weight.data, expected_w_k)
        th.testing.assert_close(module._v_linear.weight.data, expected_w_v)

        if bias:
            expected_b_q = original_bias[:out_each]
            expected_b_k = original_bias[out_each:out_each*2]
            expected_b_v = original_bias[out_each*2:]

            th.testing.assert_close(module._q_linear.bias.data, expected_b_q)
            th.testing.assert_close(module._k_linear.bias.data, expected_b_k)
            th.testing.assert_close(module._v_linear.bias.data, expected_b_v)

    @given(in_features=in_features, bias=bias_flags)
    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    def test_set_split_qkv_split_to_merge(self, in_features: int, bias: bool):
        """Test converting from split to merged QKV projections."""
        module = QKVLinear(in_features=in_features, bias=bias, split_qkv=True)

        # Initialize with some weights
        if bias:
            module._q_linear.bias.data.uniform_(-0.1, 0.1)
            module._k_linear.bias.data.uniform_(-0.1, 0.1)
            module._v_linear.bias.data.uniform_(-0.1, 0.1)

        module._q_linear.weight.data.uniform_(-0.1, 0.1)
        module._k_linear.weight.data.uniform_(-0.1, 0.1)
        module._v_linear.weight.data.uniform_(-0.1, 0.1)

        original_w_q = module._q_linear.weight.data.clone()
        original_w_k = module._k_linear.weight.data.clone()
        original_w_v = module._v_linear.weight.data.clone()
        original_b_q = module._q_linear.bias.data.clone() if bias else None
        original_b_k = module._k_linear.bias.data.clone() if bias else None
        original_b_v = module._v_linear.bias.data.clone() if bias else None

        # Convert to merged
        module.set_split_qkv(False)

        assert not module.split_qkv
        assert module._q_linear is None
        assert module._k_linear is None
        assert module._v_linear is None
        assert module._qkv_linear is not None

        # Check that weights were correctly merged
        expected_weight = th.cat([original_w_q, original_w_k, original_w_v], dim=0)
        th.testing.assert_close(module._qkv_linear.weight.data, expected_weight)

        if bias:
            expected_bias = th.cat([original_b_q, original_b_k, original_b_v], dim=0)
            th.testing.assert_close(module._qkv_linear.bias.data, expected_bias)

    @given(in_features=in_features, bias=bias_flags)
    @settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
    def test_load_state_dict_auto_switch(self, in_features: int, bias: bool):
        """Test that load_state_dict automatically switches split_qkv when needed."""
        # Create source module in split mode and get its state dict
        source_split = QKVLinear(in_features=in_features, bias=bias, split_qkv=True)
        state_dict_split = source_split.state_dict()

        # Create target module in merged mode
        target_merged = QKVLinear(in_features=in_features, bias=bias, split_qkv=False)

        # This should trigger the auto-switch logic
        target_merged.load_state_dict(state_dict_split, strict=True)

        # Should have switched to split mode
        assert target_merged.split_qkv

        # Now test the reverse
        source_merged = QKVLinear(in_features=in_features, bias=bias, split_qkv=False)
        state_dict_merged = source_merged.state_dict()

        target_split = QKVLinear(in_features=in_features, bias=bias, split_qkv=True)
        target_split.load_state_dict(state_dict_merged, strict=True)

        # Should have switched to merged mode
        assert not target_split.split_qkv

    @given(in_features=in_features, bias=bias_flags)
    @settings(max_examples=10, suppress_health_check=[HealthCheck.too_slow], deadline=10_000)
    def test_round_trip_conversion(self, in_features: int, bias: bool):
        """Test that converting back and forth preserves functionality."""
        module = QKVLinear(in_features=in_features, bias=bias, split_qkv=False)

        # Initialize and create test input
        module.init_weights()
        x = th.randn(2, in_features)

        # Get output in merged mode
        output_merged = module(x)

        # Convert to split and get output
        module.set_split_qkv(True)

        output_split = module(x)

        # Convert back to merged and get output
        module.set_split_qkv(False)
        output_merged_again = module(x)

        # All outputs should be equivalent
        if isinstance(output_merged, th.Tensor) and isinstance(output_split, tuple):
            # For comparison, concatenate the split outputs
            output_split_cat = th.cat(output_split, dim=-1)
            th.testing.assert_close(output_merged, output_split_cat)
            th.testing.assert_close(output_merged, output_merged_again)


# Additional tests for edge cases
class TestQKVLinearEdgeCases:
    """Tests for edge cases and error conditions."""

    @given(in_features=st.integers(3, 100).filter(lambda x: x % 3 != 0))
    @settings(max_examples=10)
    def test_set_split_invalid_merged_features(self, in_features: int):
        """Test error when merged projection output features not divisible by 3."""
        # This is a bit tricky to test since the module ensures divisibility by 3
        # We'll create a malicious state that violates the assumption
        module = QKVLinear(in_features=in_features, bias=False, split_qkv=False)

        # Manually create invalid weight matrix
        with th.no_grad():
            module._qkv_linear.weight.data = th.randn(in_features * 3 + 1, in_features)

        # This should raise an error when trying to split
        with pytest.raises(ValueError, match="divisible by 3"):
            module.set_split_qkv(True)

    @given(in_features=st.integers(1, 50))
    @settings(max_examples=10)
    def test_set_split_inconsistent_bias(self, in_features: int):
        """Test error when q/k/v have inconsistent bias presence."""
        # This would require manually modifying the internal state
        # since the constructor ensures consistency
        in_features = 16
        module = QKVLinear(in_features=in_features, bias=True, split_qkv=True)

        # Manually create inconsistent bias state
        module._q_linear.bias = None

        with pytest.raises(ValueError, match="Bias presence inconsistent"):
            module.set_split_qkv(False)
