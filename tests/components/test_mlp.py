import pytest
import torch as th
import torch.nn as nn
from hypothesis import given, strategies as st, assume, settings
from hypothesis.strategies._internal.core import sampled_from

from ml_utils.components import MLPBlock, MLP, MLPBlockConfig, MLPContextConfig, MLPConfig


# Shared strategies for hypothesis
floats = st.floats(min_value=0.0, max_value=0.5, allow_nan=False, allow_infinity=False)
ints = st.integers(min_value=1, max_value=128)
bools = st.booleans()
activations = sampled_from(["SiLU", "ReLU", "GELU", "Tanh"])
norms = sampled_from([None, "LayerNorm", "BatchNorm1d"])


class TestMLPBlock:
    @given(
        in_features=ints,
        out_features=ints,
        context_features=ints,
        activation=activations,
        norm=norms,
        dropout=floats,
        do_residual=bools,
        no_bias=bools,
    )
    @settings(max_examples=50, deadline=None)
    def test_mlp_block_creation(
        self, in_features, out_features, context_features, activation,
        norm, dropout, do_residual, no_bias
    ):
        """Test that MLPBlock can be created with various configurations."""
        assume(not do_residual or in_features == out_features)

        config = MLPBlockConfig(
            activation=activation,
            norm=norm,
            dropout=dropout,
            do_residual=do_residual,
            no_bias=no_bias,
        )

        block = MLPBlock(
            in_features=in_features,
            out_features=out_features,
            config=config,
            context_features=context_features,
        )

        assert block.in_features == in_features
        assert block.out_features == out_features
        assert block.expects_context == (context_features > 0)

    @given(
        batch_size=ints,
        in_features=ints,
        out_features=ints,
        context_features=ints,
        do_residual=bools,
    )
    @settings(max_examples=30, deadline=None)
    def test_mlp_block_forward(
        self, batch_size, in_features, out_features, context_features, do_residual
    ):
        """Test forward pass with and without context."""
        assume(not do_residual or in_features == out_features)

        config = MLPBlockConfig(do_residual=do_residual)
        block = MLPBlock(
            in_features=in_features,
            out_features=out_features,
            config=config,
            context_features=context_features,
        )

        x = th.randn(batch_size, in_features)

        # Test without context when not expected
        if context_features == 0:
            output = block(x)
            assert output.shape == (batch_size, out_features)
        else:
            # Test with context
            context = th.randn(batch_size, context_features)
            output = block(x, context=context)
            assert output.shape == (batch_size, out_features)

            # Test context mismatch error
            with pytest.raises(ValueError, match="Cannot resolve dimensional mismatch"):
                wrong_context = th.randn(batch_size + 1, context_features)
                block(x, context=wrong_context)

    @given(in_features=ints, out_features=ints)
    @settings(max_examples=10, deadline=None)
    def test_mlp_block_context_validation(self, in_features, out_features):
        """Test context validation logic."""
        assume(in_features != out_features)  # Ensure no residual

        # Block that expects context
        block_with_context = MLPBlock(
            in_features=in_features,
            out_features=out_features,
            config=MLPBlockConfig(),
            context_features=16,
        )

        x = th.randn(4, in_features)

        # Should error when context expected but not provided
        with pytest.raises(ValueError, match="Context features expected"):
            block_with_context(x)

        # Should error when context provided but not expected
        block_no_context = MLPBlock(
            in_features=in_features,
            out_features=out_features,
            config=MLPBlockConfig(),
            context_features=0,
        )

        with pytest.raises(ValueError, match="Context features not expected"):
            block_no_context(x, context=th.randn(4, 16))


class TestMLP:
    @given(
        in_features=ints,
        out_features=ints,
        hidden_features=ints,
        num_layers=st.integers(min_value=2, max_value=8),
        context_features=ints,
    )
    @settings(max_examples=30, deadline=None)
    def test_mlp_creation_and_forward(
        self, in_features, out_features, hidden_features, num_layers, context_features
    ):
        """Test MLP creation and forward pass with various configurations."""
        assume(num_layers >= 2)  # Need at least input and output layers

        context_config = MLPContextConfig(
            context_features=context_features,
            apply_on_input=True,
            apply_on_hidden=True,
            apply_on_output=True,
        )
        config = MLPConfig(
            hidden_features=hidden_features,
            num_layers=num_layers,
            context_config=context_config,
        )

        mlp = MLP(
            in_features=in_features,
            out_features=out_features,
            config=config
        )

        # Setup layers (normally done before forward pass)
        mlp._setup_layers()

        batch_size = 4
        x = th.randn(batch_size, in_features)

        if context_features > 0:
            context = th.randn(batch_size, context_features)
            output = mlp(x, context=context)
        else:
            output = mlp(x)

        assert output.shape == (batch_size, out_features)
        assert mlp.in_features == in_features
        assert mlp.out_features == out_features

    @given(
        in_features=ints,
        out_features=ints,
        hidden_features=ints,
        context_features=ints,
        apply_on_input=bools,
        apply_on_hidden=bools,
        apply_on_output=bools,
    )
    @settings(max_examples=30, deadline=None)
    def test_mlp_context_application(
        self, in_features, out_features, hidden_features, context_features,
        apply_on_input, apply_on_hidden, apply_on_output
    ):
        """Test that context is applied only to the specified layers."""
        assume(context_features > 0)
        assume(apply_on_input or apply_on_hidden or apply_on_output)  # At least one must be True

        context_config = MLPContextConfig(
            context_features=context_features,
            apply_on_input=apply_on_input,
            apply_on_hidden=apply_on_hidden,
            apply_on_output=apply_on_output,
        )

        config = MLPConfig(
            hidden_features=hidden_features,
            num_layers=3,  # Fixed to ensure we have hidden layers to test
            context_config=context_config,
        )
        mlp = MLP(
            in_features=in_features,
            out_features=out_features,
            config=config
        )
        mlp._setup_layers()

        # Verify each block has the correct context expectations
        assert mlp._in_block.expects_context == apply_on_input
        assert all(
            block.expects_context == apply_on_hidden
                for block in mlp._hidden_blocks
        )
        assert mlp._out_block.expects_context == apply_on_output

        # Test forward pass with context to ensure it works end-to-end
        batch_size = 4
        x = th.randn(batch_size, in_features)
        context = th.randn(batch_size, context_features)

        output = mlp(x, context=context)
        assert output.shape == (batch_size, out_features)

    @given(context_features=ints)
    @settings(max_examples=10, deadline=None)
    def test_mlp_context_config_validation(self, context_features):
        """Test MLPContextConfig validation."""
        assume(context_features > 0)

        # Should raise error if context features provided but no application specified
        with pytest.raises(ValueError, match="At least one of"):
            MLPContextConfig(
                context_features=context_features,
                apply_on_input=False,
                apply_on_hidden=False,
                apply_on_output=False,
            )


class TestEdgeCases:
    @given(in_features=ints, out_features=ints)
    @settings(max_examples=10, deadline=None)
    def test_residual_connection_validation(self, in_features, out_features):
        """Test that residual connection requires matching dimensions."""
        if in_features != out_features:
            with pytest.raises(ValueError, match="in_features must be equal"):
                MLPBlock(
                    in_features=in_features,
                    out_features=out_features,
                    config=MLPBlockConfig(do_residual=True),
                )
        else:
            # Should work when dimensions match
            block = MLPBlock(
                in_features=in_features,
                out_features=out_features,
                config=MLPBlockConfig(do_residual=True),
            )
            assert block._do_residual

    def test_custom_activation_and_norm(self):
        """Test with custom activation and normalization modules."""
        config = MLPBlockConfig(
            activation=nn.ELU(),
            norm=nn.LayerNorm(32),
        )

        block = MLPBlock(
            in_features=16,
            out_features=32,
            config=config,
        )

        x = th.randn(4, 16)
        output = block(x)
        assert output.shape == (4, 32)
