import pytest
import torch as th

from ml_utils.components.attention import SelfAttentionConfig
from ml_utils.components.transformer import (
    TransformerEncoder,
    TransformerEncoderBlockConfig,
    TransformerEncoderConfig,
)

# Currently not using hypothesis for these tests, but could be useful in the future.


# Helper functions to construct model and generate dummy data
def _construct_transformer_encoder(
    nheads: int = 4,
    head_dim: int = 16,
    num_layers: int = 2,
    num_registers: int = 0,
    context_dim: int = 0,
):
    config = TransformerEncoderConfig(
        num_layers=num_layers,
        num_registers=num_registers,
        transformer_config=TransformerEncoderBlockConfig(
            SelfAttentionConfig(
                nheads=nheads,
            )
        ),
    )
    model = TransformerEncoder(nheads * head_dim, config, context_dim)
    # Only going to test on gpu, cpu would be too slow.
    return model.to("cuda")


def _generate_dummy_sample(dim: int = 64):
    x = th.randn(10, dim)
    cu_seqlens = th.tensor([0, 2, 7, 10], dtype=th.int32, device="cuda")
    return x.to("cuda"), cu_seqlens


def _generate_dummy_sample_with_context(dim: int = 64, context_dim: int = 16):
    x = th.randn(10, dim)
    cu_seqlens = th.tensor([0, 2, 7, 10], dtype=th.int32, device="cuda")
    context = th.randn(3, context_dim)
    return x.to("cuda"), cu_seqlens, context.to("cuda")


# Test cases
def test_flash_attention_toggle():
    # Test that toggling flash attention on the encoder updates all layers.
    model = _construct_transformer_encoder()

    model.use_flash_attention = True
    uses_flash_attention = [layer.use_flash_attention for layer in model._layers]
    assert all(uses_flash_attention)

    model.use_flash_attention = False
    uses_flash_attention = [layer.use_flash_attention for layer in model._layers]
    assert not any(uses_flash_attention)


def test_transformer_encoder():
    if not th.cuda.is_available():
        pytest.skip("No cuda available")
    model = _construct_transformer_encoder()

    x, cu_seqlens = _generate_dummy_sample(64)

    out, cu_seqlens_out, _ = model(x, cu_seqlens)
    assert out.shape == (10, 64)
    assert th.equal(cu_seqlens, cu_seqlens_out)


def test_transformer_encoder_shape_with_registers():
    if not th.cuda.is_available():
        pytest.skip("No cuda available")
    model_with_registers = _construct_transformer_encoder(num_registers=2)
    model_without_registers = _construct_transformer_encoder(num_registers=0)

    x, cu_seqlens = _generate_dummy_sample(64)

    out_with_registers, cu_seqlens_out_with_registers, _ = model_with_registers(
        x, cu_seqlens
    )
    out_without_registers, cu_seqlens_out_without_registers, _ = (
        model_without_registers(x, cu_seqlens)
    )
    assert out_with_registers.shape == out_without_registers.shape

    # Check that cu_seqlens are unchanged
    assert th.equal(cu_seqlens_out_with_registers, cu_seqlens_out_without_registers)
    assert th.equal(cu_seqlens, cu_seqlens_out_with_registers)


def test_transformer_encoder_with_context():
    if not th.cuda.is_available():
        pytest.skip("No cuda available")
    model_with_context = _construct_transformer_encoder(context_dim=16)
    model_without_context = _construct_transformer_encoder(context_dim=0)

    x, cu_seqlens, context = _generate_dummy_sample_with_context(64, 16)

    out_with_context, cu_seqlens_with_context, _ = model_with_context(
        x, cu_seqlens, context=context
    )
    out_without_context, cu_seqlens_without_context, _ = model_without_context(
        x, cu_seqlens
    )
    assert out_with_context.shape == out_without_context.shape
    # Check that cu_seqlens are unchanged
    assert th.equal(cu_seqlens_with_context, cu_seqlens_without_context)
    assert th.equal(cu_seqlens, cu_seqlens_with_context)


def test_transformer_context_mismatch():
    if not th.cuda.is_available():
        pytest.skip("No cuda available")
    model_without_context = _construct_transformer_encoder(context_dim=0)

    x, cu_seqlens, context = _generate_dummy_sample_with_context(64, 16)

    with pytest.raises(ValueError):
        model_without_context(x, cu_seqlens, context=context)
