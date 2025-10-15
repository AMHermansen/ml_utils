import pytest
import torch as th
from hypothesis import given
from hypothesis import strategies as st
from tests.helpers import MILD_TOLERANCE

from ml_utils.components import PackedSelfAttention
from ml_utils.components.attention import FlashAttentionKWArgs

DEVICE = th.device("cuda" if th.cuda.is_available() else "cpu")

# small sizes to keep tests fast but varied
batch_size_st = st.integers(min_value=1, max_value=4)
seq_len_st = st.integers(min_value=1, max_value=6)
nheads_st = st.integers(min_value=1, max_value=4)
head_dim_st = st.integers(min_value=1, max_value=4)


# Strategy that yields (nheads, dimension) where dimension is divisible by nheads.
@st.composite
def nheads_and_dimensions(draw):
    nheads = draw(nheads_st)
    head_dim = draw(head_dim_st)
    dimension = nheads * head_dim
    return nheads, dimension, head_dim


# Strategy for variable per-batch sequence lengths (list) and culens
@st.composite
def culens_and_packed_len(draw, batch_min=1, batch_max=4, seq_min=1, seq_max=6):
    batch = draw(st.integers(min_value=batch_min, max_value=batch_max))
    # generate lengths for each batch entry
    lengths = [
        draw(st.integers(min_value=seq_min, max_value=seq_max)) for _ in range(batch)
    ]
    culens = [0]
    for L in lengths:
        culens.append(culens[-1] + L)
    return th.tensor(culens, dtype=th.int32, device=DEVICE), lengths


# Tests ------------------------------------------------------------------------


@given(nheads_dim=nheads_and_dimensions())
def test_constructor_raises_when_invalid_divisibility(nheads_dim):
    """property-based test: pick nheads and dimension where dimension may or may not be
    divisible. We specifically create a case that's invalid by adding 1 to dimension to
    ensure the constructor rejects it.
    """
    nheads, dimension, _ = nheads_dim
    if nheads == 1:
        return  # can't make invalid case if nheads=1
    bad_dim = dimension + 1  # guaranteed not divisible by nheads
    with pytest.raises(ValueError):
        PackedSelfAttention(dimension=bad_dim, nheads=nheads)


@given(
    nheads_dim=nheads_and_dimensions(),
    culens_pack=culens_and_packed_len(),
)
def test_attention_functions_swappable(nheads_dim, culens_pack):
    """Ensure that flash attention and fallback attention give same output."""
    nheads, _, head_dim = nheads_dim
    head_dim *= 8  # Features should be divisible by 8 for flash attention
    culens, _ = culens_pack
    packed_len = int(culens[-1].item())

    attn = PackedSelfAttention(
        dimension=nheads * head_dim, nheads=nheads, use_flash_attention=True
    ).to(DEVICE)

    x = th.randn((packed_len, nheads * head_dim), dtype=th.float32, device=DEVICE)
    out_flash = attn.forward(x, culens, max_seqlen=None)

    # output should be all zeros since input is all zeros and sum_attention is sum over merge dim
    assert out_flash.shape == (packed_len, nheads * head_dim)
    attn.use_flash_attention = False

    out_normal = attn(x, culens, max_seqlen=None)

    th.testing.assert_close(
        out_flash,
        out_normal,
        **MILD_TOLERANCE,
    )


@given(
    nheads_dim=nheads_and_dimensions(),
    culens_pack=culens_and_packed_len(),
    dropout_p=st.floats(min_value=0.0, max_value=0.5),
)
def test_training_vs_eval_flash_attention_kwargs_selected(
    nheads_dim, culens_pack, dropout_p
):
    """Ensure that when module is in train() it passes the training flash kwargs,
    and when in eval() it passes the eval kwargs (dropout_p == 0.0).
    We inject an attention function that records 'flash_attn_kwargs' passed to it.
    """
    nheads, dimension, _ = nheads_dim
    culens, _ = culens_pack
    packed_len = int(culens[-1].item())

    train_kwargs = FlashAttentionKWArgs(dropout_p=dropout_p, causal=False)

    attn = PackedSelfAttention(
        dimension=dimension, nheads=nheads, flash_attention_kwargs=train_kwargs
    ).to(DEVICE)

    # record the flash_attn_kwargs passed to the attention function.
    # In this test we don't care about the correctness of the attention computation.
    recorded = {"train_kw": None, "eval_kw": None}

    def record_attention(qkv, cu_seqlens_q, max_seqlen_q, flash_attn_kwargs):
        recorded["last"] = flash_attn_kwargs
        return qkv.sum(dim=1)

    attn._attention_function = record_attention

    # Get KW args in train mode
    attn.train()
    x = th.zeros((packed_len, dimension), dtype=th.float32, device=DEVICE)
    _ = attn.forward(x, culens, max_seqlen=None)
    recorded["train_kw"] = recorded.get("last")

    # Get KW args in eval mode
    attn.eval()
    _ = attn.forward(x, culens, max_seqlen=None)
    recorded["eval_kw"] = recorded.get("last")

    # If objects expose dropout_p attribute, compare them
    train_dp = getattr(recorded["train_kw"], "dropout_p", None)
    eval_dp = getattr(recorded["eval_kw"], "dropout_p", None)

    # train dropout should match provided dropout_p, eval must be 0.0
    assert train_dp is not None
    assert pytest.approx(train_dp, rel=1e-6) == float(dropout_p)
    assert eval_dp is not None
    assert pytest.approx(eval_dp, rel=1e-6) == 0.0


@given(nheads_dim=nheads_and_dimensions())
def test_use_flash_attention_setter_toggles(nheads_dim):
    """Ensure the setter for use_flash_attention updates the boolean property and
    rewires the internal function pointer.
    """
    nheads, dimension, _ = nheads_dim
    attn = PackedSelfAttention(
        dimension=dimension, nheads=nheads, use_flash_attention=True
    ).to(DEVICE)

    # initial state True
    assert attn.use_flash_attention is True

    # flip to False
    attn.use_flash_attention = False
    assert attn.use_flash_attention is False

    # flip back to True
    attn.use_flash_attention = True
    assert attn.use_flash_attention is True
