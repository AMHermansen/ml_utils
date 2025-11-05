import numpy as np
import pytest
import torch as th
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from ml_utils.torch_utils.packing import pack_tensors, unpack_tensors, pack_tensor, unpack_tensor

# Helpers & strategies -------------------------------------------------------

# Strategy for sizes: limit sizes so tests run quickly but explore variety
batch_sizes = st.integers(min_value=1, max_value=5)
lengths = st.integers(min_value=1, max_value=8)
feat_dims = st.integers(min_value=1, max_value=6)

# Floating point elements to fill arrays
float_elements = st.floats(
    min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False
)


# Convert numpy arrays from Hypothesis into torch tensors
def np_to_torch(a: np.ndarray):
    return th.from_numpy(a).to(th.float32)


# Strategy to generate cu_seqlens and a corresponding packed tensor (T, *feat_shape)
@st.composite
def cu_seqlens_and_packed(draw, batch_min=1, batch_max=4, len_min=1, len_max=6):
    # batch size
    B = draw(st.integers(min_value=batch_min, max_value=batch_max))
    # per-example lengths
    lengths = [
        draw(st.integers(min_value=len_min, max_value=len_max)) for _ in range(B)
    ]
    cu = [0]
    for L in lengths:
        cu.append(cu[-1] + L)
    T = cu[-1]

    # choose number of extra trailing dims (0..2) and sizes
    extra_ndims = draw(st.integers(min_value=0, max_value=2))
    extra_dims = tuple(
        draw(st.integers(min_value=1, max_value=4)) for _ in range(extra_ndims)
    )
    # final feature dim (last axis)
    F = draw(st.integers(min_value=1, max_value=5))

    # packed shape is (T, *extra_dims, F)
    shape = (T,) + extra_dims + (F,)

    # draw numpy array for packed values
    arr = draw(
        hnp.arrays(
            dtype=np.float32,
            shape=shape,
            elements=st.floats(
                min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False
            ),
        )
    )
    cu_tensor = th.tensor(cu, dtype=th.int32)
    packed_tensor = th.from_numpy(arr)
    return cu_tensor, packed_tensor


@st.composite
def cu_seqlens_and_two_packed(draw, **kwargs):
    cu, p1 = draw(cu_seqlens_and_packed(**kwargs))
    # generate a second packed with same shape as p1
    shape = tuple(p1.shape)
    arr2 = draw(
        hnp.arrays(
            dtype=np.float32,
            shape=shape,
            elements=st.floats(
                min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False
            ),
        )
    )
    p2 = th.from_numpy(arr2)
    return cu, p1, p2


# Strategy: generate cu_seqlens and a list of packed tensors with possibly different trailing shapes
@st.composite
def cu_seqlens_and_packed_list(
    draw,
    batch_min=1,
    batch_max=4,
    len_min=1,
    len_max=6,
    n_tensors_min=1,
    n_tensors_max=4,
):
    B = draw(st.integers(min_value=batch_min, max_value=batch_max))
    lengths = [
        draw(st.integers(min_value=len_min, max_value=len_max)) for _ in range(B)
    ]
    cu = [0]
    for L in lengths:
        cu.append(cu[-1] + L)
    T = cu[-1]

    extra_ndims_choice = st.integers(min_value=0, max_value=2)
    n_tensors = draw(st.integers(min_value=n_tensors_min, max_value=n_tensors_max))

    packed_tensors = []
    for _ in range(n_tensors):
        extra_ndims = draw(extra_ndims_choice)
        extra_dims = tuple(
            draw(st.integers(min_value=1, max_value=4)) for _ in range(extra_ndims)
        )
        F = draw(st.integers(min_value=1, max_value=5))
        shape = (T,) + extra_dims + (F,)
        arr = draw(
            hnp.arrays(
                dtype=np.float32,
                shape=shape,
                elements=st.floats(
                    min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False
                ),
            )
        )
        packed_tensors.append(th.from_numpy(arr))
    cu_tensor = th.tensor(cu, dtype=th.int32)
    return cu_tensor, packed_tensors


# Tests ----------------------------------------------------------------------


@given(
    B=batch_sizes,
    N=lengths,
    F=feat_dims,
    feature_like_dimensions=st.integers(min_value=1, max_value=5),
    data=st.data(),
)
@settings(deadline=10_000)
def test_pack_unpack_roundtrip_single_tensor(B, N, F, feature_like_dimensions, data):
    """
    Generate a random mask and a random (B,N,F) tensor, pack it and then unpack.
    After round-trip, the unpacked batched tensor should match the original at mask positions,
    and be zero where mask is False.
    """
    # draw mask and data arrays with Hypothesis
    mask_np = data.draw(
        hnp.arrays(
            dtype=np.bool_,
            shape=(B, N),
            elements=st.booleans(),
        ),
        label="mask",
    )
    # make sure there is at least one True somewhere (so pack has non-zero length)
    if not mask_np.any():
        # flip one random position to True
        i = data.draw(st.integers(0, B - 1))
        j = data.draw(st.integers(0, N - 1))
        mask_np[i, j] = True

    feature_like = (F for _ in range(feature_like_dimensions))

    arr = data.draw(
        hnp.arrays(
            dtype=np.float32,
            shape=(B, N, *feature_like),
            elements=float_elements,
        ),
        label="data_array",
    )

    mask = th.from_numpy(mask_np)
    batched = th.from_numpy(arr)

    cu_seqlens, (packed,) = pack_tensors(mask, batched)

    # Check cu_seqlens is strictly increasing, length B+1, starts with 0
    assert cu_seqlens.shape[0] == B + 1
    assert int(cu_seqlens[0].item()) == 0
    diffs = th.diff(cu_seqlens)
    assert diffs.shape[0] == B
    # diffs sum should equal total number of True mask entries
    assert int(diffs.sum().item()) == int(mask.sum().item())

    # Now unpack using the cu_seqlens produced by pack_tensors
    # NOTE: this asserts the correct behavior of unpack_tensors; if unpack_tensors contains a bug,
    # this test will fail — which is desirable to catch regressions.
    unmask, (unbatched,) = unpack_tensors(cu_seqlens, packed, max_length=None)

    # Masks should match
    assert th.equal(
        unmask[unmask], mask[mask]
    )  # We will allow different padding lengths

    # Where mask is True, data should be equal
    # Because unbatched may be float we compare with allclose
    assert th.allclose(unbatched[unmask], batched[mask], atol=1e-6, rtol=1e-6)

    # Where mask is False, unbatched should be zero (that's how unpack pads)
    assert th.all(unbatched[~unmask] == 0)


@given(
    B=batch_sizes,
    N=lengths,
    F=feat_dims,
    data=st.data(),
)
@settings(deadline=10_000)
def test_pack_unpack_roundtrip_multiple_tensors(B, N, F, data):
    """
    Test pack/unpack roundtrip with multiple tensors (two tensors with same shapes).
    """
    mask_np = data.draw(
        hnp.arrays(dtype=np.bool_, shape=(B, N), elements=st.booleans()),
        label="mask_multi",
    )
    if not mask_np.any():
        i = data.draw(st.integers(0, B - 1))
        j = data.draw(st.integers(0, N - 1))
        mask_np[i, j] = True

    arr1 = data.draw(
        hnp.arrays(dtype=np.float32, shape=(B, N, F), elements=float_elements)
    )
    arr2 = data.draw(
        hnp.arrays(dtype=np.float32, shape=(B, N, F), elements=float_elements)
    )

    mask = th.from_numpy(mask_np)
    batched1 = th.from_numpy(arr1)
    batched2 = th.from_numpy(arr2)

    cu_seqlens, (packed1, packed2) = pack_tensors(mask, batched1, batched2)

    unmask, (unbatched1, unbatched2) = unpack_tensors(cu_seqlens, packed1, packed2)

    assert th.equal(unmask[unmask], mask[mask])
    assert th.allclose(unbatched1[unmask], batched1[mask])
    assert th.allclose(unbatched2[unmask], batched2[mask])
    assert th.all(unbatched1[~unmask] == 0)
    assert th.all(unbatched2[~unmask] == 0)


@given(
    B=batch_sizes,
    N=lengths,
    F=feat_dims,
    data=st.data(),
)
def test_pack_raises_on_1d_tensor(B, N, F, data):
    """
    pack_tensors should raise ValueError if a tensor with fewer than 2 dims is passed.
    """
    mask_np = data.draw(
        hnp.arrays(dtype=np.bool_, shape=(B, N), elements=st.booleans())
    )
    if not mask_np.any():
        mask_np[0, 0] = True

    mask = th.from_numpy(mask_np)

    # create a 1D tensor of length B (invalid)
    bad = th.rand(B)

    with pytest.raises(ValueError):
        pack_tensors(mask, bad)


@given(
    B=batch_sizes,
    N=lengths,
    F=feat_dims,
    data=st.data(),
)
def test_pack_raises_on_shape_mismatch(B, N, F, data):
    """
    pack_tensors should raise ValueError if mask's first two dims don't match tensor's first two dims.
    """
    mask_np = data.draw(
        hnp.arrays(dtype=np.bool_, shape=(B, N), elements=st.booleans())
    )
    if not mask_np.any():
        mask_np[0, 0] = True
    mask = th.from_numpy(mask_np)

    # Create a tensor with mismatching batch or length
    wrong_B = B + 1
    wrong_arr = data.draw(
        hnp.arrays(dtype=np.float32, shape=(wrong_B, N, F), elements=float_elements)
    )
    wrong = th.from_numpy(wrong_arr)

    with pytest.raises(ValueError):
        pack_tensors(mask, wrong)


def test_unpack_max_length_padding_behavior():
    """
    Explicit test: ensure that specifying max_length pads to that length, and mask is generated accordingly.
    """
    # small deterministic example
    mask = th.tensor([[True, True, False], [True, False, False]])
    batched = th.tensor([[[1.0], [2.0], [0.0]], [[3.0], [0.0], [0.0]]])  # shape (2,3,1)

    cu_seqlens, (packed,) = pack_tensors(mask, batched)
    # Force a larger max_length than max actual length (which is 2 here)
    max_len = 4

    unmask, (unbatched,) = unpack_tensors(cu_seqlens, packed, max_length=max_len)

    assert unmask.shape == (2, max_len)
    # True counts per batch remain correct
    assert unmask.sum(dim=1).tolist() == [2, 1]
    # Values at true positions match original
    assert th.allclose(unbatched[unmask], packed)
    # Padded positions (beyond real length) should be zero
    assert th.all(unbatched[~unmask] == 0)


@given(cu_and_packed=cu_seqlens_and_packed())
def test_unpack_then_pack_roundtrip_single_tensor(cu_and_packed):
    """
    Generate (cu_seqlens, packed) -> unpack -> pack -> assert returned cu_seqlens and packed match original.
    """
    cu_seqlens, packed = cu_and_packed
    # unpack -> returns mask (B, L) and batched (B, L, *feat)
    mask, (batched,) = unpack_tensors(cu_seqlens, packed, max_length=None)

    # pack the unpacked batched using the returned mask
    cu2, (packed2,) = pack_tensors(mask, batched)

    # cu_seqlens should match exactly
    assert th.equal(cu_seqlens, cu2)

    # packed shapes must match and values must match (allow tiny fp tolerance)
    assert packed.shape == packed2.shape
    assert th.allclose(packed, packed2, rtol=1e-6, atol=1e-6)


@given(cu_and_packed=cu_seqlens_and_packed())
def test_unpack_then_pack_roundtrip_with_explicit_max_length(cu_and_packed):
    """
    When unpack is called with a larger explicit max_length, packing the unpacked result
    should still reproduce the canonical cu_seqlens and packed tensors.
    """
    cu_seqlens, packed = cu_and_packed
    # compute original per-example lengths and maximum
    lengths = th.diff(cu_seqlens).tolist()
    orig_max = int(max(lengths)) if len(lengths) > 0 else 0
    # choose a larger max_length (orig_max + 1..orig_max+3) if orig_max > 0, else 1
    add = 1
    max_length = orig_max + add if orig_max > 0 else 1

    mask, (batched_un,) = unpack_tensors(cu_seqlens, packed, max_length=max_length)
    cu2, (packed2,) = pack_tensors(mask, batched_un)

    assert th.equal(cu_seqlens, cu2)
    assert packed.shape == packed2.shape
    assert th.allclose(packed, packed2, rtol=1e-6, atol=1e-6)


@given(cu_and_packed=cu_seqlens_and_two_packed())
def test_unpack_then_pack_roundtrip_multiple_tensors(cu_and_packed):
    """
    For two packed tensors sharing the same cu_seqlens, unpack then pack should reproduce both packed tensors.
    """
    cu_seqlens, packed1, packed2 = cu_and_packed
    mask, (batched1, batched2) = unpack_tensors(
        cu_seqlens, packed1, packed2, max_length=None
    )
    assert len(mask.shape) == 2
    cu2, (packed1_2, packed2_2) = pack_tensors(mask, batched1, batched2)

    assert th.equal(cu_seqlens, cu2)

    assert packed1.shape == packed1_2.shape
    assert packed2.shape == packed2_2.shape

    assert th.allclose(packed1, packed1_2, rtol=1e-6, atol=1e-6)
    assert th.allclose(packed2, packed2_2, rtol=1e-6, atol=1e-6)


def test_small_deterministic_unpack_pack():
    """
    Deterministic small example sanity check.
    """
    # two examples: lengths 3 and 2 -> cu_seqlens [0,3,5]
    cu = th.tensor([0, 3, 5], dtype=th.int32)
    packed = th.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])  # shape (5,1)

    mask, (batched,) = unpack_tensors(cu, packed, max_length=None)
    cu2, (packed2,) = pack_tensors(mask, batched)

    assert th.equal(cu, cu2)
    assert packed.shape == packed2.shape
    assert th.allclose(packed, packed2)


@given(cu_and_packed_list=cu_seqlens_and_packed_list())
def test_unpack_then_pack_roundtrip_varied_tensors(cu_and_packed_list):
    """
    For multiple packed tensors that share the same cu_seqlens but have different trailing shapes,
    unpack -> pack should reproduce the original cu_seqlens and all packed tensors.
    """
    cu_seqlens, packed_list = cu_and_packed_list
    # Unpack all packed tensors together
    mask, batched_list = unpack_tensors(cu_seqlens, *packed_list, max_length=None)

    # Pack back using the mask returned by unpack_tensors
    cu2, packed_list_2 = pack_tensors(mask, *batched_list)

    # cu_seqlens must match canonical
    assert th.equal(cu_seqlens, cu2)

    # Ensure same number of tensors and same shapes/values
    assert len(packed_list) == len(packed_list_2)
    for p_orig, p_new in zip(packed_list, packed_list_2):
        assert p_orig.shape == p_new.shape
        assert th.allclose(p_orig, p_new, rtol=1e-6, atol=1e-6)


# Additional tests to cover mismatched first-two-dimension errors ----------------


@given(
    B=st.integers(1, 4),
    N=st.integers(1, 6),
    F=st.integers(1, 5),
)
def test_pack_raises_on_mask_tensor_first_two_dim_mismatch(B, N, F):
    """
    pack_tensors should raise ValueError if mask's first two dims don't match a tensor's first two dims.
    """
    mask = th.zeros((B, N), dtype=th.bool)
    # Create a tensor with mismatching batch (B+1) dimension
    wrong = th.randn((B + 1, N, F))
    try:
        pack_tensors(mask, wrong)
        raise AssertionError(
            "pack_tensors did not raise ValueError on mismatched first two dims"
        )
    except ValueError:
        pass  # expected


@given(
    B=st.integers(1, 4),
    N=st.integers(1, 6),
    F=st.integers(1, 5),
)
def test_pack_accepts_tensors_with_different_trailing_shapes(B, N, F):
    """
    Verify pack_tensors accepts multiple tensors that only need to match in the first two dims
    and can have different trailing feature shapes.
    """
    mask = th.ones((B, N), dtype=th.bool)
    # two tensors with same (B,N) but different trailing shapes
    t1 = th.randn((B, N, F))
    t2 = th.randn((B, N, 2, F))
    cu, (p1, p2) = pack_tensors(mask, t1, t2)

    # Verify shapes: p1 -> (T, F), p2 -> (T, 2, F)
    T = int(mask.sum().item())
    assert p1.shape == (T, F)
    assert p2.shape == (T, 2, F)

    # Now unpack and pack back to ensure roundtrip works with different trailing shapes
    mask2, (u1, u2) = unpack_tensors(cu, p1, p2, max_length=None)
    cu2, (p1b, p2b) = pack_tensors(mask2, u1, u2)

    assert th.equal(cu, cu2)
    assert th.allclose(p1, p1b, rtol=1e-6, atol=1e-6)
    assert th.allclose(p2, p2b, rtol=1e-6, atol=1e-6)


# Test pack_tensor and unpack_tensor
# Just testing that they behave the same as pack_tensors and unpack_tensors for single
# tensors, and then extensive testing is already done above.


@given(cu_and_packed=cu_seqlens_and_packed())
def test_unpack_tensor_equals_unpack_tensors(cu_and_packed):
    """
    Test that unpack_tensor produces the same result as unpack_tensors for a single tensor.
    """
    cu_seqlens, packed = cu_and_packed
    print(cu_seqlens.shape)
    print(packed.shape)
    mask1, (batched1,) = unpack_tensors(cu_seqlens, packed, max_length=None)
    mask2, batched2 = unpack_tensor(cu_seqlens, packed, max_length=None)

    assert th.equal(mask1, mask2)
    assert th.allclose(batched1, batched2, rtol=1e-6, atol=1e-6)



@given(
    B=batch_sizes,
    N=lengths,
    F=feat_dims,
    data=st.data(),
)
def test_pack_tensor_equals_pack_tensors(B, N, F, data):
    """
    Test that pack_tensor produces the same result as pack_tensors for a single tensor.
    """
    mask_np = data.draw(
        hnp.arrays(dtype=np.bool_, shape=(B, N), elements=st.booleans())
    )
    if not mask_np.any():
        i = data.draw(st.integers(0, B - 1))
        j = data.draw(st.integers(0, N - 1))
        mask_np[i, j] = True

    arr = data.draw(
        hnp.arrays(dtype=np.float32, shape=(B, N, F), elements=float_elements)
    )

    mask = th.from_numpy(mask_np)
    batched = th.from_numpy(arr)

    print(mask.shape)
    print(batched.shape)

    cu1, (packed1,) = pack_tensors(mask, batched)
    cu2, packed2 = pack_tensor(mask, batched)

    assert th.equal(cu1, cu2)
    assert th.allclose(packed1, packed2, rtol=1e-6, atol=1e-6)
