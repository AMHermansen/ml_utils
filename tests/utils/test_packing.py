import numpy as np
import pytest
import torch as th
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from ml_utils.torch_utils.packing import pack_tensors, unpack_tensors

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


# Tests ----------------------------------------------------------------------


@given(
    B=batch_sizes,
    N=lengths,
    F=feat_dims,
    data=st.data(),
)
def test_pack_unpack_roundtrip_single_tensor(B, N, F, data):
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

    arr = data.draw(
        hnp.arrays(dtype=np.float32, shape=(B, N, F), elements=float_elements),
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

    assert th.equal(unmask, mask)
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
