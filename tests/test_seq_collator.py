import pytest
import torch as th
from hypothesis import given
from hypothesis import strategies as st

from ml_utils.data_utils import CumulativeSeqlengthCollator


@st.composite
def batch_list_strategy(draw):
    num_seqlen_keys = draw(st.integers(min_value=1, max_value=3))
    feature_dim = draw(st.integers(min_value=1, max_value=8))
    batch_count = draw(st.integers(min_value=1, max_value=5))
    label_dim = draw(st.integers(min_value=1, max_value=4))
    batches = []
    for _ in range(batch_count):
        batch = {}
        for _key in range(num_seqlen_keys):
            seq_len = draw(st.integers(min_value=1, max_value=10))
            batch[f"features{_key}"] = th.randn(seq_len, feature_dim)
            batch[f"label{_key}"] = th.randint(0, 10, (label_dim,))
            batch[f"cu_seqlen{_key}"] = th.tensor([seq_len])
        batches.append(batch)
    return batches


@given(batches=batch_list_strategy())
def test_cumulative_seqlength_collator_hypothesis(batches):
    collator = CumulativeSeqlengthCollator(
        seqlen_variable_names=[key for key in batches[0] if key.startswith("features")],
        seqlen_keys=[key for key in batches[0] if key.startswith("cu_seqlen")],
    )
    result = collator(batches)

    # Check features shape
    for key in result:
        if key.startswith("features"):
            total_seq_len = sum(batch[key].shape[0] for batch in batches)
            feature_dim = batches[0][key].shape[1]
            assert result[key].shape == (total_seq_len, feature_dim)

    # Check cu_seqlens are cumulative
    for key in result:
        if key.startswith("cu_seqlen"):
            cumulative_lengths = 0
            for idx, batch in enumerate(batches):
                cumulative_lengths += batch[key].item()
                produced_cumulative_lengths = result[key][
                    idx + 1
                ].item()  # +1 for fencepost
                assert cumulative_lengths == produced_cumulative_lengths
            assert len(result[key]) == len(batches) + 1  # Fencepost

    # Check cu_seqlen shape
    for key in result:
        if key.startswith("cu_seqlen"):
            assert result[key].shape == (len(batches) + 1,)

    # Check labels
    for key in result:
        if key.startswith("label"):
            labels = th.stack([batch[key] for batch in batches], dim=0)
            assert th.equal(result[key], labels)


def test_cumulative_seqlength_collator_missing_feature():
    collator = CumulativeSeqlengthCollator(
        seqlen_variable_names=["features", "missing_feature"]
    )
    batch = {
        "features": th.randn(3, 5),
        "cu_seqlen": th.tensor([3]),
        "label": th.tensor([0]),
    }
    batches = [batch]
    with pytest.raises(
        ValueError, match="All batches must contain the key 'missing_feature'"
    ):
        collator(batches)
