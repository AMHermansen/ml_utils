import unittest

from hypothesis import given
from hypothesis import strategies as st

from ml_utils.torch_utils.misc import recurse_and_apply


class TestRecurseAndApply(unittest.TestCase):
    @given(st.lists(st.integers()))
    def test_apply_to_list_elements(self, int_list):
        """Test applying function to all elements in a flat list."""

        def increment(x):
            return x + 1

        result = recurse_and_apply(int_list, increment, int)
        expected = [x + 1 for x in int_list]
        assert result == expected

    @given(st.tuples(st.integers(), st.integers(), st.integers()))
    def test_apply_to_tuple_elements(self, int_tuple):
        """Test applying function to all elements in a flat tuple."""

        def increment(x):
            return x + 1

        result = recurse_and_apply(int_tuple, increment, int)
        expected = tuple(x + 1 for x in int_tuple)
        assert result == expected

    @given(st.dictionaries(st.text(), st.integers()))
    def test_apply_to_dict_values(self, int_dict):
        """Test applying function to all values in a flat dict."""

        def increment(x):
            return x + 1

        result = recurse_and_apply(int_dict, increment, int)
        expected = {k: v + 1 for k, v in int_dict.items()}
        assert result == expected

    @given(st.lists(st.lists(st.integers())))
    def test_nested_lists(self, nested_list):
        """Test applying function to integers in nested lists."""

        def increment(x):
            return x + 1

        result = recurse_and_apply(nested_list, increment, int)

        # Manually build expected result
        def build_expected(lst):
            if isinstance(lst, list):
                return [build_expected(item) for item in lst]
            if isinstance(lst, int):
                return lst + 1
            return lst

        expected = build_expected(nested_list)
        assert result == expected

    @given(
        st.recursive(
            st.one_of(st.integers(), st.text()),
            lambda children: st.lists(children)
            | st.tuples(children)
            | st.dictionaries(st.text(), children),
            max_leaves=20,
        )
    )
    def test_complex_nested_structures(self, nested_structure):
        """Test with complex nested structures of lists, tuples, and dicts."""

        def stringify(x):
            return f"processed_{x}"

        # Only apply to strings
        result = recurse_and_apply(nested_structure, stringify, str)

        # Verify structure is preserved and only strings are processed
        def verify_structure(original, processed):
            if isinstance(original, list):
                assert isinstance(processed, list)
                assert len(original) == len(processed)
                for o, p in zip(original, processed):
                    verify_structure(o, p)
            elif isinstance(original, tuple):
                assert isinstance(processed, tuple)
                assert len(original) == len(processed)
                for o, p in zip(original, processed):
                    verify_structure(o, p)
            elif isinstance(original, dict):
                assert isinstance(processed, dict)
                assert set(original.keys()) == set(processed.keys())
                for key in original:
                    verify_structure(original[key], processed[key])
            elif isinstance(original, str):
                assert processed == f"processed_{original}"
            else:
                assert processed == original

        verify_structure(nested_structure, result)

    def test_target_type_selection(self):
        """Test that function is only applied to specified target type."""
        test_data = [1, "hello", 2.5, [3, "world", 4.2]]

        # Only apply to integers
        def double(x):
            return x * 2

        result = recurse_and_apply(test_data, double, int)
        expected = [2, "hello", 2.5, [6, "world", 4.2]]
        assert result == expected

    @given(st.lists(st.integers()))
    def test_preserves_list_structure(self, int_list):
        """Test that list structure is preserved."""

        def identity(x):
            return x

        result = recurse_and_apply(int_list, identity, int)
        assert isinstance(result, list)
        assert len(result) == len(int_list)
        assert result == int_list

    @given(st.tuples(st.integers(), st.text(), st.floats(allow_nan=False)))
    def test_preserves_tuple_structure(self, mixed_tuple):
        """Test that tuple structure is preserved."""
        # Need to exclude nan, since nan != nan...

        def identity(x):
            return x

        result = recurse_and_apply(mixed_tuple, identity, int)
        assert isinstance(result, tuple)
        assert len(result) == len(mixed_tuple)
        # Only integers should be processed, but structure preserved
        assert result[1] == mixed_tuple[1]  # string unchanged
        assert result[2] == mixed_tuple[2]  # float unchanged

    @given(st.dictionaries(st.text(), st.one_of(st.integers(), st.text())))
    def test_preserves_dict_structure(self, mixed_dict):
        """Test that dict structure is preserved."""

        def identity(x):
            return x

        result = recurse_and_apply(mixed_dict, identity, int)
        assert isinstance(result, dict)
        assert set(result.keys()) == set(mixed_dict.keys())
        # Structure and keys should be preserved

    def test_multiple_target_types(self):
        """Test applying function to multiple target types."""
        test_data = [1, "hello", 2.5, [3, "world"]]

        def process(x):
            return f"processed_{x}"

        # Apply to both integers and strings
        result = recurse_and_apply(test_data, process, (int, str))
        expected = [
            "processed_1",
            "processed_hello",
            2.5,
            ["processed_3", "processed_world"],
        ]
        assert result == expected

    def test_empty_structures(self):
        """Test with empty lists, tuples, and dicts."""

        def increment(x):
            return x + 1

        # Empty list
        assert recurse_and_apply([], increment, int) == []

        # Empty tuple
        assert recurse_and_apply((), increment, int) == ()

        # Empty dict
        assert recurse_and_apply({}, increment, int) == {}

    @given(st.integers())
    def test_direct_target_type_match(self, value):
        """Test when the input itself is the target type."""

        def double(x):
            return x * 2

        result = recurse_and_apply(value, double, int)
        assert result == value * 2


if __name__ == "__main__":
    unittest.main()
