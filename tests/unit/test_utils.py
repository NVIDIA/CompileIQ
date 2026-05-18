"""
Tests for nested dict utilities in compileiq/utils/helpers.py.
"""

from compileiq.utils.helpers import (
    flatten_nested_dict,
    restore_nested_search_space,
    _merge_nested_dictionaries,
    _decode_from_core,
)
import pytest


@pytest.fixture
def nested_params():
    return {
        "y": {"yy1": {"yyy1": {"yyyy1": 2}}, "yy2": 2, "yy3": 3, "yy4": {"yyy4": 5}},
        "x": {"xx1": 7.5, "xx2": 6.0},
        "z": 10,
    }


def test_flatten_nested_dict(nested_params):
    test_case = nested_params

    flat = flatten_nested_dict(test_case)

    assert all(not isinstance(value, dict) for value in flat.values())

    test_case = {
        "x": "val",
        "z": 5,
    }

    flat = flatten_nested_dict(test_case)
    result = {_decode_from_core(key): flat[key] for key in flat}
    assert test_case == result


def test_merge_nested_dictionaries():
    val4 = {"x": {"xx1": 7.5}}
    val5 = {"x": {"xx2": 6.0}}
    val1 = {"y": {"yy1": {"yyy1": {"yyyy1": 2}}}}
    val2 = {"y": {"yy3": 3, "yy4": {"yyy4": 5}}}
    val3 = {"y": {"yy2": 2}}
    val6 = {"z": 10}

    test_case = [val1, val2, val3, val4, val5, val6]

    result = {}
    for val in test_case:
        result = _merge_nested_dictionaries(result, val)

    assert result == {
        "y": {"yy1": {"yyy1": {"yyyy1": 2}}, "yy2": 2, "yy3": 3, "yy4": {"yyy4": 5}},
        "x": {"xx1": 7.5, "xx2": 6.0},
        "z": 10,
    }


def test_restore_nested_search_space(nested_params):
    test_case = flatten_nested_dict(nested_params)

    restored = restore_nested_search_space(test_case)

    assert restored == nested_params

    # Test with a simple case
    simple_case = {"x": "val", "z": 5}
    flat = flatten_nested_dict(simple_case)
    restored_simple = restore_nested_search_space(flat)

    assert restored_simple == simple_case


# ---------------------------------------------------------------------------
# Edge cases — these catch subtle bugs in the flatten/restore pipeline
# ---------------------------------------------------------------------------


class TestFlattenRestoreEdgeCases:
    """These edge cases are not exotic — they represent real user scenarios
    that would silently produce wrong results if the pipeline has off-by-one
    or boundary bugs."""

    def test_empty_dict(self):
        """Empty dict is the baseline measurement (BASELINE_CONFIG = {}).
        It flows through the same pipeline, so it must survive."""
        flat = flatten_nested_dict({})
        assert flat == {}
        restored = restore_nested_search_space(flat)
        assert restored == {}

    def test_single_level_dict(self):
        """Flat search spaces (no nesting) are the most common case.
        The flatten/restore should be a no-op on the structure."""
        original = {"x": "val_x", "y": "val_y", "z": "val_z"}
        flat = flatten_nested_dict(original)
        restored = restore_nested_search_space(flat)
        assert restored == original

    def test_deep_nesting_five_levels(self):
        """Some users have deeply nested configs (e.g. model.encoder.layer.attention.heads).
        Verify the pipeline handles 5+ levels without losing structure."""
        original = {"a": {"b": {"c": {"d": {"e": 42}}}}}
        flat = flatten_nested_dict(original)
        # Should produce exactly one leaf
        assert len(flat) == 1
        assert all(not isinstance(v, dict) for v in flat.values())
        restored = restore_nested_search_space(flat)
        assert restored == original

    def test_keys_with_underscores(self):
        """Underscores are the separator in the flattened key format.
        Base64 encoding prevents collisions, but this is worth verifying
        since 'learning_rate' is an extremely common parameter name."""
        original = {"learning_rate": 0.01, "batch_size": 32}
        flat = flatten_nested_dict(original)
        restored = restore_nested_search_space(flat)
        assert restored == original

    def test_nested_keys_with_underscores(self):
        """Nested keys with underscores in both parent and child — the worst
        case for separator collision."""
        original = {"opt_config": {"learning_rate": 0.01, "weight_decay": 0.001}}
        flat = flatten_nested_dict(original)
        restored = restore_nested_search_space(flat)
        assert restored == original

    def test_mixed_leaf_types(self):
        """Values can be strings, ints, floats — all must survive the round-trip."""
        original = {
            "str_val": "hello",
            "int_val": 42,
            "float_val": 3.14,
            "nested": {"inner": "world"},
        }
        flat = flatten_nested_dict(original)
        restored = restore_nested_search_space(flat)
        assert restored == original


class TestMergeEdgeCases:
    def test_merge_empty_into_populated(self):
        result = _merge_nested_dictionaries({"a": 1}, {})
        assert result == {"a": 1}

    def test_merge_populated_into_empty(self):
        result = _merge_nested_dictionaries({}, {"a": 1})
        assert result == {"a": 1}

    def test_merge_both_empty(self):
        result = _merge_nested_dictionaries({}, {})
        assert result == {}

    def test_merge_overwrites_leaf_with_leaf(self):
        """When both dicts have the same key with non-dict values, dict2 wins."""
        result = _merge_nested_dictionaries({"a": 1}, {"a": 2})
        assert result == {"a": 2}
