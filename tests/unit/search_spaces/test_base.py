"""
Tests for search space builders in compileiq/search_spaces/base.py.
"""

import json
import pytest
from compileiq.search_spaces import base as ss
from compileiq.search_spaces.models import (
    RangeParamConfig,
    ChoiceParamConfig,
    LiteralParamConfig,
)
from compileiq.utils.helpers import _literal_dive
from compileiq.utils._setup_files import _setup_dna_with_dict


# ---------------------------------------------------------------------------
# Knockout probability
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fn,kwargs",
    [
        (ss.range, {"start": 0, "end": 10}),
        (ss.choice, {"choice_list": [1, 2, 3]}),
        (ss.literal, {"const_value": 5}),
        (ss.log_sampling, {"start": 0.001, "end": 1.0}),
    ],
)
def test_knockout_prob_absent_by_default(fn, kwargs):
    result = fn(**kwargs)
    assert result.knockout_threshold is None


@pytest.mark.parametrize(
    "fn,kwargs",
    [
        (ss.range, {"start": 0, "end": 10}),
        (ss.choice, {"choice_list": [1, 2, 3]}),
        (ss.literal, {"const_value": 5}),
        (ss.log_sampling, {"start": 0.001, "end": 1.0}),
    ],
)
def test_knockout_prob_present_when_set(fn, kwargs):
    result = fn(**kwargs, knockout_prob=0.7)
    assert result.knockout_threshold is not None


def test_knockout_threshold_is_inverted():
    # threshold = 1.0 - knockout_prob
    result = ss.literal(const_value=1, knockout_prob=0.3)
    assert result.knockout_threshold == pytest.approx(0.7)


def test_literal_dive_threads_knockout():
    param_set = {"lr": 0.01, "batch_size": 32}
    result = _literal_dive(param_set, knockout=0.5)
    for v in result.values():
        assert isinstance(v, LiteralParamConfig)
        assert v.knockout_threshold == pytest.approx(0.5)


def test_literal_dive_no_knockout_by_default():
    param_set = {"lr": 0.01}
    result = _literal_dive(param_set)
    for v in result.values():
        assert isinstance(v, LiteralParamConfig)
        assert v.knockout_threshold is None


# ---------------------------------------------------------------------------
# range() — model output and input validation
# ---------------------------------------------------------------------------


class TestRange:
    """range() generates a RangeParamConfig model that serializes to
    JSON for the core binary."""

    def test_basic_output_structure(self):
        result = ss.range(start=0, end=10, step=2)
        assert isinstance(result, RangeParamConfig)
        assert result.type == "range"
        assert result.low == 0
        assert result.high == 10
        assert result.step == 2

    def test_end_less_than_start_raises(self):
        """start=10, end=5 makes no sense — catch it before the core sees it."""
        with pytest.raises(ValueError, match="proper range"):
            ss.range(start=10, end=5)

    def test_end_equal_to_start_raises(self):
        """A zero-width range would produce no candidates."""
        with pytest.raises(ValueError, match="proper range"):
            ss.range(start=5, end=5)

    def test_negative_step_raises(self):
        """The core only accepts positive steps."""
        with pytest.raises(ValueError, match="positive step"):
            ss.range(start=0, end=10, step=-1)

    def test_float_step_precision(self):
        """Small float steps (e.g. 0.01) are common for learning rate tuning."""
        result = ss.range(start=0.0, end=1.0, step=0.01)
        assert result.step == 0.01

    def test_seed_range_valid(self):
        """seed_low/seed_high constrain initialization to a subset of the range."""
        result = ss.range(start=0, end=100, step=1, seed_low=20, seed_high=80)
        assert result.seed_low == 20
        assert result.seed_high == 80

    def test_seed_low_below_start_raises(self):
        with pytest.raises(ValueError, match="outside the correct range"):
            ss.range(start=10, end=100, seed_low=5, seed_high=50)

    def test_seed_high_above_end_raises(self):
        with pytest.raises(ValueError, match="outside the correct range"):
            ss.range(start=0, end=100, seed_low=10, seed_high=150)

    def test_seed_high_less_than_seed_low_raises(self):
        with pytest.raises(ValueError, match="outside the correct range"):
            ss.range(start=0, end=100, seed_low=60, seed_high=40)


# ---------------------------------------------------------------------------
# choice() — value handling for JSON serialization
# ---------------------------------------------------------------------------


class TestChoice:
    """choice() generates a ChoiceParamConfig model. Values must be properly
    typed for JSON serialization to the core."""

    def test_basic_int_list(self):
        result = ss.choice([1, 2, 3])
        assert isinstance(result, ChoiceParamConfig)
        assert result.type == "enum"
        assert result.vals == [1, 2, 3]

    def test_string_values(self):
        result = ss.choice(["adam", "sgd"])
        assert result.vals == ["adam", "sgd"]

    def test_bool_converted_to_int(self):
        """Bools must become 0/1 for the core's parser."""
        result = ss.choice([True, False, 1])
        assert result.vals == [1, 0, 1]

    def test_single_element_list(self):
        """A choice with one element is valid — it's a constant that participates
        in knockout."""
        result = ss.choice([42])
        assert result.type == "enum"
        assert result.vals == [42]


# ---------------------------------------------------------------------------
# literal() — constant values with optional knockout
# ---------------------------------------------------------------------------


class TestLiteral:
    """literal() represents a fixed value. Its main use case is with
    knockout_prob — parameters that can be toggled on/off."""

    def test_invalid_const_value_raises(self):
        with pytest.raises(ValueError, match="`const_value` must be a str, int, or float"):
            ss.literal(const_value=lambda x: print("Oops, I am not a valid literal"))

    def test_int_value(self):
        result = ss.literal(const_value=5)
        assert isinstance(result, LiteralParamConfig)
        assert result.type == "literal"
        assert result.value == 5

    def test_string_value(self):
        result = ss.literal(const_value="hello")
        assert result.value == "hello"

    def test_bool_converted_to_int(self):
        """Bools must become 0/1."""
        result = ss.literal(const_value=True)
        assert isinstance(result, LiteralParamConfig)
        assert result.type == "literal"
        assert result.value == 1

        result = ss.literal(const_value=False)
        assert isinstance(result, LiteralParamConfig)
        assert result.type == "literal"
        assert result.value == 0


# ---------------------------------------------------------------------------
# log_sampling() — logarithmic distribution emulated via choice
# ---------------------------------------------------------------------------


class TestLogSampling:
    """log_sampling() creates a logarithmic distribution by generating samples
    via numpy.geomspace and passing them to choice()."""

    def test_output_is_choice_model(self):
        result = ss.log_sampling(start=0.001, end=1.0, total=5)
        assert isinstance(result, ChoiceParamConfig)
        assert len(result.vals) == 5

    def test_total_controls_sample_count(self):
        result_5 = ss.log_sampling(start=0.1, end=10.0, total=5)
        result_20 = ss.log_sampling(start=0.1, end=10.0, total=20)
        assert len(result_20.vals) > len(result_5.vals)

    def test_start_and_end_in_output(self):
        result = ss.log_sampling(start=1.0, end=100.0, total=10)
        assert isinstance(result, ChoiceParamConfig)
        assert len(result.vals) == 10


# ---------------------------------------------------------------------------
# JSON DNA serialization — _setup_dna_with_dict
# ---------------------------------------------------------------------------


def test_setup_dna_with_dict_produces_valid_json():
    search_space = {
        "x": ss.range(start=1.0, end=20.0, step=0.5),
        "y": ss.choice([1, 2, 3]),
        "z": ss.literal(const_value=42),
        "flag": ss.choice(["O0", "O1", "O2"]),
    }
    result = _setup_dna_with_dict(search_space)

    # Must be valid JSON
    parsed = json.loads(result)

    assert parsed["format"] == "compileiq-search-space-v1"
    assert "classes" in parsed
    assert "dna" in parsed

    # DNA section has braces and all keys
    assert parsed["dna"][0] == "{"
    assert parsed["dna"][-1] == "}"
    assert "x" in parsed["dna"]
    assert "y" in parsed["dna"]

    # String vals are plain strings (no embedded quotes in JSON)
    flag_vals = parsed["classes"]["flag"]["vals"]
    assert flag_vals == ["O0", "O1", "O2"]

    # Numbers are numbers, not strings
    x_class = parsed["classes"]["x"]
    assert isinstance(x_class["low"], (int, float))
    assert isinstance(x_class["step"], (int, float))


def test_setup_dna_with_dict_seed_uses_hyphens():
    search_space = {
        "x": ss.range(start=0, end=100, step=1, seed_low=10, seed_high=50),
    }
    result = _setup_dna_with_dict(search_space)
    parsed = json.loads(result)

    x_class = parsed["classes"]["x"]
    assert "seed-low" in x_class
    assert "seed-high" in x_class
    assert "seed_low" not in x_class
    assert "seed_high" not in x_class
