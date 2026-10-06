"""
Tests for search space builders in compileiq/search_spaces/base.py.
"""

import pytest
from compileiq.search_spaces import base as ss
from compileiq.search_spaces.models import (
    RangeParamConfig,
    ChoiceParamConfig,
    LiteralParamConfig,
)
from compileiq.backends import search_space_identity
from compileiq.utils.helpers import _literal_dive


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
    assert result.knockout_prob is None
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
    assert result.knockout_prob == pytest.approx(0.7)


def test_knockout_threshold_compatibility_read_is_inverted():
    """Keep the legacy complementary threshold readable without storing it as shared state."""

    result = ss.literal(const_value=1, knockout_prob=0.3)

    assert result.knockout_prob == pytest.approx(0.3)
    assert result.knockout_threshold == pytest.approx(0.7)


@pytest.mark.parametrize(
    "fn,kwargs",
    [
        (ss.range, {"start": 0, "end": 10}),
        (ss.choice, {"choice_list": [1, 2, 3]}),
        (ss.literal, {"const_value": 5}),
        (ss.log_sampling, {"start": 0.001, "end": 1.0}),
    ],
)
def test_knockout_prob_point_seven_has_exact_quantized_contract(fn, kwargs):
    """Read the legacy complementary threshold exactly: 0.7 gives 0.3, not 0.30000000000000004."""

    result = fn(**kwargs, knockout_prob=0.7)

    assert result.knockout_prob == 0.7
    assert result.knockout_threshold == 0.3


@pytest.mark.parametrize(
    "fn,kwargs",
    [
        (ss.range, {"start": 0, "end": 10}),
        (ss.choice, {"choice_list": [1, 2, 3]}),
        (ss.literal, {"const_value": 5}),
        (ss.log_sampling, {"start": 0.001, "end": 1.0}),
    ],
)
@pytest.mark.parametrize("knockout_prob", [-0.01, 1.01])
def test_knockout_probability_rejects_out_of_range_values(fn, kwargs, knockout_prob):
    """Apply the same public probability domain to every search-space builder."""

    with pytest.raises(ValueError, match="between 0.0 and 1.0"):
        fn(**kwargs, knockout_prob=knockout_prob)


@pytest.mark.parametrize("knockout_prob,threshold", [(0.0, 1.0), (1.0, 0.0)])
def test_knockout_probability_accepts_closed_interval_boundaries(
    knockout_prob,
    threshold,
):
    """Treat always-present and always-knocked-out as valid endpoint policies."""

    result = ss.literal(const_value=1, knockout_prob=knockout_prob)

    assert result.knockout_prob == knockout_prob
    assert result.knockout_threshold == threshold


@pytest.mark.parametrize("knockout_prob", [True, False, "0.3", object()])
def test_knockout_probability_rejects_non_numeric_values(knockout_prob):
    """Reject bool coercion and string parsing at the provider-neutral model boundary."""

    with pytest.raises(TypeError, match="knockout_prob must be a number"):
        ss.literal(const_value=1, knockout_prob=knockout_prob)


def test_legacy_knockout_threshold_input_canonicalizes_to_probability():
    """Accept existing direct model construction while emitting only semantic state."""

    result = LiteralParamConfig.model_validate(
        {"value": 1, "knockout_threshold": 0.7}
    )

    assert result.knockout_prob == pytest.approx(0.3)
    assert result.knockout_threshold == pytest.approx(0.7)
    assert result.model_dump() == {
        "knockout_prob": pytest.approx(0.3),
        "type": "literal",
        "value": 1,
    }


def test_consistent_probability_and_legacy_threshold_are_accepted():
    """Permit compatibility callers that redundantly provide the same policy."""

    result = LiteralParamConfig.model_validate(
        {
            "value": 1,
            "knockout_prob": 0.3,
            "knockout_threshold": 0.7,
        }
    )

    assert result.knockout_prob == pytest.approx(0.3)


def test_inconsistent_probability_and_legacy_threshold_are_rejected():
    """Fail instead of silently choosing one of two conflicting knockout policies."""

    with pytest.raises(ValueError, match="must be complementary"):
        LiteralParamConfig.model_validate(
            {
                "value": 1,
                "knockout_prob": 0.3,
                "knockout_threshold": 0.6,
            }
        )


def test_legacy_threshold_and_probability_have_the_same_search_space_identity():
    """Canonicalize compatibility input before checkpoint identity is computed."""

    semantic = {"flag": ss.literal(1, knockout_prob=0.3)}
    legacy = {
        "flag": LiteralParamConfig.model_validate(
            {"value": 1, "knockout_threshold": 0.7}
        )
    }

    assert search_space_identity(semantic) == search_space_identity(legacy)


def test_literal_dive_threads_knockout():
    param_set = {"lr": 0.01, "batch_size": 32}
    result = _literal_dive(param_set, knockout=0.5)
    for v in result.values():
        assert isinstance(v, LiteralParamConfig)
        assert v.knockout_prob == pytest.approx(0.5)


def test_literal_dive_no_knockout_by_default():
    param_set = {"lr": 0.01}
    result = _literal_dive(param_set)
    for v in result.values():
        assert isinstance(v, LiteralParamConfig)
        assert v.knockout_prob is None


# ---------------------------------------------------------------------------
# range() — model output and input validation
# ---------------------------------------------------------------------------


class TestRange:
    """range() generates a RangeParamConfig model that engines consume."""

    def test_basic_output_structure(self):
        result = ss.range(start=0, end=10, step=2)
        assert isinstance(result, RangeParamConfig)
        assert result.type == "range"
        assert result.low == 0
        assert result.high == 10
        assert result.step == 2

    def test_end_less_than_start_raises(self):
        """start=10, end=5 makes no sense; catch it before an engine sees it."""
        with pytest.raises(ValueError, match="proper range"):
            ss.range(start=10, end=5)

    def test_end_equal_to_start_raises(self):
        """A zero-width range would produce no candidates."""
        with pytest.raises(ValueError, match="proper range"):
            ss.range(start=5, end=5)

    def test_negative_step_raises(self):
        """Range parameters require positive steps."""
        with pytest.raises(ValueError, match="positive step"):
            ss.range(start=0, end=10, step=-1)

    def test_zero_step_raises(self):
        """A zero step cannot define a reachable discrete range."""
        with pytest.raises(ValueError, match="positive step"):
            ss.range(start=0, end=10, step=0)

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
        """Reject seed initialization values that the declared range cannot produce."""

        with pytest.raises(ValueError, match="seed_low.*outside"):
            ss.range(start=10, end=100, seed_low=5, seed_high=50)

    def test_seed_high_above_end_raises(self):
        """Keep the seed subset inside the public range's upper boundary."""

        with pytest.raises(ValueError, match="seed_high.*outside"):
            ss.range(start=0, end=100, seed_low=10, seed_high=150)

    def test_seed_high_less_than_seed_low_raises(self):
        """Reject an inverted seed subset before it reaches either backend."""

        with pytest.raises(ValueError, match="greater than or equal"):
            ss.range(start=0, end=100, seed_low=60, seed_high=40)

    def test_range_end_can_be_non_aligned_upper_bound(self):
        """Treat end as an inclusive bound even when the stepped grid stops below it."""

        result = ss.range(start=0, end=9, step=2)

        assert result.high == 9

    def test_seed_bounds_can_fall_between_stepped_values(self):
        """Treat seed bounds as initialization limits rather than candidate values."""

        result = ss.range(start=0, end=10, step=2, seed_low=3, seed_high=9)

        assert result.seed_low == 3
        assert result.seed_high == 9

    def test_one_sided_seed_bounds_are_supported(self):
        """Preserve one-sided initialization bounds supported by both current engines."""

        low_only = ss.range(start=0, end=10, step=2, seed_low=4)
        high_only = ss.range(start=0, end=10, step=2, seed_high=6)

        assert low_only.seed_low == 4
        assert low_only.seed_high is None
        assert high_only.seed_low is None
        assert high_only.seed_high == 6


# ---------------------------------------------------------------------------
# choice() — value handling for JSON serialization
# ---------------------------------------------------------------------------


class TestChoice:
    """choice() generates a ChoiceParamConfig model. Values must be properly
    typed for JSON serialization in results and checkpoints."""

    def test_basic_int_list(self):
        result = ss.choice([1, 2, 3])
        assert isinstance(result, ChoiceParamConfig)
        assert result.type == "enum"
        assert result.vals == [1, 2, 3]

    def test_string_values(self):
        result = ss.choice(["adam", "sgd"])
        assert result.vals == ["adam", "sgd"]

    def test_bool_converted_to_int(self):
        """Bools are normalized to 0/1."""
        result = ss.choice([True, False, 1])
        assert result.vals == [1, 0, 1]

    def test_single_element_list(self):
        """A choice with one element is valid — it's a constant that participates
        in knockout."""
        result = ss.choice([42])
        assert result.type == "enum"
        assert result.vals == [42]

    def test_empty_choice_raises(self):
        """Reject a search-space dimension that cannot produce any candidate value."""

        with pytest.raises(ValueError, match="at least one value"):
            ss.choice([])


# ---------------------------------------------------------------------------
# literal() — constant values with optional knockout
# ---------------------------------------------------------------------------


class TestLiteral:
    """literal() represents a fixed value. Its main use case is with
    knockout_prob — parameters that can be toggled on/off."""

    def test_invalid_const_value_raises(self):
        with pytest.raises(ValueError, match="`const_value` must be a str, int, or float"):
            ss.literal(const_value=lambda x: print("Oops, I am not a valid literal"))  # type: ignore[arg-type]

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
