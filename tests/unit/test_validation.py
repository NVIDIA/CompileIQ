"""
Tests for compileiq/utils/validation.py — the score validation pipeline.
"""

import pytest
from compileiq.utils.validation import (
    validate_inf_nan,
    validate_scores,
    Score,
    INVALID_SCORE,
)


# ---------------------------------------------------------------------------
# validate_inf_nan — the JSON safety net
# ---------------------------------------------------------------------------


class TestValidateInfNan:
    """Core communicates over JSON, which has no inf/nan.  This function
    converts them to INVALID_SCORE ("*") before they reach the wire."""

    def test_positive_inf_becomes_invalid(self):
        assert validate_inf_nan(float("inf")) == INVALID_SCORE

    def test_negative_inf_becomes_invalid(self):
        assert validate_inf_nan(float("-inf")) == INVALID_SCORE

    def test_nan_becomes_invalid(self):
        assert validate_inf_nan(float("nan")) == INVALID_SCORE

    def test_normal_float_passes_through(self):
        assert validate_inf_nan(42.5) == 42.5

    def test_zero_passes_through(self):
        """0.0 is a perfectly valid score (e.g. perfect minimization)."""
        assert validate_inf_nan(0.0) == 0.0

    def test_negative_float_passes_through(self):
        assert validate_inf_nan(-3.14) == -3.14


# ---------------------------------------------------------------------------
# validate_scores — type-checking objective function returns
# ---------------------------------------------------------------------------


class TestValidateScoresSingleObjective:
    """Single-objective searches expect a scalar (int or float), not a list."""

    def test_valid_int(self):
        result = validate_scores(42, num_objectives=1)
        assert result == 42

    def test_valid_float(self):
        result = validate_scores(3.14, num_objectives=1)
        assert result == 3.14

    def test_invalid_score_star_passes(self):
        """Users can return INVALID_SCORE ("*") to signal a failed evaluation."""
        result = validate_scores(INVALID_SCORE, num_objectives=1)
        assert result == INVALID_SCORE

    def test_wrong_type_raises(self):
        """A string that isn't "*" should be rejected — catches typos like
        returning "done" instead of a number."""
        with pytest.raises(ValueError, match="does not match expected types"):
            validate_scores("not_a_score", num_objectives=1)

    def test_inf_becomes_invalid_via_pydantic(self):
        """Float inf should be caught by the AfterValidator on SingleScore
        and converted to INVALID_SCORE before it reaches JSON serialization."""
        result = validate_scores(float("inf"), num_objectives=1)
        assert result == INVALID_SCORE


class TestValidateScoresMultiObjective:
    """Multi-objective searches expect a tuple/list of exactly num_objectives scalars."""

    def test_correct_length_tuple(self):
        result = validate_scores((1.0, 2.0), num_objectives=2)
        assert len(result) == 2

    def test_wrong_length_raises(self):
        """Returning 3 scores when num_objectives=2 is a user bug that must
        be caught immediately, not silently truncated."""
        with pytest.raises((ValueError, Exception)):
            validate_scores((1.0, 2.0, 3.0), num_objectives=2)

    def test_mixed_int_float(self):
        result = validate_scores((1, 2.5), num_objectives=2)
        assert len(result) == 2

    def test_single_star_expands_to_all_invalid(self):
        """When a multi-objective function returns just "*", we expand it to
        ["*", "*", ...] so downstream code can treat each objective uniformly."""
        result = validate_scores(INVALID_SCORE, num_objectives=3)
        assert list(result) == [INVALID_SCORE, INVALID_SCORE, INVALID_SCORE]

    def test_partial_invalid_in_tuple(self):
        """One objective fails, the others succeed — this is valid."""
        result = validate_scores((1.0, INVALID_SCORE), num_objectives=2)
        assert result[0] == 1.0
        assert result[1] == INVALID_SCORE


# ---------------------------------------------------------------------------
# Score model — the container for all evaluation results
# ---------------------------------------------------------------------------


class TestScoreFailed:
    """Score.failed is checked after every evaluation to decide whether a
    DNA was successful.  If this is wrong, the optimizer either keeps garbage
    solutions or discards valid ones."""

    def test_single_objective_invalid_score(self):
        s = Score(score=INVALID_SCORE, params="{}", param_id=1, num_objectives=1)
        assert s.failed is True

    def test_single_objective_valid_score(self):
        s = Score(score=42.0, params="{}", param_id=1, num_objectives=1)
        assert s.failed is False

    def test_single_objective_invalid_norm_score(self):
        """Even if the raw score is fine, a failed normalization means the
        evaluation is unusable."""
        s = Score(
            score=42.0,
            norm_score=INVALID_SCORE,
            params="{}",
            param_id=1,
            num_objectives=1,
        )
        assert s.failed is True

    def test_multi_objective_one_invalid(self):
        """If any single objective fails, the whole evaluation is failed —
        you can't have a partial Pareto point."""
        s = Score(
            score=[1.0, INVALID_SCORE],
            params="{}",
            param_id=1,
            num_objectives=2,
        )
        assert s.failed is True

    def test_multi_objective_all_valid(self):
        s = Score(score=[1.0, 2.0], params="{}", param_id=1, num_objectives=2)
        assert s.failed is False

    def test_multi_objective_invalid_norm_score(self):
        s = Score(
            score=[1.0, 2.0],
            norm_score=[1.0, INVALID_SCORE],
            params="{}",
            param_id=1,
            num_objectives=2,
        )
        assert s.failed is True


class TestScoreModelValidation:
    """The Score model validator auto-normalizes the score field to match
    num_objectives.  This catches user mistakes at construction time rather
    than later when the data is already mixed into results."""

    def test_single_score_wrapped_correctly(self):
        """A single int score should be stored as-is (not wrapped in a list)
        after validation."""
        s = Score(score=5, params="{}", param_id=1, num_objectives=1)
        assert s.score == 5

    def test_multi_score_length_mismatch_raises(self):
        """Declaring num_objectives=2 but providing 3 scores is a bug."""
        with pytest.raises(Exception):
            Score(score=[1.0, 2.0, 3.0], params="{}", param_id=1, num_objectives=2)
