"""
Tests for SearchConfiguration validators in compileiq/types.py.
"""

import json
import pytest
from compileiq.types import (
    ProblemType,
    SearchConfiguration,
)


# ---------------------------------------------------------------------------
# validate_weights — objective weight sanity checks
# ---------------------------------------------------------------------------


class TestValidateWeights:
    """Objective weights let users bias multi-objective searches.  They must
    match num_objectives in length and sum to exactly 1.0."""

    def test_valid_weights_pass(self):
        cfg = SearchConfiguration(
            generations=1,
            num_objectives=2,
            objective_weights=[0.6, 0.4],
        )
        assert cfg.objective_weights == [0.6, 0.4]

    def test_length_mismatch_raises(self):
        """3 weights for 2 objectives is a user config error."""
        with pytest.raises(ValueError, match="must be equal to num_objectives"):
            SearchConfiguration(
                generations=1,
                num_objectives=2,
                objective_weights=[0.3, 0.3, 0.4],
            )

    def test_weights_not_summing_to_one_raises(self):
        with pytest.raises(ValueError, match="must add up to 1.0"):
            SearchConfiguration(
                generations=1,
                num_objectives=2,
                objective_weights=[0.5, 0.3],
            )

    def test_none_weights_is_valid(self):
        """Omitting weights entirely is fine — the optimizer treats all
        objectives equally."""
        cfg = SearchConfiguration(generations=1, num_objectives=2)
        assert cfg.objective_weights is None


# ---------------------------------------------------------------------------
# set_pool_and_cull_sizes — auto-calculation from num_objectives
# ---------------------------------------------------------------------------


class TestSetPoolAndCullSizes:
    """Auto-calculation of pool/cull sizes is tested at the helper level in
    test_helpers_encoding.py. These tests cover the SearchConfiguration
    integration — that explicit user values aren't overridden."""

    def test_explicit_values_respected(self):
        cfg = SearchConfiguration(generations=1, pool_size=50, cull_size=20)
        assert cfg.pool_size == 50
        assert cfg.cull_size == 20

    def test_explicit_pool_auto_cull(self):
        cfg = SearchConfiguration(generations=1, pool_size=40)
        assert cfg.pool_size == 40
        assert cfg.cull_size is not None
        assert cfg.cull_size % 2 == 0
        assert cfg.cull_size < cfg.pool_size
        assert (cfg.pool_size - cfg.cull_size) >= 1 + 2 * cfg.num_objectives

    def test_explicit_pool_auto_cull_large_num_objectives(self):
        """When pool_size is user-provided but small relative to num_objectives,
        auto-cull must still leave enough survivors rather than blindly using 50%."""
        num_obj = 10
        min_survivors = 1 + 2 * num_obj  # 21
        cfg = SearchConfiguration(generations=1, pool_size=32, num_objectives=num_obj)
        assert cfg.pool_size is not None and cfg.cull_size is not None
        assert cfg.cull_size % 2 == 0
        assert cfg.pool_size - cfg.cull_size >= min_survivors

    def test_cull_equal_to_pool_raises(self):
        """cull_size == pool_size leaves zero survivors; must be rejected."""
        with pytest.raises(ValueError, match="must be less than pool_size"):
            SearchConfiguration(generations=1, pool_size=20, cull_size=20)

    def test_cull_greater_than_pool_raises(self):
        """cull_size > pool_size is nonsensical and must be rejected."""
        with pytest.raises(ValueError, match="must be less than pool_size"):
            SearchConfiguration(generations=1, pool_size=20, cull_size=22)

    def test_too_few_survivors_for_objectives_raises(self):
        """pool - cull must leave at least 1 + 2*num_objectives survivors.
        With num_objectives=3, that threshold is 7; 6 survivors must raise."""
        with pytest.raises(ValueError, match="too small"):
            SearchConfiguration(generations=1, pool_size=20, cull_size=14, num_objectives=3)

    def test_pool_too_small_for_reference_directions(self):
        with pytest.raises(ValueError, match="too small"):
            SearchConfiguration(
                generations=1,
                pool_size=6,
                cull_size=2,
                num_objectives=2,
            )

    def test_valid_multi_objective_config_passes(self):
        """A well-formed multi-objective config clears every pool/cull check."""
        cfg = SearchConfiguration(generations=1, pool_size=32, cull_size=16, num_objectives=3)
        assert cfg.pool_size == 32
        assert cfg.cull_size == 16


# ---------------------------------------------------------------------------
# to_legacy / from_legacy — round-trip through the core's config format
# ---------------------------------------------------------------------------


class TestLegacyRoundTrip:
    """The core binary reads a lisp-like .config format.  to_legacy() converts
    our Python config into that format, and from_legacy() reads it back.  If
    either direction has a bug, the core binary silently gets wrong parameters
    or we lose user settings on reload."""

    def test_min_round_trip(self, tmp_path):
        original = SearchConfiguration(
            generations=5,
            pool_size=32,
            cull_size=16,
            mutate_rate=0.25,
            problem_type=ProblemType.MIN,
            num_objectives=1,
        )
        config_path = tmp_path / "test.config"
        config_path.write_text(original.to_legacy())

        restored = SearchConfiguration.from_legacy(str(config_path))

        assert restored.problem_type == ProblemType.MIN
        assert restored.generations == 5
        assert restored.pool_size == 32
        assert restored.cull_size == 16
        assert restored.mutate_rate == 0.25

    def test_max_round_trip(self, tmp_path):
        """MAX is the other half of the problem_type enum — if MIN works but
        MAX doesn't, half our users are broken."""
        original = SearchConfiguration(
            generations=3,
            pool_size=40,
            cull_size=20,
            problem_type=ProblemType.MAX,
        )
        config_path = tmp_path / "test.config"
        config_path.write_text(original.to_legacy())

        restored = SearchConfiguration.from_legacy(str(config_path))
        assert restored.problem_type == ProblemType.MAX

    def test_normalize_maps_to_qualitative(self, tmp_path):
        """Legacy format uses 'qualitative' (inverted sense) instead of
        'normalize'.  This mapping is subtle and easy to get backwards."""
        cfg = SearchConfiguration(generations=1, normalize=True)
        legacy_str = cfg.to_legacy()

        # normalize=True should produce qualitative . #f in the legacy format
        assert "(qualitative . #f)" in legacy_str

        cfg_false = SearchConfiguration(generations=1, normalize=False)
        legacy_false = cfg_false.to_legacy()
        assert "(qualitative . #t)" in legacy_false

    def test_from_legacy_requires_config_extension(self):
        """Passing a non-.config path should fail, not silently load garbage."""
        with pytest.raises(ValueError, match="extension .config"):
            SearchConfiguration.from_legacy("not_a_config.txt")

    def test_to_legacy_contains_seek_minimum(self):
        """The core binary reads 'seek_minimum', not 'problem_type'.
        Verify the key name translation happens."""
        cfg = SearchConfiguration(generations=1, problem_type=ProblemType.MIN)
        legacy = cfg.to_legacy()
        assert "seek_minimum" in legacy


# ---------------------------------------------------------------------------
# to_json_dict — JSON output for core consumption
# ---------------------------------------------------------------------------


class TestJsonDict:
    """to_json_dict() produces the dict that gets written as main_config.json.
    It uses the Python SDK field names — core handles the translation to its
    internal names (seek_minimum, qualitative) in load-json-main-config."""

    def test_problem_type_preserved(self):
        cfg = SearchConfiguration(generations=5, problem_type=ProblemType.MIN)
        d = cfg.to_json_dict()
        assert d["problem_type"] == "min"

        cfg_max = SearchConfiguration(generations=5, problem_type=ProblemType.MAX)
        d_max = cfg_max.to_json_dict()
        assert d_max["problem_type"] == "max"

    def test_normalize_preserved(self):
        cfg = SearchConfiguration(generations=1, normalize=True)
        d = cfg.to_json_dict()
        assert d["normalize"] is True

        cfg_false = SearchConfiguration(generations=1, normalize=False)
        d2 = cfg_false.to_json_dict()
        assert d2["normalize"] is False

    def test_none_values_excluded(self):
        cfg = SearchConfiguration(generations=1, objective_weights=None)
        d = cfg.to_json_dict()
        assert "objective_weights" not in d

    def test_output_is_json_serializable(self):
        cfg = SearchConfiguration(generations=5, pool_size=32, cull_size=16, mutate_rate=0.3)
        d = cfg.to_json_dict()
        roundtripped = json.loads(json.dumps(d))
        assert roundtripped == d

    def test_numeric_fields_preserved(self):
        cfg = SearchConfiguration(generations=10, pool_size=50, cull_size=20, mutate_rate=0.15)
        d = cfg.to_json_dict()
        assert d["generations"] == 10
        assert d["pool_size"] == 50
        assert d["cull_size"] == 20
        assert d["mutate_rate"] == 0.15


# ---------------------------------------------------------------------------
# Basic construction constraints
# ---------------------------------------------------------------------------


class TestSearchConfigConstraints:
    """Catch common user mistakes with clear errors rather than cryptic
    failures deep in the search loop."""

    def test_pool_size_must_be_gt_5(self):
        """The evolutionary algorithm needs a minimum viable population."""
        with pytest.raises(Exception):
            SearchConfiguration(generations=1, pool_size=4, cull_size=2)

    def test_cull_size_must_be_even(self):
        """Parent pairing requires an even number of survivors."""
        with pytest.raises(Exception):
            SearchConfiguration(generations=1, pool_size=10, cull_size=3)

    def test_generations_must_be_positive(self):
        with pytest.raises(Exception):
            SearchConfiguration(generations=0)

    def test_extra_fields_rejected(self):
        """Typos in config keys (e.g. 'generatons') should fail, not be
        silently ignored.  This is why we use extra='forbid'."""
        with pytest.raises(Exception):
            SearchConfiguration(generations=1, typo_field=True)  # type: ignore[call-arg]
