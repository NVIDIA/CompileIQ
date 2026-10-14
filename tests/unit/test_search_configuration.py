"""
Tests for SearchConfiguration validators in compileiq/types.py.
"""

import pytest
from compileiq.types import (
    ProblemType,
    SearchConfiguration,
)


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

    def test_derived_sizes_are_not_marked_as_caller_supplied(self):
        """Preserve the signal used by provider-specific configuration validation.

        The native provider inspects ``model_fields_set`` so it can reject an
        explicitly supplied retired setting, such as ``cull_size``, while
        accepting the same value when CompileIQ derives it as a default.
        """

        derived = SearchConfiguration(generations=1, pool_size=40)
        explicit = SearchConfiguration(generations=1, pool_size=40, cull_size=10)

        assert "cull_size" not in derived.model_fields_set
        assert "cull_size" in explicit.model_fields_set

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
# to_legacy / from_legacy: round-trip through the legacy .config format
# ---------------------------------------------------------------------------


@pytest.mark.filterwarnings("ignore:SearchConfiguration.*is deprecated:FutureWarning")
class TestLegacyRoundTrip:
    """Search still accepts a path to a lisp-like legacy .config file as
    ``search_config``.  to_legacy() writes that format and from_legacy() reads
    it back.  If either direction has a bug, a reloaded configuration silently
    differs from the one that was saved."""

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

    def test_from_legacy_rejects_malformed_supported_field_value(self, tmp_path):
        """Malformed supported-field values should fail at the parser boundary."""
        config_path = tmp_path / "test.config"
        config_path.write_text("(generations . 1+)\n")

        with pytest.raises(ValueError, match="Invalid value.*generations") as excinfo:
            SearchConfiguration.from_legacy(str(config_path))

        assert isinstance(excinfo.value.__cause__, SyntaxError)

    def test_to_legacy_contains_seek_minimum(self):
        """The legacy format names the field 'seek_minimum', not 'problem_type'.
        Verify the key name translation happens."""
        cfg = SearchConfiguration(generations=1, problem_type=ProblemType.MIN)
        legacy = cfg.to_legacy()
        assert "seek_minimum" in legacy

    def test_to_legacy_warns_that_it_is_deprecated(self):
        cfg = SearchConfiguration(generations=1)
        with pytest.warns(FutureWarning, match="to_legacy is deprecated.*future release"):
            cfg.to_legacy()

    def test_from_legacy_warns_that_it_is_deprecated(self, tmp_path):
        config_path = tmp_path / "test.config"
        config_path.write_text("(generations . 3)\n")
        with pytest.warns(FutureWarning, match="from_legacy is deprecated.*future release"):
            restored = SearchConfiguration.from_legacy(str(config_path))
        assert restored.generations == 3


class TestDeprecatedObjectiveWeights:
    """objective_weights has no effect since 1.1, so it is no longer validated."""

    def test_weights_are_not_validated(self):
        cfg = SearchConfiguration(
            generations=1, num_objectives=2, objective_weights=[0.5, 0.3, 0.1]
        )
        assert cfg.objective_weights == [0.5, 0.3, 0.1]


# ---------------------------------------------------------------------------
# Basic construction constraints
# ---------------------------------------------------------------------------


class TestSearchConfigConstraints:
    """Catch common user mistakes with clear errors rather than cryptic
    failures deep in the search loop."""

    def test_pool_size_must_be_gt_5(self):
        """The optimizer needs a minimum viable candidate pool."""
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
