"""
Fast unit tests for tracker classes in compileiq/tracker.py.
"""

import warnings
from compileiq.types import TrackerTypes, LoguruTrackerConfig
from compileiq.tracker import (
    DisabledTracker,
    LoguruTracker,
    logging_exception,
    _TRACKER_TYPES_TO_CLASSES,
    _TRACKER_TYPES_TO_CONFIG,
)


# ---------------------------------------------------------------------------
# DisabledTracker — must be a safe no-op
# ---------------------------------------------------------------------------


class TestDisabledTracker:
    """DisabledTracker is the default.  Every lifecycle method must silently
    succeed so that users who don't configure tracking never see errors."""

    def test_all_lifecycle_methods_succeed(self):
        tracker = DisabledTracker()
        # None of these should raise
        tracker.search_starts()
        tracker.generation_starts(generation_number=0)
        tracker.pre_objective(config={"x": 1})
        tracker.post_objective(scores=42.0)
        tracker.generation_ends(generation_number=0)
        tracker.search_ends()
        tracker.cleanup()


# ---------------------------------------------------------------------------
# LoguruTracker — must write lifecycle messages
# ---------------------------------------------------------------------------


class TestLoguruTracker:
    """LoguruTracker writes structured log messages at each lifecycle event.
    If search_starts/search_ends messages are missing, users have no way
    to tell when their search began or ended in the logs."""

    def test_search_lifecycle_messages(self, tmp_path):
        log_file = str(tmp_path / "test.log")
        # enqueue=False ensures synchronous writes so we can read immediately
        config = LoguruTrackerConfig(sink=log_file, level="DEBUG", enqueue=False)
        tracker = LoguruTracker(config)

        tracker.search_starts()
        tracker.generation_starts(generation_number=0)
        tracker.generation_ends(generation_number=0)
        tracker.search_ends()

        with open(log_file) as f:
            content = f.read()

        assert "Search started" in content
        assert "Search ended" in content
        assert "Generation 0 started" in content
        assert "Generation 0 ended" in content


# ---------------------------------------------------------------------------
# Tracker registry — must be complete
# ---------------------------------------------------------------------------


class TestTrackerRegistry:
    """The _TRACKER_TYPES_TO_CLASSES and _TRACKER_TYPES_TO_CONFIG maps are
    used to look up tracker implementations by enum value.  If a TrackerType
    is missing from either map, users get a KeyError when they try to use it."""

    def test_all_types_in_classes_map(self):
        for tt in TrackerTypes:
            if tt == TrackerTypes.DEFAULT:
                # DEFAULT aliases to DISABLED, not its own entry
                continue
            assert tt in _TRACKER_TYPES_TO_CLASSES, f"{tt} missing from _TRACKER_TYPES_TO_CLASSES"

    def test_all_types_in_config_map(self):
        for tt in TrackerTypes:
            if tt == TrackerTypes.DEFAULT:
                continue
            assert tt in _TRACKER_TYPES_TO_CONFIG, f"{tt} missing from _TRACKER_TYPES_TO_CONFIG"


# ---------------------------------------------------------------------------
# logging_exception decorator — must swallow, not re-raise
# ---------------------------------------------------------------------------


class TestLoggingException:
    """The logging_exception decorator wraps tracker methods so that a
    logging failure (e.g. disk full, MLflow connection lost) never kills
    the user's search.  It should swallow the exception and emit a warning."""

    def test_swallows_exception_and_warns(self):
        @logging_exception
        def boom():
            raise RuntimeError("simulated logging failure")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = boom()  # Should NOT raise
            assert result is None
            assert any("Exception was caught" in str(warning.message) for warning in w)

    def test_passes_through_on_success(self):
        @logging_exception
        def ok():
            return 42

        assert ok() == 42
