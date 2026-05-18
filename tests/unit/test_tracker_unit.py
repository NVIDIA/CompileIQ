"""
Fast unit tests for tracker classes in compileiq/tracker.py.
"""

import pickle
import warnings
from unittest.mock import MagicMock

import pytest

from compileiq.types import TrackerTypes, LoguruTrackerConfig, MLflowTrackerConfig
from compileiq.tracker import (
    DisabledTracker,
    LoguruTracker,
    logging_exception,
    _resolution_metadata_to_mlflow_tags,
    _TRACKER_TYPES_TO_CLASSES,
    _TRACKER_TYPES_TO_CONFIG,
)


@pytest.fixture
def mock_mlflow(monkeypatch):
    mock = MagicMock()
    mock.get_experiment_by_name.return_value.experiment_id = "test-experiment-id"
    mock.active_run.return_value.info.run_id = "test-child-run-id"
    mock.active_run.return_value.info.run_name = "test-child-run-name"
    monkeypatch.setattr("compileiq.tracker._mlflow", mock)
    return mock


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

    def test_search_space_resolution_metadata_logged(self, tmp_path):
        log_file = str(tmp_path / "test.log")
        config = LoguruTrackerConfig(sink=log_file, level="DEBUG", enqueue=False)
        tracker = LoguruTracker(config)

        tracker.search_starts(
            search_space_resolution_metadata=[
                {
                    "compiler": "ptxas",
                    "resolved_tag": "search-spaces-2026.05.05",
                    "filename": "ptxas13.3_search_space.bin",
                    "sha256": "a" * 64,
                }
            ]
        )

        with open(log_file) as f:
            content = f.read()

        assert "Resolved search space" in content
        assert "ptxas13.3_search_space.bin" in content

    def test_search_space_resolution_metadata_list_logged(self, tmp_path):
        log_file = str(tmp_path / "test.log")
        config = LoguruTrackerConfig(sink=log_file, level="DEBUG", enqueue=False)
        tracker = LoguruTracker(config)

        tracker.search_starts(
            search_space_resolution_metadata=[
                {"compiler": "ptxas", "filename": "ptxas13.3_search_space.bin"},
                {"compiler": "nvcc", "filename": "nvcc13.3_search_space.bin"},
            ]
        )

        with open(log_file) as f:
            content = f.read()

        assert "ptxas13.3_search_space.bin" in content
        assert "nvcc13.3_search_space.bin" in content


def test_resolution_metadata_to_mlflow_tags_flattens_records():
    assert _resolution_metadata_to_mlflow_tags([{"compiler": "ptxas"}]) == {
        "search_space.0.compiler": "ptxas",
    }

    assert _resolution_metadata_to_mlflow_tags(
        [
            {"compiler": "ptxas", "filename": "ptxas.bin"},
            {"compiler": "nvcc", "filename": "nvcc.bin"},
        ]
    ) == {
        "search_space.0.compiler": "ptxas",
        "search_space.0.filename": "ptxas.bin",
        "search_space.1.compiler": "nvcc",
        "search_space.1.filename": "nvcc.bin",
    }

    assert _resolution_metadata_to_mlflow_tags(None) == {}
    assert _resolution_metadata_to_mlflow_tags([]) == {}


# ---------------------------------------------------------------------------
# MLflowTracker — must stay picklable and handle every ParamArg shape
# ---------------------------------------------------------------------------


class TestMLflowTracker:
    def test_pickle_roundtrip(self, mock_mlflow):
        from compileiq.tracker import MLflowTracker

        tracker = MLflowTracker(MLflowTrackerConfig())
        restored = pickle.loads(pickle.dumps(tracker))

        assert type(restored) is MLflowTracker
        assert restored.tracker_config.model_dump() == tracker.tracker_config.model_dump()

    def test_dict_config_logs_verbatim(self, mock_mlflow):
        from compileiq.tracker import MLflowTracker

        tracker = MLflowTracker(MLflowTrackerConfig(log_config=False))
        tracker.pre_objective(config={"lr": 0.1, "batch": 32})
        mock_mlflow.log_params.assert_called_with({"lr": 0.1, "batch": 32})

    def test_str_config_logs_path(self, mock_mlflow):
        from compileiq.tracker import MLflowTracker

        tracker = MLflowTracker(MLflowTrackerConfig(log_config=False))
        tracker.pre_objective(config="/some/search_space.json")
        mock_mlflow.log_params.assert_called_with({"config_path": "/some/search_space.json"})

    def test_list_config_flattens_with_index_prefix(self, mock_mlflow):
        from compileiq.tracker import MLflowTracker

        tracker = MLflowTracker(MLflowTrackerConfig(log_config=False))
        tracker.pre_objective(config=[{"a": 1, "b": 2}, {"c": 3}])
        mock_mlflow.log_params.assert_called_with(
            {"config_0.a": 1, "config_0.b": 2, "config_1.c": 3}
        )

    def test_list_config_with_str_entry_flattens_as_config_path(self, mock_mlflow):
        from compileiq.tracker import MLflowTracker

        tracker = MLflowTracker(MLflowTrackerConfig(log_config=False))
        tracker.pre_objective(config=[{"a": 1}, "/legacy/search_space.config"])
        mock_mlflow.log_params.assert_called_with(
            {"config_0.a": 1, "config_1.config_path": "/legacy/search_space.config"}
        )


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
