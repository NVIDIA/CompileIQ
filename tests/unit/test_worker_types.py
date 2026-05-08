from unittest.mock import MagicMock
import pytest
from pydantic import ValidationError
from compileiq.types import (
    Worker,
    WorkerTypes,
    INVALID_SCORE,
)
from compileiq.worker import MultiProcessWorker, RayWorker, AsyncWorker, IsoMultiProcessWorker
from compileiq.ciq import Search
from compileiq.search_spaces import base as ss
from compileiq.types import SearchConfiguration


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

WORKER_TYPE_STRINGS = [pytest.param(m.value, m, m.worker_type(), id=m.value) for m in WorkerTypes]


def _make_search(mocker, tmp_path, **overrides):
    """Create a Search instance with mocked internals (same pattern as test_ciq.py)."""
    mocker.patch("compileiq.core.core_comms.socket.socket", return_value=MagicMock())
    mocker.patch("compileiq.worker.multiprocessing.Manager", return_value=MagicMock())
    defaults = dict(
        objective_function=lambda p: 1.0,
        search_space={"lr": ss.range(start=0, end=10)},
        search_config=SearchConfiguration(generations=2),
        cache_folder=tmp_path,
    )
    defaults.update(overrides)
    return Search(**defaults)  # pyright: ignore[reportArgumentType]


# ---------------------------------------------------------------------------
# 1. WorkerTypes enum resolves to correct worker classes
# ---------------------------------------------------------------------------


class TestWorkerTypesEnum:
    @pytest.mark.parametrize("string,expected_enum,expected_worker", WORKER_TYPE_STRINGS)
    def test_from_string(self, string, expected_enum, expected_worker):
        assert WorkerTypes(string) is expected_enum
        assert expected_enum.worker_type() is expected_worker

    def test_default_is_native(self):
        assert WorkerTypes.DEFAULT is WorkerTypes.NATIVE

    def test_default_from_string_is_native(self):
        assert WorkerTypes("native") is WorkerTypes.DEFAULT

    def test_default_string_is_not_valid(self):
        with pytest.raises(ValueError):
            WorkerTypes("default")


# ---------------------------------------------------------------------------
# 2. Search accepts WorkerType and type[Worker]
# ---------------------------------------------------------------------------


class TestSearchWorkerTypeDispatch:
    def test_accepts_worker_type_enum(self, mocker, tmp_path):
        search = _make_search(mocker, tmp_path, worker_type=WorkerTypes.NATIVE)
        assert isinstance(search._worker, MultiProcessWorker)

    def test_accepts_worker_class_directly(self, mocker, tmp_path):
        search = _make_search(mocker, tmp_path, worker_type=MultiProcessWorker)
        assert isinstance(search._worker, MultiProcessWorker)

    @pytest.mark.parametrize("string,expected_enum,expected_worker", WORKER_TYPE_STRINGS)
    def test_accepts_string(self, mocker, tmp_path, string, expected_enum, expected_worker):
        search = _make_search(mocker, tmp_path, worker_type=string)
        assert isinstance(search._worker, expected_worker)

    def test_rejects_invalid_worker_type(self, mocker, tmp_path):
        with pytest.raises((ValidationError, RuntimeError)):
            _make_search(mocker, tmp_path, worker_type="not_valid")


# ---------------------------------------------------------------------------
# 3. Worker.create() contract
# ---------------------------------------------------------------------------


class TestWorkerCreate:
    def test_multiprocess_create(self, mocker):
        mocker.patch("compileiq.worker.multiprocessing.Manager", return_value=MagicMock())
        w = MultiProcessWorker.create(cache_folder="/tmp", normalize=False, tracker=None)
        assert isinstance(w, MultiProcessWorker)

    def test_isolated_create(self, mocker):
        mocker.patch("compileiq.worker.multiprocessing.Manager", return_value=MagicMock())
        w = IsoMultiProcessWorker.create(cache_folder="/tmp", normalize=False, tracker=None)
        assert isinstance(w, IsoMultiProcessWorker)

    def test_ray_create(self):
        w = RayWorker.create(cache_folder="/tmp", normalize=True, tracker=None)
        assert isinstance(w, RayWorker)
        assert w.normalize is True

    def test_async_create(self):
        w = AsyncWorker.create(cache_folder="/tmp", normalize=False, tracker=None)
        assert isinstance(w, AsyncWorker)


# ---------------------------------------------------------------------------
# 4. respects_num_workers is set correctly
# ---------------------------------------------------------------------------


class TestRespectsNumWorkers:
    def test_multiprocess_respects(self, mocker):
        mocker.patch("compileiq.worker.multiprocessing.Manager", return_value=MagicMock())
        w = MultiProcessWorker(cache_folder="/tmp")
        assert w.respects_num_workers is True

    def test_ray_does_not_respect(self):
        w = RayWorker(cache_folder="/tmp")
        assert w.respects_num_workers is False

    def test_async_does_not_respect(self):
        w = AsyncWorker(cache_folder="/tmp")
        assert w.respects_num_workers is False

    def test_isolated_respects(self, mocker):
        mocker.patch("compileiq.worker.multiprocessing.Manager", return_value=MagicMock())
        w = IsoMultiProcessWorker(cache_folder="/tmp")
        assert w.respects_num_workers is True


# ---------------------------------------------------------------------------
# 5. Worker.normalize_scores() edge cases
# ---------------------------------------------------------------------------


class TestNormalizeScores:
    def test_raises_when_baseline_is_none(self):
        with pytest.raises(RuntimeError, match="baseline"):
            Worker.normalize_scores(1.0, None)  # type: ignore[arg-type]

    def test_single_score(self):
        assert Worker.normalize_scores(10.0, 5.0) == 2.0

    def test_multi_score(self):
        result = Worker.normalize_scores([10.0, 20.0], [5.0, 4.0])
        assert result == [2.0, 5.0]

    def test_division_by_zero_returns_invalid(self):
        result = Worker.normalize_scores(10.0, 0)
        assert result == INVALID_SCORE

    def test_division_by_zero_in_multi_returns_invalid(self):
        result = Worker.normalize_scores([10.0, 5.0], [0, 5.0])
        assert result[0] == INVALID_SCORE
        assert result[1] == 1.0
