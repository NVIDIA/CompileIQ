import socket
import subprocess
from unittest.mock import MagicMock
import pytest
import random
import json
from uuid import uuid4
from tests.utils import generate_params
from compileiq.core.core_types import ParameterSet, CompletionMessage, SingleCandidate
from compileiq.utils.helpers import _encode_for_core
import compileiq.search_spaces.base as ss


def pytest_configure():
    pytest.current_gen = 0
    pytest.max_gen = 0
    pytest.pool_size = 0
    pytest.nested_test = False
    # this is for base64 encoding to handle serialization issues (not binary blobs)
    pytest.encoded_knobs = True


@pytest.fixture
def mock_search_space():
    # TODO: Generate with hypothesis ?
    return {
        "x": ss.range(start=1.0, end=1000, step=0.5),
        "y": ss.log_sampling(start=1.0, end=20.0),
        "z": ss.range(start=-1000, end=0, step=1),
        "duration": ss.literal(2),
    }


@pytest.fixture
def mock_nested_search_space():
    return {
        "x": {"xx1": ss.range(start=1.0, end=1000, step=0.5)},
        "y": {"yy1": ss.log_sampling(start=1.0, end=20.0), "yy2": ss.choice([1, 2, 3])},
        "z": {"zz1": {"zzz1": ss.range(start=-1000, end=0, step=1)}},
        "duration": ss.literal(2),
    }


@pytest.fixture(autouse=True)
def sandbox_cache_dir(tmp_path, monkeypatch):
    cache_dir = str(tmp_path / "compileiq_cache")
    monkeypatch.setattr("compileiq.ciq._CACHE_DIR", cache_dir)
    monkeypatch.setattr("compileiq.config.const._CACHE_DIR", cache_dir)


@pytest.fixture
def mock_core_start(monkeypatch):
    """Replace process launch at the bundled-backend ownership boundary."""

    def mock_start(*args, **kwargs):
        """Return a completed-process-shaped object without starting the bundled core."""

        mock_proc = MagicMock(spec=subprocess.Popen)
        mock_proc.poll.return_value = 0
        return mock_proc

    monkeypatch.setattr("compileiq.bundled_backend.CoreIPC.start", mock_start)


@pytest.fixture
def mock_socket_listen(monkeypatch):
    """Give the bundled adapter a connected socket pair without opening a real port."""

    mock_sock = MagicMock(spec=socket.socket)
    mock_sock.accept.return_value = (MagicMock(spec=socket.socket), ("localhost", 0))
    mock_sock.getsockname.return_value = ("localhost", 0)
    monkeypatch.setattr("compileiq.bundled_backend.initialize_socket", lambda: mock_sock)


@pytest.fixture
def mock_send_to_core(monkeypatch):
    """Suppress bundled-core writes while tests exercise the public Search lifecycle."""

    def mock_send(*args, **kwargs):
        """Stand in for the response write after accepting a candidate batch."""

        pass

    monkeypatch.setattr("compileiq.bundled_backend.CoreIPC.send_to_core", mock_send)


@pytest.fixture
def mock_receive_from_core(monkeypatch):
    """Drive deterministic bundled generations through the adapter's IPC receive seam."""

    def mock_receive(*args, **kwargs):
        """Return configured candidate batches until the synthetic search completes."""

        # `pool_size`, `max_gen` and `current_gen` come from `pytest_namespace()`
        # and are updated at the test function
        if pytest.current_gen < pytest.max_gen:
            candidate_list = []
            params = generate_params(pytest.pool_size, nested=pytest.nested_test)
            for knob in params:
                encoded_knob = (
                    {
                        "_".join(map(_encode_for_core, key.split("_"))): val
                        for key, val in knob.items()
                    }
                    if isinstance(knob, dict) and pytest.encoded_knobs
                    else knob
                )
                candidate_list.append(
                    SingleCandidate(
                        id=random.randint(1, 1000000),
                        knobs=json.dumps(encoded_knob),
                    )
                )
            param_set = ParameterSet(
                params=candidate_list,
                generation_num=pytest.current_gen,
                invocation_id=uuid4().int,
            )
            pytest.current_gen += 1

            return param_set
        else:
            return CompletionMessage(complete=True)

    monkeypatch.setattr("compileiq.bundled_backend.CoreIPC.receive_from_core", mock_receive)


def _collected_tests_require_ray(items):
    return any(item.get_closest_marker("requires_ray") is not None for item in items)


@pytest.fixture(autouse=True, scope="session")
def manage_ray(request):
    # Do not start Ray for unrelated tests. A failed Ray startup can leave child
    # processes behind after pytest reports success, causing the job to exit 143.
    if not _collected_tests_require_ray(request.session.items):
        yield
        return

    import os
    import ray

    started_ray = not ray.is_initialized()
    if started_ray:
        ray.init(_temp_dir=os.environ.get("RAY_TMPDIR"), ignore_reinit_error=True)
    try:
        yield
    finally:
        if started_ray and ray.is_initialized():
            ray.shutdown()
