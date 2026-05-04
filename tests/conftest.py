import socket
import subprocess
from unittest.mock import MagicMock
import pytest
import random
import json
from uuid import uuid4
from tests.utils import generate_params
from compileiq.ciq import Search
from compileiq.core.core_types import ParameterSet, CompletionMessage, SingleDNA
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
    def mock_start(*args, **kwargs):
        mock_proc = MagicMock(spec=subprocess.Popen)
        mock_proc.poll.return_value = 0
        return mock_proc

    monkeypatch.setattr("compileiq.ciq.CoreIPC.start", mock_start)


@pytest.fixture
def mock_socket_listen(monkeypatch):
    mock_sock = MagicMock(spec=socket.socket)
    mock_sock.accept.return_value = (MagicMock(spec=socket.socket), ("localhost", 0))
    mock_sock.getsockname.return_value = ("localhost", 0)
    monkeypatch.setattr(
        Search.__private_attributes__["_listen_socket"],
        "default_factory",
        lambda: mock_sock,
    )


@pytest.fixture
def mock_send_to_core(monkeypatch):
    def mock_send(*args, **kwargs):
        pass

    monkeypatch.setattr("compileiq.ciq.CoreIPC.send_to_core", mock_send)


@pytest.fixture
def mock_receive_from_core(monkeypatch):
    def mock_receive(*args, **kwargs):
        # `pool_size`, `max_gen` and `current_gen` come from `pytest_namespace()`
        # and are updated at the test function
        if pytest.current_gen < pytest.max_gen:
            dna_list = []
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
                dna_list.append(
                    SingleDNA(
                        id=random.randint(1, 1000000),
                        knobs=json.dumps(encoded_knob),
                    )
                )
            param_set = ParameterSet(
                params=dna_list,
                generation_num=pytest.current_gen,
                invocation_id=uuid4().int,
            )
            pytest.current_gen += 1

            return param_set
        else:
            return CompletionMessage(complete=1)

    monkeypatch.setattr("compileiq.ciq.CoreIPC.receive_from_core", mock_receive)


@pytest.fixture(autouse=True, scope="session")
def manage_ray():
    try:
        import os
        import ray

        if not ray.is_initialized():
            ray.init(_temp_dir=os.environ.get("RAY_TMPDIR"), ignore_reinit_error=True)
    except ImportError:
        pass
    except Exception:
        pass
    yield
    try:
        import ray

        if ray.is_initialized():
            ray.shutdown()
    except ImportError:
        pass
