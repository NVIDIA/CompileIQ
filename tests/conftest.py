import os

import pytest
from tests.utils import SCRIPTED_ENGINE, ScriptedBackend
import compileiq.search_spaces.base as ss
from compileiq.backends import (
    BackendArtifactInfo,
    BackendCapabilities,
    BackendError,
    get_backend_registration,
    register_backend,
    unregister_backend,
)


def _amber_installed() -> bool:
    """Return whether the native amber runtime can register in this environment."""

    try:
        get_backend_registration("amber")
    except BackendError:
        return False
    return True


def pytest_collection_modifyitems(config, items):
    """Skip real-engine tests when amber is absent, unless CI requires it.

    CI jobs that install the native runtime set ``CIQ_REQUIRE_AMBER=1`` so a
    missing or broken payload fails these tests instead of silently skipping them.
    """

    del config
    marked = [item for item in items if item.get_closest_marker("requires_amber") is not None]
    if not marked or os.environ.get("CIQ_REQUIRE_AMBER") == "1" or _amber_installed():
        return
    skip = pytest.mark.skip(
        reason=(
            "the native amber runtime is not installed (run `make install`); "
            "set CIQ_REQUIRE_AMBER=1 to fail instead of skipping"
        )
    )
    for item in marked:
        item.add_marker(skip)


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
def scripted_engine():
    """Register ``ScriptedBackend`` and yield the engine name to pass as ``engine=``."""

    register_backend(
        SCRIPTED_ENGINE,
        ScriptedBackend,
        capabilities=BackendCapabilities(arbitrary_batch_size=True),
        artifact=BackendArtifactInfo(
            backend_version="test",
            abi_version=0,
            artifact_sha256="0" * 64,
        ),
        replace=True,
    )
    try:
        yield SCRIPTED_ENGINE
    finally:
        unregister_backend(SCRIPTED_ENGINE)


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
