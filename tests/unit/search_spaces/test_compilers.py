"""Tests for compileiq.search_spaces.compilers provider classes."""

from __future__ import annotations

import pathlib
import sys

import pytest

from compileiq.ciq import Search
from compileiq.search_spaces import compilers
from compileiq.search_spaces.compilers import (
    LocalSearchSpaceBin,
    NvccSearchSpace,
    PtxasSearchSpace,
    SearchSpaceProvider,
)
from compileiq.search_spaces.resolver import ResolvedSearchSpace, SearchSpaceResolutionMetadata
from compileiq.types import SearchConfiguration, Worker


class DummyWorker(Worker):
    @classmethod
    def create(cls, cache_folder, normalize, tracker):
        return cls(cache_folder=cache_folder, normalize=normalize, tracker=tracker)

    def run(self, **kwargs):
        return []


@pytest.fixture
def captured_resolve(monkeypatch):
    """Replace resolver.resolve_with_metadata with a recorder."""
    calls: list[dict] = []
    sentinel = pathlib.Path("/tmp/sentinel.bin")

    def fake_resolve_with_metadata(**kwargs):
        calls.append(kwargs)
        metadata = SearchSpaceResolutionMetadata(
            compiler=kwargs["compiler"],
            compiler_version=kwargs["compiler_version"],
            variant=kwargs["variant"],
            description=f"{kwargs['compiler']} {kwargs['compiler_version']} controls",
            requested_tag=kwargs["tag"],
            resolved_tag="search-spaces-test",
            filename=f"{kwargs['compiler']}.bin",
            sha256="a" * 64,
            size_bytes=1,
            source="cache",
            path=str(sentinel),
        )
        return ResolvedSearchSpace(path=sentinel, metadata=metadata)

    monkeypatch.setattr(compilers, "resolve_with_metadata", fake_resolve_with_metadata)
    return calls, sentinel


def test_ptxas_passes_compiler_through(captured_resolve):
    calls, sentinel = captured_resolve
    provider = PtxasSearchSpace(version="13.3")
    assert provider.retrieve() == sentinel
    assert calls[0]["compiler"] == "ptxas"
    assert calls[0]["compiler_version"] == "13.3"
    assert calls[0]["variant"] == "default"
    assert calls[0]["tag"] == "latest"
    assert provider.resolution_metadata is not None
    assert provider.resolution_metadata.filename == "ptxas.bin"


def test_ptxas_default_constructor_uses_launch_compiler_version(captured_resolve):
    calls, _ = captured_resolve
    PtxasSearchSpace().retrieve()
    assert calls[0]["compiler_version"] == "13.3"
    assert calls[0]["tag"] == "latest"


def test_nvcc_passes_compiler_through(captured_resolve):
    calls, _ = captured_resolve
    NvccSearchSpace(version="13.3").retrieve()
    assert calls[0]["compiler"] == "nvcc"


def test_variant_and_tag_pass_through(captured_resolve):
    calls, _ = captured_resolve
    PtxasSearchSpace(version="13.3", variant="att", tag="search-spaces-2026.04.27").retrieve()
    assert calls[0]["variant"] == "att"
    assert calls[0]["tag"] == "search-spaces-2026.04.27"


def test_providers_satisfy_protocol():
    assert isinstance(PtxasSearchSpace("13.3"), SearchSpaceProvider)
    assert isinstance(NvccSearchSpace("13.3"), SearchSpaceProvider)


def test_local_search_space_returns_path(tmp_path):
    target = tmp_path / "x.bin"
    target.write_bytes(b"data")
    provider = LocalSearchSpaceBin(target)
    assert provider.retrieve() == target.resolve()
    assert isinstance(provider, SearchSpaceProvider)


def test_local_search_space_validates_path():
    with pytest.raises(FileNotFoundError):
        LocalSearchSpaceBin("/does/not/exist/please.bin")


def test_local_search_space_expands_user(tmp_path, monkeypatch):
    if sys.platform == "win32":
        monkeypatch.setenv("USERPROFILE", str(tmp_path))
    else:
        monkeypatch.setenv("HOME", str(tmp_path))
    target = tmp_path / "y.bin"
    target.write_bytes(b"data")
    provider = LocalSearchSpaceBin("~/y.bin")
    assert provider.retrieve() == target.resolve()


def test_search_exposes_provider_resolution_metadata(
    captured_resolve, mock_socket_listen, tmp_path
):
    tuner = Search(
        objective_function=lambda _: 1.0,
        search_space=PtxasSearchSpace(version="13.3"),
        search_config=SearchConfiguration(generations=1, pool_size=6),
        cache_folder=tmp_path,
        worker_type=DummyWorker,
    )

    metadata = tuner.search_space_resolution_metadata
    assert isinstance(metadata, list)
    assert len(metadata) == 1
    assert metadata[0]["filename"] == "ptxas.bin"
    assert metadata[0]["resolved_tag"] == "search-spaces-test"
    assert metadata[0]["description"] == "ptxas 13.3 controls"


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="Search validation intentionally rejects multiple config search spaces on Windows",
)
def test_search_resolves_provider_inside_mixed_search_space(
    captured_resolve, mock_socket_listen, tmp_path
):
    user_space = {"block_size": {"type": "range", "min": 1, "max": 4}}
    tuner = Search(
        objective_function=lambda _: 1.0,
        search_space=[user_space, PtxasSearchSpace(version="13.3", variant="att")],
        search_config=SearchConfiguration(generations=1, pool_size=6),
        cache_folder=tmp_path,
        worker_type=DummyWorker,
    )

    assert tuner.search_space == [user_space, pathlib.Path("/tmp/sentinel.bin")]
    assert tuner.search_space_resolution_metadata == [
        {
            "compiler": "ptxas",
            "compiler_version": "13.3",
            "variant": "att",
            "description": "ptxas 13.3 controls",
            "requested_tag": "latest",
            "resolved_tag": "search-spaces-test",
            "filename": "ptxas.bin",
            "sha256": "a" * 64,
            "size_bytes": 1,
            "source": "cache",
            "path": "/tmp/sentinel.bin",
        }
    ]


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="Search validation intentionally rejects multiple config search spaces on Windows",
)
def test_search_exposes_metadata_list_for_multiple_providers(
    captured_resolve, mock_socket_listen, tmp_path
):
    tuner = Search(
        objective_function=lambda _: 1.0,
        search_space=[PtxasSearchSpace(version="13.3"), NvccSearchSpace(version="13.3")],
        search_config=SearchConfiguration(generations=1, pool_size=6),
        cache_folder=tmp_path,
        worker_type=DummyWorker,
    )

    metadata = tuner.search_space_resolution_metadata
    assert isinstance(metadata, list)
    assert [item["compiler"] for item in metadata] == ["ptxas", "nvcc"]
