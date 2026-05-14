"""End-to-end search-space resolution against a stubbed GitHub release.

Exercises the full path through PtxasSearchSpace/NvccSearchSpace ->
resolver.resolve -> manifest fetch -> binary download -> size/sha256 verify -> cache
-> second-call cache hit. Uses pytest-mock to stub requests.get so no network
or filesystem outside tmp_path is touched.
"""

from __future__ import annotations

import hashlib
import json
from unittest.mock import MagicMock

import pytest
import requests

from compileiq.search_spaces import resolver
from compileiq.search_spaces.compilers import NvccSearchSpace, PtxasSearchSpace


PTXAS_BIN = b"PTXAS-13.3-DEFAULT" * 8000  # ~144 KB, similar to real assets
PTXAS_ATT_BIN = b"PTXAS-13.3-ATT" * 8000
NVCC_BIN = b"NVCC-13.3" * 16000

PTXAS_SHA = hashlib.sha256(PTXAS_BIN).hexdigest()
PTXAS_ATT_SHA = hashlib.sha256(PTXAS_ATT_BIN).hexdigest()
NVCC_SHA = hashlib.sha256(NVCC_BIN).hexdigest()

TAG = "search-spaces-2026.04.27"


MANIFEST = {
    "manifest_format": "1.0.0",
    "tag": TAG,
    "generated_at": "2026-04-27T00:00:00Z",
    "entries": [
        {
            "compiler": "ptxas",
            "compiler_version": "13.3",
            "variant": "default",
            "filename": "ptxas13.3_search_space.bin",
            "sha256": PTXAS_SHA,
            "size_bytes": len(PTXAS_BIN),
            "search_space_format": "1.0.0",
            "description": "PTXAS 13.3 default controls",
        },
        {
            "compiler": "ptxas",
            "compiler_version": "13.3",
            "variant": "att",
            "filename": "ptxas13.3_att_search_space.bin",
            "sha256": PTXAS_ATT_SHA,
            "size_bytes": len(PTXAS_ATT_BIN),
            "search_space_format": "1.0.0",
            "description": "PTXAS 13.3 ATT controls",
        },
        {
            "compiler": "nvcc",
            "compiler_version": "13.3",
            "variant": "default",
            "filename": "nvcc13.3_search_space.bin",
            "sha256": NVCC_SHA,
            "size_bytes": len(NVCC_BIN),
            "search_space_format": "1.0.0",
            "description": "NVCC 13.3 default controls",
        },
    ],
}


@pytest.fixture
def gh_release(monkeypatch):
    """Stub GitHub: /releases/latest -> TAG; /releases/download/<TAG>/<asset>."""
    repo = resolver._release_repo()
    call_log: list[str] = []
    routes = {
        f"{resolver.GH_API}/repos/{repo}/releases/latest": ("json", {"tag_name": TAG}),
        f"{resolver.GH_DL}/{repo}/releases/download/{TAG}/manifest.json": (
            "stream",
            json.dumps(MANIFEST).encode(),
        ),
        f"{resolver.GH_DL}/{repo}/releases/download/{TAG}/ptxas13.3_search_space.bin": (
            "stream",
            PTXAS_BIN,
        ),
        f"{resolver.GH_DL}/{repo}/releases/download/{TAG}/ptxas13.3_att_search_space.bin": (
            "stream",
            PTXAS_ATT_BIN,
        ),
        f"{resolver.GH_DL}/{repo}/releases/download/{TAG}/nvcc13.3_search_space.bin": (
            "stream",
            NVCC_BIN,
        ),
    }

    def fake_get(url, *args, **kwargs):
        call_log.append(url)
        kind, payload = routes[url]
        r = MagicMock(spec=requests.Response)
        r.status_code = 200
        r.raise_for_status.return_value = None
        if kind == "json":
            r.json = lambda: payload
        else:
            r.iter_content = lambda chunk_size: [
                payload[i : i + chunk_size] for i in range(0, len(payload), chunk_size)
            ]
            r.__enter__ = lambda self: self
            r.__exit__ = lambda self, *a: None
        return r

    monkeypatch.setattr(resolver.requests, "get", fake_get)
    resolver._resolve_latest_tag.cache_clear()
    monkeypatch.delenv(resolver.LOCAL_DIR_ENV_VAR, raising=False)
    # Use the no-prefix path so we mock /releases/latest directly.
    monkeypatch.setenv(resolver.SEARCH_SPACE_TAG_PREFIX_ENV_VAR, "")
    yield call_log
    resolver._resolve_latest_tag.cache_clear()


def test_end_to_end_default_variant(gh_release):
    path = PtxasSearchSpace(version="13.3").retrieve()
    assert path.read_bytes() == PTXAS_BIN
    assert path.name.endswith("ptxas13.3_search_space.bin")


def test_end_to_end_att_variant(gh_release):
    path = PtxasSearchSpace(version="13.3", variant="att").retrieve()
    assert path.read_bytes() == PTXAS_ATT_BIN
    assert "att" in path.name


def test_end_to_end_nvcc(gh_release):
    path = NvccSearchSpace(version="13.3").retrieve()
    assert path.read_bytes() == NVCC_BIN


def test_cache_hit_avoids_redownload(gh_release):
    call_log = gh_release
    PtxasSearchSpace(version="13.3").retrieve()
    download_calls = [u for u in call_log if u.endswith("ptxas13.3_search_space.bin")]
    assert len(download_calls) == 1
    PtxasSearchSpace(version="13.3").retrieve()
    download_calls = [u for u in call_log if u.endswith("ptxas13.3_search_space.bin")]
    assert len(download_calls) == 1, "second resolve re-downloaded the binary"


def test_separate_variants_cached_separately(gh_release):
    p1 = PtxasSearchSpace(version="13.3").retrieve()
    p2 = PtxasSearchSpace(version="13.3", variant="att").retrieve()
    assert p1 != p2
    assert p1.read_bytes() == PTXAS_BIN
    assert p2.read_bytes() == PTXAS_ATT_BIN


def test_air_gap_directory_skips_network(tmp_path, monkeypatch):
    air_gap = tmp_path / "ag"
    air_gap.mkdir()
    (air_gap / "ptxas13.3_search_space.bin").write_bytes(PTXAS_BIN)
    (air_gap / "manifest.json").write_text(json.dumps(MANIFEST))
    monkeypatch.setenv(resolver.LOCAL_DIR_ENV_VAR, str(air_gap))
    monkeypatch.setattr(
        resolver.requests,
        "get",
        lambda *a, **kw: pytest.fail("network used despite CIQ_SEARCH_SPACES_DIR"),
    )
    path = PtxasSearchSpace(version="13.3").retrieve()
    assert path == air_gap / "ptxas13.3_search_space.bin"
