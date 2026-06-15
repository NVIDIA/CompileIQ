"""Tests for compileiq.search_spaces.resolver."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from unittest.mock import MagicMock

import pytest
import requests

from compileiq.search_spaces import resolver


BIN_BYTES = b"PTX-SS-13.3-DEFAULT" * 200
BIN_SHA = hashlib.sha256(BIN_BYTES).hexdigest()
BIN_FILENAME = "ptxas13.3_search_space.bin"
TAG = "search-spaces-2026.04.27"
OLD_TAG = "search-spaces-2026.01.01"
NOW = dt.datetime(2026, 6, 15, 12, 0, tzinfo=dt.timezone.utc)


def _manifest_payload(
    filename: str = BIN_FILENAME,
    sha: str = BIN_SHA,
    size: int = len(BIN_BYTES),
    tag: str = TAG,
) -> str:
    return json.dumps(
        {
            "manifest_format": "1.0.0",
            "tag": tag,
            "generated_at": "2026-04-27T00:00:00Z",
            "entries": [
                {
                    "compiler": "ptxas",
                    "compiler_version": "13.3",
                    "variant": "default",
                    "filename": filename,
                    "sha256": sha,
                    "size_bytes": size,
                    "search_space_format": "1.0.0",
                    "description": "PTXAS 13.3 default controls",
                }
            ],
        }
    )


def _set_now(monkeypatch, now: dt.datetime = NOW) -> None:
    monkeypatch.setattr(resolver, "_utc_now", lambda: now)


def _write_latest_cache(
    repo: str,
    tag_prefix: str,
    resolved_tag: str,
    fetched_at: dt.datetime = NOW,
    ttl_days: float = resolver.DEFAULT_LATEST_TAG_TTL_DAYS,
) -> None:
    cache = resolver._latest_tag_cache_path(repo, tag_prefix)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(
        json.dumps(
            {
                "repo": repo,
                "tag_prefix": tag_prefix,
                "resolved_tag": resolved_tag,
                "fetched_at": fetched_at.isoformat(),
                "ttl_days": ttl_days,
            }
        )
    )


@pytest.fixture(autouse=True)
def reset_lru(monkeypatch):
    resolver._resolve_latest_tag.cache_clear()
    monkeypatch.delenv(resolver.LOCAL_DIR_ENV_VAR, raising=False)
    monkeypatch.delenv(resolver.LATEST_TAG_TTL_DAYS_ENV_VAR, raising=False)
    # Force the no-prefix code path by default so tests can target
    # /releases/latest cleanly. test_default_tag_prefix_used_when_unset and
    # test_tag_prefix_filters_releases override to exercise the prefix path.
    monkeypatch.setenv(resolver.SEARCH_SPACE_TAG_PREFIX_ENV_VAR, "")
    monkeypatch.delenv(resolver.RELEASE_REPO_ENV_VAR, raising=False)
    yield
    resolver._resolve_latest_tag.cache_clear()


@pytest.fixture
def fake_http(monkeypatch):
    """Stub requests.get with a per-URL handler dict."""
    handlers: dict[str, list] = {}

    def _stream_response(content: bytes, status: int = 200):
        r = MagicMock(spec=requests.Response)
        r.status_code = status
        r.iter_content = lambda chunk_size: [
            content[i : i + chunk_size] for i in range(0, len(content), chunk_size)
        ]
        if status >= 400:
            err = requests.HTTPError(response=r)
            r.raise_for_status.side_effect = err
        else:
            r.raise_for_status.return_value = None
        r.__enter__ = lambda self: self
        r.__exit__ = lambda self, *a: None
        return r

    def _json_response(payload, status: int = 200):
        r = MagicMock(spec=requests.Response)
        r.status_code = status
        r.json = lambda: payload
        if status >= 400:
            err = requests.HTTPError(response=r)
            r.raise_for_status.side_effect = err
        else:
            r.raise_for_status.return_value = None
        return r

    def fake_get(url, *args, **kwargs):
        handler = handlers.get(url)
        if handler is None:
            raise AssertionError(f"unexpected URL: {url}")
        if isinstance(handler, list):
            return handler.pop(0)
        return handler

    monkeypatch.setattr(resolver.requests, "get", fake_get)
    return handlers, _stream_response, _json_response


def _add_asset_routes(handlers, stream, repo: str, tag: str = TAG) -> None:
    handlers[f"{resolver.GH_DL}/{repo}/releases/download/{tag}/manifest.json"] = stream(
        _manifest_payload(tag=tag).encode()
    )
    handlers[f"{resolver.GH_DL}/{repo}/releases/download/{tag}/{BIN_FILENAME}"] = stream(BIN_BYTES)


def test_air_gap_short_circuit(tmp_path, monkeypatch):
    air_gap = tmp_path / "ag"
    air_gap.mkdir()
    (air_gap / BIN_FILENAME).write_bytes(BIN_BYTES)
    (air_gap / "manifest.json").write_text(_manifest_payload())
    monkeypatch.setenv(resolver.LOCAL_DIR_ENV_VAR, str(air_gap))

    # If air-gap path is taken, requests.get must NOT be called.
    monkeypatch.setattr(
        resolver.requests,
        "get",
        lambda *a, **kw: pytest.fail(f"network used despite {resolver.LOCAL_DIR_ENV_VAR}"),
    )

    path = resolver.resolve("ptxas", "13.3")
    assert path == air_gap / BIN_FILENAME
    assert path.read_bytes() == BIN_BYTES


def test_air_gap_metadata(tmp_path, monkeypatch):
    air_gap = tmp_path / "ag"
    air_gap.mkdir()
    (air_gap / BIN_FILENAME).write_bytes(BIN_BYTES)
    (air_gap / "manifest.json").write_text(_manifest_payload())
    monkeypatch.setenv(resolver.LOCAL_DIR_ENV_VAR, str(air_gap))

    resolved = resolver.resolve_with_metadata("ptxas", "13.3")

    assert resolved.path == air_gap / BIN_FILENAME
    assert resolved.metadata.source == resolver.LOCAL_DIR_ENV_VAR
    assert resolved.metadata.filename == BIN_FILENAME
    assert resolved.metadata.sha256 == BIN_SHA
    assert resolved.metadata.description == "PTXAS 13.3 default controls"
    assert resolved.metadata.requested_tag == "latest"
    assert resolved.metadata.resolved_tag == TAG


def test_air_gap_pinned_tag_must_match_manifest(tmp_path, monkeypatch):
    air_gap = tmp_path / "ag"
    air_gap.mkdir()
    (air_gap / BIN_FILENAME).write_bytes(BIN_BYTES)
    (air_gap / "manifest.json").write_text(_manifest_payload(tag="search-spaces-other"))
    monkeypatch.setenv(resolver.LOCAL_DIR_ENV_VAR, str(air_gap))

    with pytest.raises(ValueError, match="Local manifest tag mismatch"):
        resolver.resolve_with_metadata("ptxas", "13.3", tag=TAG)


def test_air_gap_pinned_tag_metadata_preserves_requested_tag(tmp_path, monkeypatch):
    air_gap = tmp_path / "ag"
    air_gap.mkdir()
    (air_gap / BIN_FILENAME).write_bytes(BIN_BYTES)
    (air_gap / "manifest.json").write_text(_manifest_payload())
    monkeypatch.setenv(resolver.LOCAL_DIR_ENV_VAR, str(air_gap))

    resolved = resolver.resolve_with_metadata("ptxas", "13.3", tag=TAG)

    assert resolved.metadata.requested_tag == TAG
    assert resolved.metadata.resolved_tag == TAG


def test_air_gap_sha_mismatch_raises(tmp_path, monkeypatch):
    air_gap = tmp_path / "ag"
    air_gap.mkdir()
    (air_gap / BIN_FILENAME).write_bytes(b"x" * len(BIN_BYTES))
    (air_gap / "manifest.json").write_text(_manifest_payload())
    monkeypatch.setenv(resolver.LOCAL_DIR_ENV_VAR, str(air_gap))

    with pytest.raises(ValueError, match="sha256 mismatch"):
        resolver.resolve("ptxas", "13.3")


def test_air_gap_size_mismatch_raises(tmp_path, monkeypatch):
    air_gap = tmp_path / "ag"
    air_gap.mkdir()
    (air_gap / BIN_FILENAME).write_bytes(BIN_BYTES)
    (air_gap / "manifest.json").write_text(_manifest_payload(size=len(BIN_BYTES) + 1))
    monkeypatch.setenv(resolver.LOCAL_DIR_ENV_VAR, str(air_gap))

    with pytest.raises(ValueError, match="size mismatch"):
        resolver.resolve("ptxas", "13.3")


def test_air_gap_missing_manifest_raises(tmp_path, monkeypatch):
    air_gap = tmp_path / "ag"
    air_gap.mkdir()
    monkeypatch.setenv(resolver.LOCAL_DIR_ENV_VAR, str(air_gap))
    with pytest.raises(FileNotFoundError, match="manifest.json"):
        resolver.resolve("ptxas", "13.3")


def test_air_gap_missing_binary_raises(tmp_path, monkeypatch):
    air_gap = tmp_path / "ag"
    air_gap.mkdir()
    (air_gap / "manifest.json").write_text(_manifest_payload())
    monkeypatch.setenv(resolver.LOCAL_DIR_ENV_VAR, str(air_gap))
    with pytest.raises(FileNotFoundError, match=BIN_FILENAME):
        resolver.resolve("ptxas", "13.3")


def test_resolve_latest_then_download(fake_http, tmp_path):
    handlers, stream, jsn = fake_http
    repo = resolver._release_repo()
    handlers[f"{resolver.GH_API}/repos/{repo}/releases/latest"] = jsn({"tag_name": TAG})
    handlers[f"{resolver.GH_DL}/{repo}/releases/download/{TAG}/manifest.json"] = stream(
        _manifest_payload().encode()
    )
    handlers[f"{resolver.GH_DL}/{repo}/releases/download/{TAG}/{BIN_FILENAME}"] = stream(BIN_BYTES)

    path = resolver.resolve("ptxas", "13.3")
    assert path.read_bytes() == BIN_BYTES
    assert TAG in str(path)


def test_resolve_with_metadata_records_download(fake_http):
    handlers, stream, jsn = fake_http
    repo = resolver._release_repo()
    handlers[f"{resolver.GH_API}/repos/{repo}/releases/latest"] = jsn({"tag_name": TAG})
    handlers[f"{resolver.GH_DL}/{repo}/releases/download/{TAG}/manifest.json"] = stream(
        _manifest_payload().encode()
    )
    handlers[f"{resolver.GH_DL}/{repo}/releases/download/{TAG}/{BIN_FILENAME}"] = stream(BIN_BYTES)

    resolved = resolver.resolve_with_metadata("ptxas", "13.3")

    assert resolved.path.read_bytes() == BIN_BYTES
    assert resolved.metadata.as_dict()["compiler"] == "ptxas"
    assert resolved.metadata.requested_tag == "latest"
    assert resolved.metadata.resolved_tag == TAG
    assert resolved.metadata.source == "github_release"
    assert resolved.metadata.size_bytes == len(BIN_BYTES)
    assert resolved.metadata.description == "PTXAS 13.3 default controls"


def test_cache_hit_skips_download(fake_http, tmp_path):
    handlers, stream, jsn = fake_http
    repo = resolver._release_repo()
    handlers[f"{resolver.GH_API}/repos/{repo}/releases/latest"] = jsn({"tag_name": TAG})
    handlers[f"{resolver.GH_DL}/{repo}/releases/download/{TAG}/manifest.json"] = stream(
        _manifest_payload().encode()
    )
    bin_url = f"{resolver.GH_DL}/{repo}/releases/download/{TAG}/{BIN_FILENAME}"
    handlers[bin_url] = stream(BIN_BYTES)

    p1 = resolver.resolve("ptxas", "13.3")
    # Replace the bin handler with one that fails if called again.
    handlers[bin_url] = stream(b"SHOULD-NOT-FETCH", status=500)
    p2 = resolver.resolve("ptxas", "13.3")
    assert p1 == p2


def test_cache_hit_metadata_source(fake_http):
    handlers, stream, jsn = fake_http
    repo = resolver._release_repo()
    handlers[f"{resolver.GH_API}/repos/{repo}/releases/latest"] = jsn({"tag_name": TAG})
    handlers[f"{resolver.GH_DL}/{repo}/releases/download/{TAG}/manifest.json"] = stream(
        _manifest_payload().encode()
    )
    bin_url = f"{resolver.GH_DL}/{repo}/releases/download/{TAG}/{BIN_FILENAME}"
    handlers[bin_url] = stream(BIN_BYTES)

    resolver.resolve("ptxas", "13.3")
    handlers[bin_url] = stream(b"SHOULD-NOT-FETCH", status=500)
    resolved = resolver.resolve_with_metadata("ptxas", "13.3")

    assert resolved.metadata.source == "cache"


def test_corrupt_cache_redownloads(fake_http):
    handlers, stream, jsn = fake_http
    repo = resolver._release_repo()
    handlers[f"{resolver.GH_API}/repos/{repo}/releases/latest"] = jsn({"tag_name": TAG})
    handlers[f"{resolver.GH_DL}/{repo}/releases/download/{TAG}/manifest.json"] = stream(
        _manifest_payload().encode()
    )
    bin_url = f"{resolver.GH_DL}/{repo}/releases/download/{TAG}/{BIN_FILENAME}"
    handlers[bin_url] = stream(BIN_BYTES)

    path = resolver.resolve("ptxas", "13.3")
    path.write_bytes(b"stale")
    handlers[bin_url] = stream(BIN_BYTES)

    assert resolver.resolve("ptxas", "13.3").read_bytes() == BIN_BYTES


def test_sha_mismatch_on_download_raises_and_cleans_up(fake_http):
    handlers, stream, jsn = fake_http
    repo = resolver._release_repo()
    handlers[f"{resolver.GH_API}/repos/{repo}/releases/latest"] = jsn({"tag_name": TAG})
    handlers[f"{resolver.GH_DL}/{repo}/releases/download/{TAG}/manifest.json"] = stream(
        _manifest_payload().encode()
    )
    handlers[f"{resolver.GH_DL}/{repo}/releases/download/{TAG}/{BIN_FILENAME}"] = stream(
        b"x" * len(BIN_BYTES)
    )

    with pytest.raises(ValueError, match="sha256 mismatch"):
        resolver.resolve("ptxas", "13.3")
    cached = resolver._cache_path(TAG, BIN_SHA, BIN_FILENAME)
    assert not cached.exists()


def test_size_mismatch_on_download_raises_and_cleans_up(fake_http):
    handlers, stream, jsn = fake_http
    repo = resolver._release_repo()
    handlers[f"{resolver.GH_API}/repos/{repo}/releases/latest"] = jsn({"tag_name": TAG})
    handlers[f"{resolver.GH_DL}/{repo}/releases/download/{TAG}/manifest.json"] = stream(
        _manifest_payload(size=len(BIN_BYTES) + 1).encode()
    )
    handlers[f"{resolver.GH_DL}/{repo}/releases/download/{TAG}/{BIN_FILENAME}"] = stream(
        BIN_BYTES
    )

    with pytest.raises(ValueError, match="size mismatch"):
        resolver.resolve("ptxas", "13.3")
    cached = resolver._cache_path(TAG, BIN_SHA, BIN_FILENAME)
    assert not cached.exists()


def test_latest_resolution_memoized_per_process(fake_http):
    handlers, stream, jsn = fake_http
    repo = resolver._release_repo()
    handlers[f"{resolver.GH_API}/repos/{repo}/releases/latest"] = jsn({"tag_name": TAG})
    handlers[f"{resolver.GH_DL}/{repo}/releases/download/{TAG}/manifest.json"] = stream(
        _manifest_payload().encode()
    )
    handlers[f"{resolver.GH_DL}/{repo}/releases/download/{TAG}/{BIN_FILENAME}"] = stream(BIN_BYTES)

    resolver.resolve("ptxas", "13.3")
    # Replace the latest endpoint with a 500. If lru_cache works, the second
    # resolve() never re-enters _resolve_latest_tag and so never sees the 500.
    handlers[f"{resolver.GH_API}/repos/{repo}/releases/latest"] = jsn(
        {"tag_name": "WRONG"}, status=500
    )
    resolver.resolve("ptxas", "13.3")  # would raise if cache miss re-fetched


def test_fresh_latest_disk_cache_skips_github_latest(fake_http, monkeypatch):
    _set_now(monkeypatch)
    handlers, stream, _jsn = fake_http
    repo = resolver._release_repo()
    _write_latest_cache(repo, "", TAG, fetched_at=NOW)
    _add_asset_routes(handlers, stream, repo, TAG)

    path = resolver.resolve("ptxas", "13.3")

    assert path.read_bytes() == BIN_BYTES
    assert TAG in str(path)


def test_missing_latest_disk_cache_refreshes_and_writes(fake_http, monkeypatch):
    _set_now(monkeypatch)
    handlers, stream, jsn = fake_http
    repo = resolver._release_repo()
    handlers[f"{resolver.GH_API}/repos/{repo}/releases/latest"] = jsn({"tag_name": TAG})
    _add_asset_routes(handlers, stream, repo, TAG)

    resolver.resolve("ptxas", "13.3")

    cache = json.loads(resolver._latest_tag_cache_path(repo, "").read_text())
    assert cache["repo"] == repo
    assert cache["tag_prefix"] == ""
    assert cache["resolved_tag"] == TAG
    assert cache["fetched_at"] == NOW.isoformat()
    assert cache["ttl_days"] == resolver.DEFAULT_LATEST_TAG_TTL_DAYS


def test_expired_latest_disk_cache_refreshes_and_updates(fake_http, monkeypatch):
    _set_now(monkeypatch)
    handlers, stream, jsn = fake_http
    repo = resolver._release_repo()
    stale_time = NOW - dt.timedelta(days=8)
    _write_latest_cache(repo, "", OLD_TAG, fetched_at=stale_time)
    handlers[f"{resolver.GH_API}/repos/{repo}/releases/latest"] = jsn({"tag_name": TAG})
    _add_asset_routes(handlers, stream, repo, TAG)

    path = resolver.resolve("ptxas", "13.3")

    cache = json.loads(resolver._latest_tag_cache_path(repo, "").read_text())
    assert path.read_bytes() == BIN_BYTES
    assert TAG in str(path)
    assert cache["resolved_tag"] == TAG
    assert cache["fetched_at"] == NOW.isoformat()


@pytest.mark.parametrize("status", [403, 429, 500])
def test_expired_latest_disk_cache_uses_stale_on_github_failure(
    fake_http,
    monkeypatch,
    status,
):
    _set_now(monkeypatch)
    handlers, stream, jsn = fake_http
    repo = resolver._release_repo()
    stale_time = NOW - dt.timedelta(days=8)
    _write_latest_cache(repo, "", OLD_TAG, fetched_at=stale_time)
    handlers[f"{resolver.GH_API}/repos/{repo}/releases/latest"] = jsn({"message": "nope"}, status)
    _add_asset_routes(handlers, stream, repo, OLD_TAG)

    with pytest.warns(RuntimeWarning, match="Using stale cached latest search-space tag"):
        path = resolver.resolve("ptxas", "13.3")

    assert path.read_bytes() == BIN_BYTES
    assert OLD_TAG in str(path)


def test_github_failure_without_latest_disk_cache_raises(fake_http):
    handlers, _stream, jsn = fake_http
    repo = resolver._release_repo()
    handlers[f"{resolver.GH_API}/repos/{repo}/releases/latest"] = jsn({"message": "nope"}, 500)

    with pytest.raises(requests.HTTPError):
        resolver.resolve("ptxas", "13.3")


def test_latest_disk_cache_can_be_disabled(fake_http, monkeypatch):
    _set_now(monkeypatch)
    monkeypatch.setenv(resolver.LATEST_TAG_TTL_DAYS_ENV_VAR, "0")
    handlers, stream, jsn = fake_http
    repo = resolver._release_repo()
    _write_latest_cache(repo, "", OLD_TAG, fetched_at=NOW)
    handlers[f"{resolver.GH_API}/repos/{repo}/releases/latest"] = jsn({"tag_name": TAG})
    _add_asset_routes(handlers, stream, repo, TAG)

    path = resolver.resolve("ptxas", "13.3")

    cache = json.loads(resolver._latest_tag_cache_path(repo, "").read_text())
    assert path.read_bytes() == BIN_BYTES
    assert TAG in str(path)
    assert cache["resolved_tag"] == OLD_TAG


@pytest.mark.parametrize("value", ["nope", "-1", "inf"])
def test_invalid_latest_cache_ttl_raises(monkeypatch, value):
    monkeypatch.setenv(resolver.LATEST_TAG_TTL_DAYS_ENV_VAR, value)

    with pytest.raises(ValueError, match=resolver.LATEST_TAG_TTL_DAYS_ENV_VAR):
        resolver.resolve("ptxas", "13.3")


def test_latest_disk_cache_key_includes_repo_and_prefix():
    default = resolver._latest_tag_cache_path("NVIDIA/CompileIQ", "search-spaces-")
    other_repo = resolver._latest_tag_cache_path("example/search-spaces-test", "search-spaces-")
    other_prefix = resolver._latest_tag_cache_path("NVIDIA/CompileIQ", "booster-packs-")

    assert default != other_repo
    assert default != other_prefix
    assert other_repo != other_prefix


def test_air_gap_ignores_latest_cache_ttl(tmp_path, monkeypatch):
    air_gap = tmp_path / "ag"
    air_gap.mkdir()
    (air_gap / BIN_FILENAME).write_bytes(BIN_BYTES)
    (air_gap / "manifest.json").write_text(_manifest_payload())
    monkeypatch.setenv(resolver.LOCAL_DIR_ENV_VAR, str(air_gap))
    monkeypatch.setenv(resolver.LATEST_TAG_TTL_DAYS_ENV_VAR, "invalid")
    monkeypatch.setattr(
        resolver.requests,
        "get",
        lambda *a, **kw: pytest.fail(f"network used despite {resolver.LOCAL_DIR_ENV_VAR}"),
    )

    path = resolver.resolve("ptxas", "13.3")

    assert path == air_gap / BIN_FILENAME


def test_default_tag_prefix_used_when_unset(fake_http, monkeypatch):
    """When CIQ_SS_TAG_PREFIX is unset, the resolver falls back to the
    default 'search-spaces-' prefix so it picks search-space releases
    rather than wheel releases on a single-repo topology."""
    handlers, stream, jsn = fake_http
    monkeypatch.delenv(resolver.SEARCH_SPACE_TAG_PREFIX_ENV_VAR, raising=False)
    repo = resolver._release_repo()
    handlers[f"{resolver.GH_API}/repos/{repo}/releases?per_page=100&page=1"] = jsn(
        [
            {"tag_name": "v0.5.0", "draft": False},
            {"tag_name": TAG, "draft": False},
        ]
    )
    handlers[f"{resolver.GH_DL}/{repo}/releases/download/{TAG}/manifest.json"] = stream(
        _manifest_payload().encode()
    )
    handlers[f"{resolver.GH_DL}/{repo}/releases/download/{TAG}/{BIN_FILENAME}"] = stream(BIN_BYTES)
    path = resolver.resolve("ptxas", "13.3")
    assert TAG in str(path)


def test_tag_prefix_filters_releases(fake_http, monkeypatch):
    handlers, stream, jsn = fake_http
    monkeypatch.setenv(resolver.SEARCH_SPACE_TAG_PREFIX_ENV_VAR, "search-spaces-")
    repo = resolver._release_repo()
    handlers[f"{resolver.GH_API}/repos/{repo}/releases?per_page=100&page=1"] = jsn(
        [
            {"tag_name": "v0.5.0", "draft": False},
            {"tag_name": "search-spaces-2026.05.01-rc1", "draft": False, "prerelease": True},
            {"tag_name": TAG, "draft": False},
            {"tag_name": "search-spaces-2026.01.01", "draft": False},
        ]
    )
    handlers[f"{resolver.GH_DL}/{repo}/releases/download/{TAG}/manifest.json"] = stream(
        _manifest_payload().encode()
    )
    handlers[f"{resolver.GH_DL}/{repo}/releases/download/{TAG}/{BIN_FILENAME}"] = stream(BIN_BYTES)

    path = resolver.resolve("ptxas", "13.3")
    assert TAG in str(path)


def test_tag_prefix_paginates_releases(fake_http, monkeypatch):
    handlers, stream, jsn = fake_http
    monkeypatch.setenv(resolver.SEARCH_SPACE_TAG_PREFIX_ENV_VAR, "search-spaces-")
    repo = resolver._release_repo()
    handlers[f"{resolver.GH_API}/repos/{repo}/releases?per_page=100&page=1"] = jsn(
        [{"tag_name": "v0.5.0", "draft": False}]
    )
    handlers[f"{resolver.GH_API}/repos/{repo}/releases?per_page=100&page=2"] = jsn(
        [{"tag_name": TAG, "draft": False}]
    )
    handlers[f"{resolver.GH_DL}/{repo}/releases/download/{TAG}/manifest.json"] = stream(
        _manifest_payload().encode()
    )
    handlers[f"{resolver.GH_DL}/{repo}/releases/download/{TAG}/{BIN_FILENAME}"] = stream(BIN_BYTES)

    path = resolver.resolve("ptxas", "13.3")
    assert TAG in str(path)


def test_manifest_404_propagates(fake_http):
    """If the release has no manifest.json, the HTTPError is not swallowed."""
    handlers, stream, jsn = fake_http
    repo = resolver._release_repo()
    handlers[f"{resolver.GH_API}/repos/{repo}/releases/latest"] = jsn({"tag_name": TAG})
    handlers[f"{resolver.GH_DL}/{repo}/releases/download/{TAG}/manifest.json"] = stream(
        b"", status=404
    )
    with pytest.raises(requests.HTTPError):
        resolver.resolve("ptxas", "13.3")


def test_manifest_tag_mismatch_raises(fake_http):
    handlers, stream, jsn = fake_http
    repo = resolver._release_repo()
    handlers[f"{resolver.GH_API}/repos/{repo}/releases/latest"] = jsn({"tag_name": TAG})
    handlers[f"{resolver.GH_DL}/{repo}/releases/download/{TAG}/manifest.json"] = stream(
        _manifest_payload(tag="search-spaces-wrong").encode()
    )
    with pytest.raises(ValueError, match="Manifest tag mismatch"):
        resolver.resolve("ptxas", "13.3")


def test_release_repo_env_override(fake_http, monkeypatch):
    monkeypatch.setenv(resolver.RELEASE_REPO_ENV_VAR, "example/search-spaces-test")
    handlers, stream, jsn = fake_http
    repo = "example/search-spaces-test"
    handlers[f"{resolver.GH_API}/repos/{repo}/releases/latest"] = jsn({"tag_name": TAG})
    handlers[f"{resolver.GH_DL}/{repo}/releases/download/{TAG}/manifest.json"] = stream(
        _manifest_payload().encode()
    )
    handlers[f"{resolver.GH_DL}/{repo}/releases/download/{TAG}/{BIN_FILENAME}"] = stream(BIN_BYTES)

    path = resolver.resolve("ptxas", "13.3")
    assert path.read_bytes() == BIN_BYTES
