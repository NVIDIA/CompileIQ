"""
Resolve a logical search-space request to a local file path.

The resolver consults a ``manifest.json`` published as a release asset to map
``(compiler, compiler_version, variant)`` to a concrete binary, downloads +
caches it under ``~/.cache/compileiq/<tag>/``, and verifies its size + sha256.

Two ways to bypass the GitHub release path:
  - Set ``CIQ_SEARCH_SPACES_DIR`` to a directory containing ``manifest.json``
    plus its referenced binaries; the resolver reads from that directory and
    skips the network entirely (air-gapped/CI mirror use).
  - Use ``LocalSearchSpaceBin(path=...)`` from ``compileiq.search_spaces.compilers``
    when you have a single binary on disk and want to skip manifest lookup.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import datetime as dt
import functools
import hashlib
import json
import math
import os
import pathlib
from typing import Literal
import warnings

import requests

from compileiq.search_spaces.manifest import SearchSpaceEntry, SearchSpaceManifestModel

# Environment variable used to specify the GitHub repository that hosts
# search-space releases, e.g. "NVIDIA/CompileIQ" or a staging repo.
RELEASE_REPO_ENV_VAR = "CIQ_SEARCH_SPACES_REPO"
DEFAULT_RELEASE_REPO = "NVIDIA/CompileIQ"

# Environment variable used to point at an offline mirror directory containing
# manifest.json plus the referenced .bin assets.
LOCAL_DIR_ENV_VAR = "CIQ_SEARCH_SPACES_DIR"

# Environment variable used to choose which GitHub release tags are considered
# when resolving tag="latest" in repositories with multiple release streams.
SEARCH_SPACE_TAG_PREFIX_ENV_VAR = "CIQ_SS_TAG_PREFIX"

# In single-repo topology, the wheel and search-space tag namespaces are
# partitioned by prefix. The default selects search-space releases when
# resolving "latest"; override via CIQ_SS_TAG_PREFIX.
DEFAULT_SEARCH_SPACE_TAG_PREFIX = "search-spaces-"

# Environment variable used to choose how long a resolved "latest" search-space
# release tag can be reused from disk before checking GitHub again. Set to "0"
# to disable the on-disk latest cache.
LATEST_TAG_TTL_DAYS_ENV_VAR = "CIQ_SS_LATEST_TTL_DAYS"
DEFAULT_LATEST_TAG_TTL_DAYS = 7.0

# GitHub API host for release metadata and GitHub web host for release-asset
# downloads.
GH_API = "https://api.github.com"
GH_DL = "https://github.com"

# A minute is conservative for release asset downloads while still failing
# promptly enough for scripts and notebooks to surface network issues.
_HTTP_TIMEOUT_SEC = 60
# 64 KiB download chunks keep streaming responsive without tiny write calls.
_CHUNK_BYTES = 1 << 16
# 1 MiB hash chunks avoid loading large artifacts into memory.
_HASH_CHUNK_BYTES = 1 << 20
_LATEST_TAG_CACHE_DIR = "latest-tags"


@dataclass(frozen=True)
class SearchSpaceResolutionMetadata:
    """Traceability metadata for a resolved search-space artifact."""

    compiler: str
    compiler_version: str
    variant: str
    description: str | None
    requested_tag: str
    resolved_tag: str
    filename: str
    sha256: str
    size_bytes: int
    source: Literal["github_release", "cache", "CIQ_SEARCH_SPACES_DIR"]
    path: str

    def as_dict(self) -> dict[str, str | int | None]:
        return asdict(self)


@dataclass(frozen=True)
class ResolvedSearchSpace:
    """Resolved local path plus metadata needed for support/reproducibility."""

    path: pathlib.Path
    metadata: SearchSpaceResolutionMetadata


def _release_repo() -> str:
    return os.getenv(RELEASE_REPO_ENV_VAR, DEFAULT_RELEASE_REPO)


def _cache_root() -> pathlib.Path:
    # Read at call time so the test suite's monkeypatch on
    # compileiq.config.const._CACHE_DIR is honored.
    from compileiq.config import const

    return pathlib.Path(const._CACHE_DIR)


def _utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _latest_tag_ttl_days() -> float:
    """Return the configured latest-tag cache TTL in days."""
    raw = os.getenv(LATEST_TAG_TTL_DAYS_ENV_VAR, str(DEFAULT_LATEST_TAG_TTL_DAYS))
    try:
        ttl_days = float(raw)
    except ValueError as exc:
        raise ValueError(
            f"{LATEST_TAG_TTL_DAYS_ENV_VAR} must be a non-negative number of days, got {raw!r}"
        ) from exc
    if not math.isfinite(ttl_days) or ttl_days < 0:
        raise ValueError(
            f"{LATEST_TAG_TTL_DAYS_ENV_VAR} must be a non-negative number of days, got {raw!r}"
        )
    return ttl_days


def _latest_tag_cache_path(repo: str, tag_prefix: str) -> pathlib.Path:
    """Return the repo/prefix-specific cache path for a latest-tag record."""
    key = hashlib.sha256(f"{repo}\0{tag_prefix}".encode("utf-8")).hexdigest()
    return _cache_root() / _LATEST_TAG_CACHE_DIR / f"{key}.json"


def _read_latest_tag_cache(repo: str, tag_prefix: str) -> dict[str, object] | None:
    """Load a matching latest-tag record, ignoring missing or invalid files."""
    path = _latest_tag_cache_path(repo, tag_prefix)
    try:
        payload = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("repo") != repo or payload.get("tag_prefix") != tag_prefix:
        return None
    if not isinstance(payload.get("resolved_tag"), str):
        return None
    if not isinstance(payload.get("fetched_at"), str):
        return None
    return payload


def _latest_tag_cache_age(record: dict[str, object]) -> dt.timedelta | None:
    """Return the cache record age, or None when fetched_at is invalid."""
    try:
        fetched_at = dt.datetime.fromisoformat(str(record["fetched_at"]))
    except (KeyError, ValueError):
        return None
    if fetched_at.tzinfo is None:
        fetched_at = fetched_at.replace(tzinfo=dt.timezone.utc)
    return _utc_now() - fetched_at


def _latest_tag_cache_is_fresh(record: dict[str, object], ttl_days: float) -> bool:
    """Return whether the cache record is within the configured TTL."""
    age = _latest_tag_cache_age(record)
    if age is None:
        return False
    return age <= dt.timedelta(days=ttl_days)


def _write_latest_tag_cache(
    repo: str,
    tag_prefix: str,
    resolved_tag: str,
    ttl_days: float,
) -> None:
    """Persist the resolved latest tag atomically for later cold processes."""
    path = _latest_tag_cache_path(repo, tag_prefix)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    payload = {
        "repo": repo,
        "tag_prefix": tag_prefix,
        "resolved_tag": resolved_tag,
        "fetched_at": _utc_now().isoformat(),
        "ttl_days": ttl_days,
    }
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def _fetch_latest_tag_from_github(repo: str, tag_prefix: str) -> str:
    """Resolve latest through GitHub without consulting the on-disk cache."""
    if tag_prefix:
        page = 1
        while True:
            r = requests.get(
                f"{GH_API}/repos/{repo}/releases?per_page=100&page={page}",
                timeout=_HTTP_TIMEOUT_SEC,
            )
            r.raise_for_status()
            releases = r.json()
            if not releases:
                break
            for rel in releases:
                if (
                    rel["tag_name"].startswith(tag_prefix)
                    and not rel.get("draft", False)
                    and not rel.get("prerelease", False)
                ):
                    return rel["tag_name"]
            page += 1
        raise RuntimeError(f"No release with tag prefix {tag_prefix!r} on {repo}")
    r = requests.get(f"{GH_API}/repos/{repo}/releases/latest", timeout=_HTTP_TIMEOUT_SEC)
    r.raise_for_status()
    return r.json()["tag_name"]


@functools.lru_cache(maxsize=8)
def _resolve_latest_tag(repo: str, tag_prefix: str) -> str:
    """Resolve "latest" to a concrete release tag.

    A fresh on-disk record is returned without a GitHub request. Missing or
    expired records are refreshed from GitHub, then rewritten for future cold
    processes. With no prefix, refresh uses GitHub's ``/releases/latest``
    endpoint. With a prefix, refresh walks ``/releases`` and returns the first
    non-draft, non-prerelease release whose tag begins with the prefix.
    """
    ttl_days = _latest_tag_ttl_days()
    cached = _read_latest_tag_cache(repo, tag_prefix)
    if ttl_days > 0 and cached is not None and _latest_tag_cache_is_fresh(cached, ttl_days):
        return str(cached["resolved_tag"])

    try:
        resolved_tag = _fetch_latest_tag_from_github(repo, tag_prefix)
    except Exception:
        if ttl_days > 0 and cached is not None and _latest_tag_cache_age(cached) is not None:
            warnings.warn(
                f"Using stale cached latest search-space tag {cached['resolved_tag']!r} "
                "because GitHub latest-tag resolution failed.",
                RuntimeWarning,
                stacklevel=2,
            )
            return str(cached["resolved_tag"])
        raise

    if ttl_days > 0:
        _write_latest_tag_cache(repo, tag_prefix, resolved_tag, ttl_days)
    return resolved_tag


def _cache_path(tag: str, sha256: str, filename: str) -> pathlib.Path:
    return _cache_root() / tag / f"{sha256[:12]}_{filename}"


def _sha256_of(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(_HASH_CHUNK_BYTES), b""):
            h.update(chunk)
    return h.hexdigest()


def _verify_asset(path: pathlib.Path, entry: SearchSpaceEntry) -> None:
    actual_size = path.stat().st_size
    if actual_size != entry.size_bytes:
        raise ValueError(
            f"size mismatch for {path}: expected {entry.size_bytes}, got {actual_size}"
        )
    actual_sha = _sha256_of(path)
    if actual_sha != entry.sha256:
        raise ValueError(f"sha256 mismatch for {path}: expected {entry.sha256}, got {actual_sha}")


def _download(url: str, dest: pathlib.Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=_HTTP_TIMEOUT_SEC) as r:
        r.raise_for_status()
        with tmp.open("wb") as f:
            for chunk in r.iter_content(_CHUNK_BYTES):
                if chunk:
                    f.write(chunk)
    tmp.replace(dest)


def _resolved(
    path: pathlib.Path,
    entry: SearchSpaceEntry,
    requested_tag: str,
    resolved_tag: str,
    source: Literal["github_release", "cache", "CIQ_SEARCH_SPACES_DIR"],
) -> ResolvedSearchSpace:
    return ResolvedSearchSpace(
        path=path,
        metadata=SearchSpaceResolutionMetadata(
            compiler=entry.compiler,
            compiler_version=entry.compiler_version,
            variant=entry.variant,
            description=entry.description,
            requested_tag=requested_tag,
            resolved_tag=resolved_tag,
            filename=entry.filename,
            sha256=entry.sha256,
            size_bytes=entry.size_bytes,
            source=source,
            path=str(path),
        ),
    )


def _fetch_manifest(repo: str, tag: str) -> SearchSpaceManifestModel:
    cache = _cache_root() / tag / "manifest.json"
    if not cache.exists():
        url = f"{GH_DL}/{repo}/releases/download/{tag}/manifest.json"
        _download(url, cache)
    manifest = SearchSpaceManifestModel.model_validate_json(cache.read_text())
    if manifest.tag != tag:
        raise ValueError(f"Manifest tag mismatch: expected {tag!r}, got {manifest.tag!r}")
    return manifest


def _local_lookup(
    compiler: str,
    compiler_version: str,
    variant: str,
    requested_tag: str,
) -> ResolvedSearchSpace | None:
    """Read manifest + binaries from CIQ_SEARCH_SPACES_DIR if set; else None."""
    root = os.getenv(LOCAL_DIR_ENV_VAR)
    if not root:
        return None
    root_path = pathlib.Path(root)
    manifest_path = root_path / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"{LOCAL_DIR_ENV_VAR}={root} but manifest.json is missing. "
            "An air-gapped directory must contain manifest.json plus the .bin assets."
        )
    manifest = SearchSpaceManifestModel.model_validate_json(manifest_path.read_text())
    if requested_tag != "latest" and manifest.tag != requested_tag:
        raise ValueError(
            f"Local manifest tag mismatch: expected {requested_tag!r}, got {manifest.tag!r}"
        )
    entry = manifest.find(compiler, compiler_version, variant)
    bin_path = root_path / entry.filename
    if not bin_path.exists():
        raise FileNotFoundError(
            f"Manifest references {entry.filename} but the file is missing in {root}"
        )
    _verify_asset(bin_path, entry)
    return _resolved(
        bin_path,
        entry,
        requested_tag=requested_tag,
        resolved_tag=manifest.tag,
        source=LOCAL_DIR_ENV_VAR,
    )


def resolve(
    compiler: str,
    compiler_version: str,
    variant: str = "default",
    tag: str = "latest",
) -> pathlib.Path:
    """Resolve a search-space request to a local file path."""
    return resolve_with_metadata(
        compiler=compiler,
        compiler_version=compiler_version,
        variant=variant,
        tag=tag,
    ).path


def resolve_with_metadata(
    compiler: str,
    compiler_version: str,
    variant: str = "default",
    tag: str = "latest",
) -> ResolvedSearchSpace:
    """Resolve a search-space request to a local file path.

    Lookup order:
      1. ``CIQ_SEARCH_SPACES_DIR`` if set (air-gapped).
      2. Manifest from the GitHub release at ``tag`` (resolved via API if
         ``tag == "latest"``).
    """
    requested_tag = tag
    local = _local_lookup(compiler, compiler_version, variant, requested_tag)
    if local is not None:
        return local

    repo = _release_repo()
    if tag == "latest":
        tag = _resolve_latest_tag(
            repo,
            os.getenv(SEARCH_SPACE_TAG_PREFIX_ENV_VAR, DEFAULT_SEARCH_SPACE_TAG_PREFIX),
        )

    manifest = _fetch_manifest(repo, tag)
    entry = manifest.find(compiler, compiler_version, variant)
    cached = _cache_path(tag, entry.sha256, entry.filename)
    if cached.exists():
        try:
            _verify_asset(cached, entry)
            return _resolved(cached, entry, requested_tag, tag, source="cache")
        except ValueError:
            # Treat a corrupt or stale cache entry as a cache miss and replace it.
            pass

    url = f"{GH_DL}/{repo}/releases/download/{tag}/{entry.filename}"
    _download(url, cached)
    try:
        _verify_asset(cached, entry)
    except ValueError:
        cached.unlink(missing_ok=True)
        raise
    return _resolved(cached, entry, requested_tag, tag, source="github_release")
