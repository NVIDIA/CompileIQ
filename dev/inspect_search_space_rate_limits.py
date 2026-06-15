#!/usr/bin/env python3
"""Inspect search-space release rate-limit exposure.

The online search-space resolver stores manifests and binaries on disk, but it
must resolve tag="latest" to a concrete release tag before it can look in that
cache. This script shows which GitHub REST API endpoint is involved, the
current rate-limit bucket, and whether the resolved release is already cached.

By default the script only calls the GitHub REST API for rate metadata and
latest-tag resolution. It does not download manifests or search-space binaries.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import datetime as dt
import hashlib
import json
import math
import os
import pathlib
import sys
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


RELEASE_REPO_ENV_VAR = "CIQ_SEARCH_SPACES_REPO"
DEFAULT_RELEASE_REPO = "NVIDIA/CompileIQ"
LOCAL_DIR_ENV_VAR = "CIQ_SEARCH_SPACES_DIR"
SEARCH_SPACE_TAG_PREFIX_ENV_VAR = "CIQ_SS_TAG_PREFIX"
DEFAULT_SEARCH_SPACE_TAG_PREFIX = "search-spaces-"
LATEST_TAG_TTL_DAYS_ENV_VAR = "CIQ_SS_LATEST_TTL_DAYS"
DEFAULT_LATEST_TAG_TTL_DAYS = 7.0
GH_API = "https://api.github.com"
HTTP_TIMEOUT_SEC = 60
HASH_CHUNK_BYTES = 1 << 20
LATEST_TAG_CACHE_DIR = "latest-tags"


@dataclass(frozen=True)
class RateSnapshot:
    limit: int | None
    remaining: int | None
    reset_epoch: int | None
    used: int | None
    resource: str | None


@dataclass(frozen=True)
class LatestResolution:
    tag: str
    api_calls: int
    endpoint_kind: str
    snapshot: RateSnapshot | None
    source: str


@dataclass(frozen=True)
class LatestCacheStatus:
    path: pathlib.Path
    state: str
    ttl_days: float
    resolved_tag: str | None
    age: dt.timedelta | None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect GitHub API rate-limit exposure for search-space retrieval."
    )
    parser.add_argument(
        "--repo",
        default=os.getenv(RELEASE_REPO_ENV_VAR, DEFAULT_RELEASE_REPO),
        help=(
            f"GitHub owner/repo to inspect. Defaults to {RELEASE_REPO_ENV_VAR} "
            f"or {DEFAULT_RELEASE_REPO}."
        ),
    )
    parser.add_argument(
        "--tag",
        default="latest",
        help="Requested search-space release tag. Use a concrete tag to avoid latest lookup.",
    )
    parser.add_argument(
        "--tag-prefix",
        default=os.getenv(
            SEARCH_SPACE_TAG_PREFIX_ENV_VAR,
            DEFAULT_SEARCH_SPACE_TAG_PREFIX,
        ),
        help=(
            f"Prefix used when resolving latest. Defaults to "
            f"{SEARCH_SPACE_TAG_PREFIX_ENV_VAR} or "
            f"{DEFAULT_SEARCH_SPACE_TAG_PREFIX!r}."
        ),
    )
    parser.add_argument("--compiler", default="ptxas", help="Compiler key to inspect.")
    parser.add_argument("--compiler-version", default="13.3", help="Compiler version to inspect.")
    parser.add_argument("--variant", default="default", help="Search-space variant to inspect.")
    parser.add_argument(
        "--cache-root",
        type=pathlib.Path,
        default=pathlib.Path.home() / ".cache" / "compileiq",
        help="CompileIQ cache root to inspect.",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=5,
        help="Maximum release-list pages to scan when resolving latest with a prefix.",
    )
    parser.add_argument(
        "--no-network",
        action="store_true",
        help="Skip GitHub API calls and only explain the local/cache behavior.",
    )
    parser.add_argument(
        "--hash-cache",
        action="store_true",
        help="Hash a cached binary to verify sha256. Size is checked either way.",
    )
    return parser.parse_args()


def _github_token() -> str | None:
    for name in ("GITHUB_TOKEN", "GH_TOKEN"):
        token = os.getenv(name)
        if token:
            return token
    return None


def _latest_tag_ttl_days() -> float:
    """Return the resolver's latest-tag cache TTL in days."""
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


def _latest_tag_cache_path(cache_root: pathlib.Path, repo: str, tag_prefix: str) -> pathlib.Path:
    """Return the resolver-compatible latest-tag cache path for repo/prefix."""
    key = hashlib.sha256(f"{repo}\0{tag_prefix}".encode("utf-8")).hexdigest()
    return cache_root / LATEST_TAG_CACHE_DIR / f"{key}.json"


def _read_latest_cache(path: pathlib.Path, repo: str, tag_prefix: str) -> dict[str, object] | None:
    """Load a matching latest-tag cache record, ignoring malformed records."""
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


def _latest_cache_age(record: dict[str, object]) -> dt.timedelta | None:
    """Return a latest-tag cache record's age, if its timestamp is parseable."""
    try:
        fetched_at = dt.datetime.fromisoformat(str(record["fetched_at"]))
    except (KeyError, ValueError):
        return None
    if fetched_at.tzinfo is None:
        fetched_at = fetched_at.replace(tzinfo=dt.timezone.utc)
    return dt.datetime.now(tz=dt.timezone.utc) - fetched_at


def _latest_cache_status(
    cache_root: pathlib.Path,
    repo: str,
    tag_prefix: str,
    ttl_days: float,
) -> LatestCacheStatus:
    """Classify the latest-tag cache as disabled, missing, invalid, fresh, or expired."""
    path = _latest_tag_cache_path(cache_root, repo, tag_prefix)
    if ttl_days == 0:
        return LatestCacheStatus(path, "disabled", ttl_days, None, None)

    record = _read_latest_cache(path, repo, tag_prefix)
    if record is None:
        return LatestCacheStatus(path, "missing", ttl_days, None, None)

    age = _latest_cache_age(record)
    if age is None:
        return LatestCacheStatus(path, "invalid", ttl_days, None, None)

    state = "fresh" if age <= dt.timedelta(days=ttl_days) else "expired"
    return LatestCacheStatus(path, state, ttl_days, str(record["resolved_tag"]), age)


def _write_latest_cache(
    cache_root: pathlib.Path,
    repo: str,
    tag_prefix: str,
    resolved_tag: str,
    ttl_days: float,
) -> None:
    """Write the latest-tag cache in the same format used by the resolver."""
    path = _latest_tag_cache_path(cache_root, repo, tag_prefix)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    payload = {
        "repo": repo,
        "tag_prefix": tag_prefix,
        "resolved_tag": resolved_tag,
        "fetched_at": dt.datetime.now(tz=dt.timezone.utc).isoformat(),
        "ttl_days": ttl_days,
    }
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def _headers() -> dict[str, str]:
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "compileiq-search-space-rate-limit-inspector",
    }
    token = _github_token()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _rate_snapshot(headers: Any) -> RateSnapshot:
    def as_int(name: str) -> int | None:
        value = headers.get(name)
        if value is None:
            return None
        try:
            return int(str(value))
        except ValueError:
            return None

    return RateSnapshot(
        limit=as_int("X-RateLimit-Limit"),
        remaining=as_int("X-RateLimit-Remaining"),
        reset_epoch=as_int("X-RateLimit-Reset"),
        used=as_int("X-RateLimit-Used"),
        resource=headers.get("X-RateLimit-Resource"),
    )


def _get_json(headers: dict[str, str], url: str) -> tuple[Any, RateSnapshot]:
    """GET a GitHub JSON endpoint and return the response plus rate headers."""
    request = Request(url, headers=headers)
    try:
        with urlopen(request, timeout=HTTP_TIMEOUT_SEC) as response:
            body = response.read().decode("utf-8")
            return json.loads(body), _rate_snapshot(response.headers)
    except HTTPError as exc:
        if exc.code in {403, 429}:
            _print_http_limit_details(exc)
        raise


def _rate_limit(headers: dict[str, str]) -> tuple[dict[str, Any], RateSnapshot]:
    """Fetch GitHub rate-limit metadata, preferring the primary core bucket."""
    payload, snapshot = _get_json(headers, f"{GH_API}/rate_limit")
    core = payload.get("resources", {}).get("core", {})
    if core:
        snapshot = RateSnapshot(
            limit=core.get("limit"),
            remaining=core.get("remaining"),
            reset_epoch=core.get("reset"),
            used=core.get("used"),
            resource="core",
        )
    return payload, snapshot


def _resolve_latest(
    headers: dict[str, str],
    repo: str,
    tag_prefix: str,
    max_pages: int,
) -> LatestResolution:
    """Resolve latest from GitHub and report which endpoint and call count were used."""
    if tag_prefix:
        for page in range(1, max_pages + 1):
            url = f"{GH_API}/repos/{repo}/releases?per_page=100&page={page}"
            releases, snapshot = _get_json(headers, url)
            if not releases:
                break
            for release in releases:
                tag = str(release.get("tag_name", ""))
                if (
                    tag.startswith(tag_prefix)
                    and not release.get("draft", False)
                    and not release.get("prerelease", False)
                ):
                    return LatestResolution(
                        tag=tag,
                        api_calls=page,
                        endpoint_kind="/repos/{repo}/releases",
                        snapshot=snapshot,
                        source="github",
                    )
        raise RuntimeError(
            f"No non-draft, non-prerelease tag with prefix {tag_prefix!r} "
            f"found in first {max_pages} release pages."
        )

    url = f"{GH_API}/repos/{repo}/releases/latest"
    release, snapshot = _get_json(headers, url)
    return LatestResolution(
        tag=str(release["tag_name"]),
        api_calls=1,
        endpoint_kind="/repos/{repo}/releases/latest",
        snapshot=snapshot,
        source="github",
    )


def _format_reset(reset_epoch: int | None) -> str:
    if reset_epoch is None:
        return "unknown"
    reset = dt.datetime.fromtimestamp(reset_epoch, tz=dt.timezone.utc)
    now = dt.datetime.now(tz=dt.timezone.utc)
    seconds = max(0, int((reset - now).total_seconds()))
    minutes, rem = divmod(seconds, 60)
    return f"{reset.isoformat()} UTC ({minutes}m {rem}s from now)"


def _format_rate(snapshot: RateSnapshot | None) -> str:
    if snapshot is None:
        return "unknown"
    parts = [
        f"resource={snapshot.resource or 'unknown'}",
        f"remaining={_fmt_int(snapshot.remaining)}",
        f"limit={_fmt_int(snapshot.limit)}",
        f"used={_fmt_int(snapshot.used)}",
        f"reset={_format_reset(snapshot.reset_epoch)}",
    ]
    return ", ".join(parts)


def _fmt_int(value: int | None) -> str:
    return "unknown" if value is None else str(value)


def _format_age(age: dt.timedelta | None) -> str:
    if age is None:
        return "unknown"
    total_seconds = max(0, int(age.total_seconds()))
    days, rem = divmod(total_seconds, 24 * 60 * 60)
    hours, rem = divmod(rem, 60 * 60)
    minutes, _ = divmod(rem, 60)
    if days:
        return f"{days}d {hours}h {minutes}m"
    if hours:
        return f"{hours}h {minutes}m"
    return f"{minutes}m"


def _load_cached_manifest(cache_root: pathlib.Path, tag: str) -> dict[str, Any] | None:
    """Load a cached manifest for a concrete tag when it is present."""
    manifest_path = cache_root / tag / "manifest.json"
    if not manifest_path.exists():
        return None
    return json.loads(manifest_path.read_text())


def _find_entry(
    manifest: dict[str, Any],
    compiler: str,
    compiler_version: str,
    variant: str,
) -> dict[str, Any]:
    """Find the manifest entry matching the requested compiler, version, and variant."""
    entries = manifest.get("entries")
    if not isinstance(entries, list):
        raise KeyError("manifest has no entries list")
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if (
            entry.get("compiler") == compiler
            and entry.get("compiler_version") == compiler_version
            and entry.get("variant", "default") == variant
        ):
            return entry
    raise KeyError(
        f"no manifest entry for compiler={compiler!r}, "
        f"compiler_version={compiler_version!r}, variant={variant!r}"
    )


def _cached_entry_status(
    cache_root: pathlib.Path,
    tag: str,
    entry: dict[str, Any],
    hash_cache: bool,
) -> list[str]:
    """Describe whether the manifest entry's binary is present and valid locally."""
    filename = str(entry["filename"])
    sha256 = str(entry["sha256"])
    size_bytes = int(entry["size_bytes"])
    cached = cache_root / tag / f"{sha256[:12]}_{filename}"
    lines = [f"  binary cache: {cached}"]
    if not cached.exists():
        lines.append("  binary status: missing, resolver would download the asset")
        return lines

    actual_size = cached.stat().st_size
    if actual_size == size_bytes:
        lines.append(f"  binary size: ok ({actual_size} bytes)")
    else:
        lines.append(
            f"  binary size: mismatch, expected {size_bytes}, found {actual_size}"
        )

    if hash_cache:
        actual_sha = _sha256(cached)
        if actual_sha == sha256:
            lines.append("  binary sha256: ok")
        else:
            lines.append(f"  binary sha256: mismatch, expected {sha256}, found {actual_sha}")
    else:
        lines.append("  binary sha256: not checked, pass --hash-cache to verify")
    return lines


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cached_latest_candidates(
    cache_root: pathlib.Path,
    tag_prefix: str,
    compiler: str,
    compiler_version: str,
    variant: str,
) -> list[str]:
    """List cached concrete tags that could satisfy a latest request offline."""
    if not cache_root.exists():
        return []
    tags: list[str] = []
    for manifest_path in sorted(cache_root.glob("*/manifest.json")):
        tag = manifest_path.parent.name
        if tag_prefix and not tag.startswith(tag_prefix):
            continue
        try:
            manifest = json.loads(manifest_path.read_text())
            _find_entry(manifest, compiler, compiler_version, variant)
        except Exception:
            continue
        tags.append(tag)
    return tags


def _print_http_limit_details(error: HTTPError) -> None:
    """Print rate-limit headers attached to a GitHub 403 or 429 response."""
    print("GitHub returned a possible rate-limit response:", file=sys.stderr)
    print(f"  status: {error.code}", file=sys.stderr)
    print(f"  url: {error.url}", file=sys.stderr)
    print(f"  rate: {_format_rate(_rate_snapshot(error.headers))}", file=sys.stderr)
    retry_after = error.headers.get("Retry-After")
    if retry_after:
        print(f"  retry-after: {retry_after} seconds", file=sys.stderr)


def _print_settings(args: argparse.Namespace) -> None:
    token_state = "present" if _github_token() else "not present"
    print("Resolver settings")
    print(f"  repo: {args.repo}")
    print(f"  requested tag: {args.tag}")
    print(f"  tag prefix: {args.tag_prefix!r}")
    print(f"  cache root: {args.cache_root}")
    print(f"  local mirror env: {os.getenv(LOCAL_DIR_ENV_VAR) or 'not set'}")
    print(f"  GitHub token env: {token_state}")
    print()


def _print_latest_cache(status: LatestCacheStatus) -> None:
    print("Latest tag cache")
    print(f"  path: {status.path}")
    print(f"  state: {status.state}")
    print(f"  ttl: {status.ttl_days:g} days")
    print(f"  age: {_format_age(status.age)}")
    print(f"  resolved tag: {status.resolved_tag or 'unknown'}")
    would_call = status.state != "fresh"
    print(f"  would call GitHub for latest: {'yes' if would_call else 'no'}")
    print()


def _print_behavior(args: argparse.Namespace, latest: LatestResolution | None) -> None:
    print("Resolver network behavior")
    if os.getenv(LOCAL_DIR_ENV_VAR):
        print(f"  {LOCAL_DIR_ENV_VAR} is set, so the resolver should skip GitHub.")
        print()
        return

    if args.tag != "latest":
        print("  Concrete tag requested, so latest-tag GitHub API lookup is skipped.")
        print("  Manifest and binary downloads use github.com release assets only if absent.")
        print()
        return

    endpoint = latest.endpoint_kind if latest else (
        "/repos/{repo}/releases" if args.tag_prefix else "/repos/{repo}/releases/latest"
    )
    calls = latest.api_calls if latest else "at least 1"
    source = latest.source if latest else "unknown"
    print(f"  tag='latest' uses GitHub REST endpoint: {endpoint}")
    print(f"  API calls per cold Python process for latest-tag resolution: {calls}")
    print(f"  latest-tag source for this run: {source}")
    print("  The resolver memoizes that lookup only inside the current Python process.")
    print("  Disk-cached manifests/binaries are checked after latest resolves to a concrete tag.")
    print()


def _print_cache(args: argparse.Namespace, resolved_tag: str | None) -> None:
    print("Local cache")
    if resolved_tag is None:
        if args.tag == "latest":
            candidates = _cached_latest_candidates(
                args.cache_root,
                args.tag_prefix,
                args.compiler,
                args.compiler_version,
                args.variant,
            )
            if candidates:
                print("  cached candidate tags with matching manifest entry:")
                for tag in candidates:
                    print(f"    {tag}")
            else:
                print("  no cached matching manifests found")
            print("  freshness cannot be confirmed without resolving latest.")
        else:
            print("  not inspected")
        print()
        return

    manifest_path = args.cache_root / resolved_tag / "manifest.json"
    print(f"  manifest cache: {manifest_path}")
    manifest = _load_cached_manifest(args.cache_root, resolved_tag)
    if manifest is None:
        print("  manifest status: missing, resolver would download manifest.json")
        print()
        return
    print("  manifest status: present")
    entry = _find_entry(manifest, args.compiler, args.compiler_version, args.variant)
    for line in _cached_entry_status(args.cache_root, resolved_tag, entry, args.hash_cache):
        print(line)
    print()


def _print_estimate(snapshot: RateSnapshot | None, api_calls: int | None) -> None:
    print("Rate-limit estimate")
    if api_calls == 0:
        print("  latest-tag API calls per cold resolver process: 0")
        print("  GitHub REST primary quota is not consumed by this latest-tag path.")
        print()
        return
    if api_calls is None:
        print("  latest-tag API calls per cold resolver process: not measured")
        print("  With tag='latest', expect at least 1 GitHub REST call per cold process.")
        print("  Run without --no-network to calculate the current page count and quota.")
        print()
        return

    print(f"  observed GitHub rate bucket: {_format_rate(snapshot)}")
    if snapshot and snapshot.remaining is not None:
        remaining_processes = snapshot.remaining // api_calls
        print(
            "  cold resolver processes before this bucket reaches zero: "
            f"about {remaining_processes}"
        )
    else:
        print("  remaining process estimate unavailable because rate headers were absent")
    print(
        "  Note: release asset downloads are outside this REST core bucket, "
        "but GitHub/network layers can still throttle large downloads separately."
    )
    print()


def main() -> int:
    args = _parse_args()
    try:
        ttl_days = _latest_tag_ttl_days()
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    _print_settings(args)

    resolved_tag: str | None = args.tag if args.tag != "latest" else None
    latest: LatestResolution | None = None
    rate_before: RateSnapshot | None = None
    latest_cache: LatestCacheStatus | None = None
    use_latest_cache = args.tag == "latest" and not os.getenv(LOCAL_DIR_ENV_VAR)
    if use_latest_cache:
        latest_cache = _latest_cache_status(args.cache_root, args.repo, args.tag_prefix, ttl_days)
        _print_latest_cache(latest_cache)
        if latest_cache.state == "fresh":
            resolved_tag = latest_cache.resolved_tag

    if args.no_network:
        print("GitHub rate bucket")
        print("  skipped because --no-network was supplied")
        print()
    else:
        headers = _headers()
        try:
            _, rate_before = _rate_limit(headers)
            print("GitHub rate bucket before latest probe")
            print(f"  {_format_rate(rate_before)}")
            print()

            if use_latest_cache and latest_cache and latest_cache.state == "fresh":
                latest = LatestResolution(
                    tag=str(latest_cache.resolved_tag),
                    api_calls=0,
                    endpoint_kind="on-disk latest cache",
                    snapshot=rate_before,
                    source="on-disk latest cache",
                )
            elif use_latest_cache:
                try:
                    latest = _resolve_latest(headers, args.repo, args.tag_prefix, args.max_pages)
                except (HTTPError, RuntimeError, URLError) as exc:
                    can_use_stale = (
                        latest_cache is not None
                        and latest_cache.resolved_tag is not None
                        and latest_cache.state == "expired"
                    )
                    if can_use_stale:
                        print(
                            "WARNING: using stale cached latest tag because GitHub "
                            f"latest resolution failed: {exc}",
                            file=sys.stderr,
                        )
                        latest = LatestResolution(
                            tag=latest_cache.resolved_tag,
                            api_calls=0,
                            endpoint_kind="stale on-disk latest cache",
                            snapshot=rate_before,
                            source="stale on-disk latest cache",
                        )
                    else:
                        raise
                if latest.source == "github" and ttl_days > 0:
                    _write_latest_cache(
                        args.cache_root,
                        args.repo,
                        args.tag_prefix,
                        latest.tag,
                        ttl_days,
                    )
            if latest is not None:
                resolved_tag = latest.tag
                print("Latest resolution")
                print(f"  resolved tag: {latest.tag}")
                print(f"  endpoint kind: {latest.endpoint_kind}")
                print(f"  source: {latest.source}")
                print(f"  API calls used for this resolution: {latest.api_calls}")
                print(f"  rate after latest probe: {_format_rate(latest.snapshot)}")
                print()
        except HTTPError as exc:
            print(f"ERROR: GitHub request failed: {exc}", file=sys.stderr)
            return 2
        except RuntimeError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
        except URLError as exc:
            print(
                f"ERROR: GitHub request failed before receiving a response: {exc}",
                file=sys.stderr,
            )
            return 2

    _print_behavior(args, latest)
    _print_cache(args, resolved_tag)

    if args.tag == "latest" and not os.getenv(LOCAL_DIR_ENV_VAR):
        api_calls = latest.api_calls if latest else None
        snapshot = latest.snapshot if latest else rate_before
    else:
        api_calls = 0
        snapshot = rate_before
    _print_estimate(snapshot, api_calls)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
