#!/usr/bin/env python3
"""Validate Search Space catalog release assets from a local or downloaded directory."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import sys
from collections.abc import Iterable

from pydantic import ValidationError

from compileiq.search_spaces.manifest import SearchSpaceManifestModel


HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")
FIXME_RE = re.compile(r"\bFIXME\b", re.IGNORECASE)
DEFAULT_DOCS_URL = "https://nvidia.github.io/CompileIQ/stable/compilers_overview.html"


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: pathlib.Path, errors: list[str]) -> object | None:
    try:
        return json.loads(path.read_text())
    except OSError as exc:
        errors.append(f"{path.name}: could not read JSON file: {exc}")
    except json.JSONDecodeError as exc:
        errors.append(f"{path.name}: invalid JSON: {exc}")
    return None


def _parse_checksums(path: pathlib.Path, errors: list[str]) -> dict[str, str]:
    checksums: dict[str, str] = {}
    try:
        lines = path.read_text().splitlines()
    except OSError as exc:
        errors.append(f"SHA256SUMS.txt: could not read checksum file: {exc}")
        return checksums

    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 2:
            errors.append(f"SHA256SUMS.txt:{line_number}: expected '<sha256>  <asset-name>'")
            continue
        digest, name = fields
        if not HEX_SHA256.fullmatch(digest):
            errors.append(f"SHA256SUMS.txt:{line_number}: invalid SHA256 digest")
        if "/" in name or "\\" in name:
            errors.append(f"SHA256SUMS.txt:{line_number}: asset names must be top-level filenames")
        if name in checksums:
            errors.append(f"SHA256SUMS.txt:{line_number}: duplicate asset {name}")
        checksums[name] = digest
    return checksums


def _validate_no_fixme(value: object, context: str, errors: list[str]) -> None:
    if isinstance(value, str):
        if FIXME_RE.search(value):
            errors.append(f"{context}: contains unresolved FIXME")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_no_fixme(item, f"{context}[{index}]", errors)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _validate_no_fixme(item, f"{context}.{key}", errors)


def _validate_checksum_file(
    asset_dir: pathlib.Path,
    extra_ok: set[str],
    errors: list[str],
) -> dict[str, str]:
    checksum_path = asset_dir / "SHA256SUMS.txt"
    if not checksum_path.is_file():
        errors.append("missing required release asset SHA256SUMS.txt")
        return {}

    checksums = _parse_checksums(checksum_path, errors)
    asset_files = {
        path.name
        for path in asset_dir.iterdir()
        if path.is_file() and not path.name.startswith(".")
    }
    hashable_assets = asset_files - {"SHA256SUMS.txt"}

    missing = set(checksums) - hashable_assets
    if missing:
        errors.append("SHA256SUMS.txt references missing assets: " + ", ".join(sorted(missing)))

    extra = hashable_assets - set(checksums) - extra_ok
    if extra:
        errors.append(
            "release directory contains assets not listed in SHA256SUMS.txt: "
            + ", ".join(sorted(extra))
        )

    for name, expected in sorted(checksums.items()):
        path = asset_dir / name
        if not path.is_file():
            continue
        actual = sha256_file(path)
        if actual != expected:
            errors.append(f"{name}: SHA256 mismatch, expected {expected}, got {actual}")

    return checksums


def _validate_release_body(
    asset_dir: pathlib.Path,
    required_names: set[str],
    docs_url: str,
    require_release_body: bool,
    errors: list[str],
) -> None:
    release_body_path = asset_dir / "release-body.md"
    if not release_body_path.is_file():
        if require_release_body:
            errors.append("missing required local staging file release-body.md")
        return

    try:
        body = release_body_path.read_text()
    except OSError as exc:
        errors.append(f"release-body.md: could not read release body: {exc}")
        return

    if docs_url not in body:
        errors.append(f"release-body.md: missing public docs URL {docs_url}")
    if FIXME_RE.search(body):
        errors.append("release-body.md: contains unresolved FIXME")
    for name in sorted(required_names):
        if name not in body:
            errors.append(f"release-body.md: missing asset name {name}")


def validate_release_assets(
    asset_dir: pathlib.Path,
    tag: str,
    extra_ok: Iterable[str] = (),
    docs_url: str = DEFAULT_DOCS_URL,
    require_release_body: bool = False,
) -> list[str]:
    """Return validation errors for a Search Space release asset directory."""
    errors: list[str] = []
    asset_dir = asset_dir.resolve()
    if not asset_dir.is_dir():
        return [f"asset directory does not exist: {asset_dir}"]

    checksums = _validate_checksum_file(asset_dir, set(extra_ok), errors)

    manifest_path = asset_dir / "manifest.json"
    if not manifest_path.is_file():
        errors.append("missing required release asset manifest.json")
        return errors
    if checksums and "manifest.json" not in checksums:
        errors.append("manifest.json is not listed in SHA256SUMS.txt")

    raw_manifest = _load_json(manifest_path, errors)
    _validate_no_fixme(raw_manifest, "manifest.json", errors)
    if raw_manifest is None:
        return errors

    try:
        manifest = SearchSpaceManifestModel.model_validate(raw_manifest)
    except ValidationError as exc:
        errors.append(f"manifest.json: schema validation failed: {exc}")
        return errors

    if manifest.tag != tag:
        errors.append(f"manifest.json: tag is {manifest.tag!r}, expected {tag!r}")

    entry_filenames = {entry.filename for entry in manifest.entries}
    bin_files = {path.name for path in asset_dir.glob("*.bin") if path.is_file()}
    missing_bins = entry_filenames - bin_files
    extra_bins = bin_files - entry_filenames
    if missing_bins:
        errors.append(
            "manifest.json references missing .bin assets: " + ", ".join(sorted(missing_bins))
        )
    if extra_bins:
        errors.append(
            "release directory contains .bin assets not in manifest.json: "
            + ", ".join(sorted(extra_bins))
        )

    for entry in manifest.entries:
        asset_path = asset_dir / entry.filename
        if not asset_path.is_file():
            continue
        if checksums and entry.filename not in checksums:
            errors.append(f"{entry.filename} is not listed in SHA256SUMS.txt")
        actual_size = asset_path.stat().st_size
        if actual_size != entry.size_bytes:
            errors.append(
                f"{entry.filename}: size_bytes mismatch, expected {entry.size_bytes}, "
                f"got {actual_size}"
            )
        actual_sha = sha256_file(asset_path)
        if actual_sha != entry.sha256:
            errors.append(
                f"{entry.filename}: sha256 mismatch, expected {entry.sha256}, got {actual_sha}"
            )

    required_names = {"manifest.json", "SHA256SUMS.txt", *entry_filenames}
    _validate_release_body(asset_dir, required_names, docs_url, require_release_body, errors)
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "asset_dir",
        type=pathlib.Path,
        help="Directory containing downloaded or staged Search Space release assets.",
    )
    parser.add_argument("--tag", required=True, help="Expected search-spaces-* release tag.")
    parser.add_argument(
        "--extra-ok",
        action="append",
        default=[],
        help="Local staging file allowed to be absent from SHA256SUMS.txt.",
    )
    parser.add_argument(
        "--docs-url",
        default=DEFAULT_DOCS_URL,
        help=(
            "Expected public docs URL when release-body.md is present. "
            f"Default: {DEFAULT_DOCS_URL}"
        ),
    )
    parser.add_argument(
        "--require-release-body",
        action="store_true",
        help="Require release-body.md and validate its public docs URL and asset names.",
    )
    args = parser.parse_args(argv)

    errors = validate_release_assets(
        args.asset_dir,
        args.tag,
        args.extra_ok,
        args.docs_url,
        args.require_release_body,
    )
    if errors:
        print(f"FAIL: Search Space release validation failed for {args.tag}.", file=sys.stderr)
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1

    print(f"PASS: Validated Search Space release assets for {args.tag} in {args.asset_dir}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
