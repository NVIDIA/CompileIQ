#!/usr/bin/env python3
"""Validate Booster Pack release assets from a local or downloaded directory."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import sys
import zipfile
from collections.abc import Iterable


HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")
FIXME_RE = re.compile(r"\bFIXME\b", re.IGNORECASE)
DEFAULT_DOCS_URL = "https://nvidia.github.io/CompileIQ/stable/booster_packs.html"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


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


def _mapping(value: object, context: str, errors: list[str]) -> dict[str, object] | None:
    if not isinstance(value, dict):
        errors.append(f"{context}: expected object")
        return None
    return value


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


def _list(value: object, context: str, errors: list[str]) -> list[object] | None:
    if not isinstance(value, list):
        errors.append(f"{context}: expected list")
        return None
    return value


def _entry_map(
    entries: object,
    context: str,
    errors: list[str],
) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    entry_list = _list(entries, context, errors)
    if entry_list is None:
        return result

    for index, entry in enumerate(entry_list):
        entry_context = f"{context}[{index}]"
        entry_obj = _mapping(entry, entry_context, errors)
        if entry_obj is None:
            continue

        filename = entry_obj.get("filename")
        if not isinstance(filename, str) or not filename:
            errors.append(f"{entry_context}: missing filename")
            continue
        if filename in result:
            errors.append(f"{entry_context}: duplicate filename {filename}")
        result[filename] = entry_obj

        digest = entry_obj.get("sha256")
        if not isinstance(digest, str) or not HEX_SHA256.fullmatch(digest):
            errors.append(f"{entry_context}: invalid sha256 for {filename}")

        size = entry_obj.get("size_bytes")
        if not isinstance(size, int) or size <= 0:
            errors.append(f"{entry_context}: invalid size_bytes for {filename}")

    return result


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


def _validate_pack(
    asset_dir: pathlib.Path,
    tag: str,
    catalog_version: object,
    pack: dict[str, object],
    checksums: dict[str, str],
    errors: list[str],
) -> None:
    pack_id = pack.get("pack_id")
    artifact_name = pack.get("artifact_name")
    if not isinstance(pack_id, str) or not pack_id:
        errors.append("catalog pack entry missing pack_id")
        pack_id = "<unknown-pack>"
    if not isinstance(artifact_name, str) or not artifact_name:
        errors.append(f"{pack_id}: missing artifact_name")
        return

    artifact_path = asset_dir / artifact_name
    if not artifact_path.is_file():
        errors.append(f"{pack_id}: missing artifact asset {artifact_name}")
        return
    if checksums and artifact_name not in checksums:
        errors.append(f"{pack_id}: {artifact_name} is not listed in SHA256SUMS.txt")

    expected_artifact_sha = pack.get("artifact_sha256")
    actual_artifact_sha = sha256_file(artifact_path)
    if expected_artifact_sha != actual_artifact_sha:
        errors.append(
            f"{pack_id}: catalog artifact_sha256 mismatch for {artifact_name}, "
            f"expected {expected_artifact_sha}, got {actual_artifact_sha}"
        )

    expected_artifact_size = pack.get("artifact_size_bytes")
    actual_artifact_size = artifact_path.stat().st_size
    if expected_artifact_size != actual_artifact_size:
        errors.append(
            f"{pack_id}: catalog artifact_size_bytes mismatch for {artifact_name}, "
            f"expected {expected_artifact_size}, got {actual_artifact_size}"
        )

    manifest_path = pack.get("manifest_path")
    if not isinstance(manifest_path, str) or not manifest_path:
        errors.append(f"{pack_id}: missing manifest_path")
        return

    try:
        with zipfile.ZipFile(artifact_path) as archive:
            names = set(archive.namelist())
            if manifest_path not in names:
                errors.append(f"{pack_id}: zip missing manifest {manifest_path}")
                return

            manifest_bytes = archive.read(manifest_path)
            expected_manifest_sha = pack.get("manifest_sha256")
            actual_manifest_sha = sha256_bytes(manifest_bytes)
            if expected_manifest_sha != actual_manifest_sha:
                errors.append(
                    f"{pack_id}: manifest_sha256 mismatch, expected "
                    f"{expected_manifest_sha}, got {actual_manifest_sha}"
                )

            try:
                manifest = json.loads(manifest_bytes)
            except json.JSONDecodeError as exc:
                errors.append(f"{pack_id}: invalid zip manifest JSON: {exc}")
                return

            manifest_obj = _mapping(manifest, f"{pack_id}: manifest", errors)
            if manifest_obj is None:
                return

            if manifest_obj.get("release_tag") != tag:
                errors.append(
                    f"{pack_id}: manifest release_tag is {manifest_obj.get('release_tag')!r}, "
                    f"expected {tag!r}"
                )
            if manifest_obj.get("catalog_version") != catalog_version:
                errors.append(
                    f"{pack_id}: manifest catalog_version is "
                    f"{manifest_obj.get('catalog_version')!r}, expected {catalog_version!r}"
                )
            if manifest_obj.get("pack_id") != pack_id:
                errors.append(f"{pack_id}: manifest pack_id is {manifest_obj.get('pack_id')!r}")

            catalog_acfs = _entry_map(pack.get("acfs"), f"{pack_id}: catalog acfs", errors)
            manifest_acfs = _entry_map(
                manifest_obj.get("acfs"),
                f"{pack_id}: manifest acfs",
                errors,
            )
            if catalog_acfs and manifest_acfs and set(catalog_acfs) != set(manifest_acfs):
                errors.append(
                    f"{pack_id}: catalog and manifest ACF filenames differ: "
                    f"catalog={sorted(catalog_acfs)}, manifest={sorted(manifest_acfs)}"
                )

            acf_count = pack.get("acf_count")
            if isinstance(acf_count, int) and manifest_acfs and acf_count != len(manifest_acfs):
                errors.append(
                    f"{pack_id}: acf_count is {acf_count}, manifest lists {len(manifest_acfs)}"
                )

            member_prefix = pathlib.PurePosixPath(manifest_path).parent.as_posix()
            zip_acfs = {
                pathlib.PurePosixPath(name).name
                for name in names
                if name.startswith(f"{member_prefix}/") and name.endswith(".acf")
            }
            if manifest_acfs and zip_acfs != set(manifest_acfs):
                errors.append(
                    f"{pack_id}: zip ACF files differ from manifest: "
                    f"zip={sorted(zip_acfs)}, manifest={sorted(manifest_acfs)}"
                )

            for filename, acf_entry in sorted(manifest_acfs.items()):
                member_name = f"{member_prefix}/{filename}"
                if member_name not in names:
                    continue
                payload = archive.read(member_name)
                if acf_entry.get("sha256") != sha256_bytes(payload):
                    errors.append(f"{pack_id}: SHA256 mismatch for {member_name}")
                if acf_entry.get("size_bytes") != len(payload):
                    errors.append(f"{pack_id}: size_bytes mismatch for {member_name}")
    except zipfile.BadZipFile as exc:
        errors.append(f"{pack_id}: invalid zip asset {artifact_name}: {exc}")


def _validate_release_body(
    asset_dir: pathlib.Path,
    catalog: dict[str, object],
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

    required_names = {"booster-pack-catalog.json", "SHA256SUMS.txt"}
    packs = _list(catalog.get("packs"), "booster-pack-catalog.json: packs", errors)
    if packs is None:
        return
    for index, pack in enumerate(packs):
        pack_obj = _mapping(pack, f"booster-pack-catalog.json: packs[{index}]", errors)
        if pack_obj is None:
            continue
        artifact_name = pack_obj.get("artifact_name")
        if isinstance(artifact_name, str) and artifact_name:
            required_names.add(artifact_name)

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
    """Return validation errors for a Booster Pack release asset directory."""
    errors: list[str] = []
    asset_dir = asset_dir.resolve()
    if not asset_dir.is_dir():
        return [f"asset directory does not exist: {asset_dir}"]

    checksums = _validate_checksum_file(asset_dir, set(extra_ok), errors)

    catalog_path = asset_dir / "booster-pack-catalog.json"
    if not catalog_path.is_file():
        errors.append("missing required release asset booster-pack-catalog.json")
        return errors
    if checksums and "booster-pack-catalog.json" not in checksums:
        errors.append("booster-pack-catalog.json is not listed in SHA256SUMS.txt")

    catalog = _load_json(catalog_path, errors)
    catalog_obj = _mapping(catalog, "booster-pack-catalog.json", errors)
    if catalog_obj is None:
        return errors
    _validate_no_fixme(catalog_obj, "booster-pack-catalog.json", errors)

    if catalog_obj.get("release_tag") != tag:
        errors.append(
            "booster-pack-catalog.json: release_tag is "
            f"{catalog_obj.get('release_tag')!r}, expected {tag!r}"
        )
    catalog_version = catalog_obj.get("catalog_version")

    packs = _list(catalog_obj.get("packs"), "booster-pack-catalog.json: packs", errors)
    if packs is None:
        return errors

    seen_pack_ids: set[str] = set()
    seen_artifacts: set[str] = set()
    for index, pack in enumerate(packs):
        pack_obj = _mapping(pack, f"booster-pack-catalog.json: packs[{index}]", errors)
        if pack_obj is None:
            continue

        pack_id = pack_obj.get("pack_id")
        artifact_name = pack_obj.get("artifact_name")
        if isinstance(pack_id, str):
            if pack_id in seen_pack_ids:
                errors.append(f"duplicate pack_id {pack_id}")
            seen_pack_ids.add(pack_id)
        if isinstance(artifact_name, str):
            if artifact_name in seen_artifacts:
                errors.append(f"duplicate artifact_name {artifact_name}")
            seen_artifacts.add(artifact_name)

        _validate_pack(asset_dir, tag, catalog_version, pack_obj, checksums, errors)

    _validate_release_body(asset_dir, catalog_obj, docs_url, require_release_body, errors)

    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "asset_dir",
        type=pathlib.Path,
        help="Directory containing downloaded or staged Booster Pack release assets.",
    )
    parser.add_argument("--tag", required=True, help="Expected booster-packs-* release tag.")
    parser.add_argument(
        "--extra-ok",
        action="append",
        default=[],
        help=(
            "Local staging file allowed to be absent from SHA256SUMS.txt, "
            "such as release-body.md."
        ),
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
        print(f"FAIL: Booster Pack release validation failed for {args.tag}.", file=sys.stderr)
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1

    print(f"PASS: Validated Booster Pack release assets for {args.tag} in {args.asset_dir}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
