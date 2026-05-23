#!/usr/bin/env python3
"""Reconcile a Booster Pack input catalog with the zip files staged for release."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import pathlib
import re
import sys
import zipfile
from collections.abc import Mapping


CATALOG_NAME = "booster-pack-catalog.json"
PRIOR_RELEASE_CATALOG_NAME = ".booster-pack-catalog.prior-release.json"
RECOVERY_CATALOG_NAME = "booster-pack-catalog.backup.json"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
TAG_RE = re.compile(r"^booster-packs-(\d{4})\.(\d{2})\.(\d{2})(?:[-.][A-Za-z0-9._-]+)?$")
GENERATED_PACK_FIELDS = {
    "artifact_sha256",
    "artifact_size_bytes",
    "manifest_sha256",
    "acf_count",
    "acfs",
}
MANIFEST_COPIED_FIELDS = (
    "display_name",
    "pack_id",
    "pack_type",
    "description",
    "cuda_version",
    "controls_stage",
    "supported_gpus",
)


@dataclasses.dataclass(frozen=True)
class CatalogUpdateSummary:
    kept: list[str]
    added: list[str]
    removed: list[str]
    replaced: list[str]
    unknown: dict[str, str]
    fixme_count: int
    release_tag: str
    catalog_version: str
    generated_at: str


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and SHA256_RE.fullmatch(value) is not None


def _date_parts_from_tag(tag: str) -> tuple[str, str]:
    match = TAG_RE.fullmatch(tag)
    if not match:
        raise ValueError("tag must match booster-packs-YYYY.MM.DD[-suffix]")
    year, month, day = match.groups()
    return f"{year}.{month}.{day}", f"{year}-{month}-{day}T00:00:00Z"


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2) + "\n").encode("utf-8")


def _mapping(value: object, context: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{context}: expected object")
    return value


def _list(value: object, context: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{context}: expected list")
    return value


def _find_manifest_path(zip_path: pathlib.Path) -> str:
    with zipfile.ZipFile(zip_path) as archive:
        manifest_paths = [
            name
            for name in archive.namelist()
            if pathlib.PurePosixPath(name).name == "booster-pack-manifest.json"
        ]

    if not manifest_paths:
        raise ValueError(f"{zip_path.name}: missing booster-pack-manifest.json")
    if len(manifest_paths) > 1:
        raise ValueError(
            f"{zip_path.name}: expected one booster-pack-manifest.json, found "
            + ", ".join(sorted(manifest_paths))
        )
    return manifest_paths[0]


def _load_manifest(zip_path: pathlib.Path) -> tuple[str, dict[str, object]]:
    manifest_path = _find_manifest_path(zip_path)
    with zipfile.ZipFile(zip_path) as archive:
        manifest = json.loads(archive.read(manifest_path))
    return manifest_path, _mapping(manifest, f"{zip_path.name}: manifest")


def _has_value(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, list | dict):
        return bool(value)
    return True


def _fixme(field: str, artifact_name: str) -> str:
    return f"FIXME: review {field} for {artifact_name}"


def _count_fixmes(value: object) -> int:
    if isinstance(value, str):
        return value.count("FIXME")
    if isinstance(value, list):
        return sum(_count_fixmes(item) for item in value)
    if isinstance(value, dict):
        return sum(_count_fixmes(item) for item in value.values())
    return 0


def _field_value(
    field: str,
    artifact_name: str,
    manifest: Mapping[str, object],
    existing: Mapping[str, object],
) -> object:
    manifest_value = manifest.get(field)
    if _has_value(manifest_value):
        return manifest_value

    existing_value = existing.get(field)
    if _has_value(existing_value):
        return existing_value

    if field == "supported_gpus":
        return [_fixme(field, artifact_name)]
    return _fixme(field, artifact_name)


def _validation_summary(
    artifact_name: str,
    manifest: Mapping[str, object],
    existing: Mapping[str, object],
) -> object:
    manifest_value = manifest.get("validation_summary")
    if _has_value(manifest_value):
        return manifest_value

    existing_value = existing.get("validation_summary")
    if _has_value(existing_value):
        return existing_value

    return {"evidence": _fixme("validation evidence", artifact_name)}


def _existing_pack_entries(catalog: Mapping[str, object]) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for index, pack in enumerate(_list(catalog.get("packs"), f"{CATALOG_NAME}: packs")):
        pack_obj = _mapping(pack, f"{CATALOG_NAME}: packs[{index}]")
        artifact_name = pack_obj.get("artifact_name")
        if not isinstance(artifact_name, str) or not artifact_name:
            raise ValueError(f"{CATALOG_NAME}: packs[{index}] missing artifact_name")
        if artifact_name in result:
            raise ValueError(f"{CATALOG_NAME}: duplicate artifact_name {artifact_name}")
        result[artifact_name] = pack_obj
    return result


def _load_prior_release_catalog(
    input_dir: pathlib.Path,
) -> dict[str, object]:
    prior_release_path = input_dir / PRIOR_RELEASE_CATALOG_NAME
    if prior_release_path.is_file():
        return _mapping(json.loads(prior_release_path.read_text()), PRIOR_RELEASE_CATALOG_NAME)

    backup_path = input_dir / RECOVERY_CATALOG_NAME
    if backup_path.is_file():
        prior_release = _mapping(json.loads(backup_path.read_text()), RECOVERY_CATALOG_NAME)
        prior_release_path.write_bytes(_json_bytes(prior_release))
        return prior_release

    raise ValueError(
        f"{input_dir}: missing {PRIOR_RELEASE_CATALOG_NAME}. "
        "Run make setup-booster-pack-release before updating the catalog."
    )


def _catalog_entry(
    zip_path: pathlib.Path,
    existing: Mapping[str, object],
) -> dict[str, object]:
    artifact_name = zip_path.name
    manifest_path, manifest = _load_manifest(zip_path)

    entry = {
        key: value
        for key, value in existing.items()
        if key not in GENERATED_PACK_FIELDS and key != "artifact_name"
    }
    entry["artifact_name"] = artifact_name
    entry["manifest_path"] = manifest_path

    for field in MANIFEST_COPIED_FIELDS:
        entry[field] = _field_value(field, artifact_name, manifest, existing)
    entry["validation_summary"] = _validation_summary(artifact_name, manifest, existing)

    return entry


def update_catalog(input_dir: pathlib.Path, tag: str) -> CatalogUpdateSummary:
    """Update booster-pack-catalog.json from staged zip files.

    Returns the artifact names kept, added, removed, replaced, and unknown.
    """
    input_dir = input_dir.resolve()
    if not input_dir.is_dir():
        raise ValueError(f"input directory does not exist: {input_dir}")

    catalog_version, generated_at = _date_parts_from_tag(tag)
    catalog_path = input_dir / CATALOG_NAME
    catalog = _mapping(json.loads(catalog_path.read_text()), CATALOG_NAME)
    current_entries = _existing_pack_entries(catalog)
    prior_release_entries = _existing_pack_entries(_load_prior_release_catalog(input_dir))

    zip_paths = sorted(input_dir.glob("booster-pack-*.zip"))
    if not zip_paths:
        raise ValueError(f"{input_dir}: no booster-pack-*.zip files found")

    updated_packs = []
    kept: list[str] = []
    added: list[str] = []
    replaced: list[str] = []
    unknown: dict[str, str] = {}
    for zip_path in zip_paths:
        current_entry = current_entries.get(zip_path.name)
        prior_release_entry = prior_release_entries.get(zip_path.name)
        if prior_release_entry is None:
            added.append(zip_path.name)
        else:
            existing_sha = prior_release_entry.get("artifact_sha256")
            if not _has_value(existing_sha):
                unknown[zip_path.name] = "no prior artifact_sha256"
            elif not _is_sha256(existing_sha):
                unknown[zip_path.name] = "invalid prior artifact_sha256"
            elif _sha256_file(zip_path) != existing_sha:
                replaced.append(zip_path.name)
            else:
                kept.append(zip_path.name)
        updated_packs.append(_catalog_entry(zip_path, current_entry or prior_release_entry or {}))

    removed = sorted(set(prior_release_entries) - {path.name for path in zip_paths})
    catalog["catalog_version"] = catalog_version
    catalog["generated_at"] = generated_at
    catalog["release_tag"] = tag
    catalog["packs"] = updated_packs
    catalog_path.write_bytes(_json_bytes(catalog))
    fixme_count = _count_fixmes(catalog)
    return CatalogUpdateSummary(
        kept=kept,
        added=added,
        removed=removed,
        replaced=replaced,
        unknown=unknown,
        fixme_count=fixme_count,
        release_tag=tag,
        catalog_version=catalog_version,
        generated_at=generated_at,
    )


def _format_names(names: list[str]) -> list[str]:
    if not names:
        return ["  - none"]
    return [f"  - {name}" for name in names]


def _format_unknown(unknown: Mapping[str, str]) -> list[str]:
    if not unknown:
        return ["  - none"]
    return [f"  - {name}: {reason}" for name, reason in unknown.items()]


def _print_summary(catalog_path: pathlib.Path, summary: CatalogUpdateSummary) -> None:
    print(f"Updated {catalog_path}:")
    print(f"release_tag: {summary.release_tag}")
    print(f"catalog_version: {summary.catalog_version}")
    print(f"generated_at: {summary.generated_at}")
    for label, names in (
        ("kept", summary.kept),
        ("added", summary.added),
        ("removed", summary.removed),
        ("replaced", summary.replaced),
    ):
        print(f"{label} ({len(names)}):")
        for line in _format_names(names):
            print(line)
    print(f"unknown ({len(summary.unknown)}):")
    for line in _format_unknown(summary.unknown):
        print(line)
    print(f"FIXME markers: {summary.fixme_count}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_dir",
        type=pathlib.Path,
        help="Directory containing booster-pack-catalog.json and booster-pack-*.zip files.",
    )
    parser.add_argument(
        "--tag",
        required=True,
        help=(
            "New Booster Pack release tag, e.g. booster-packs-YYYY.MM.DD "
            "or booster-packs-YYYY.MM.DD-rev1."
        ),
    )
    args = parser.parse_args(argv)

    try:
        summary = update_catalog(args.input_dir, args.tag)
    except (OSError, ValueError, json.JSONDecodeError, zipfile.BadZipFile) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    _print_summary(args.input_dir / CATALOG_NAME, summary)
    if summary.unknown:
        print("WARNING: resolve unknown pack comparisons before building release assets.")
    if summary.fixme_count:
        print("Resolve every FIXME marker before building the release assets.")
    else:
        print("No FIXME markers found in the updated catalog.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
