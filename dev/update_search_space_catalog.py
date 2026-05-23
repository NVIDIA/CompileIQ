#!/usr/bin/env python3
"""Reconcile search-space release inputs with the staged .bin files."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import pathlib
import re
import sys
from collections.abc import Mapping

import yaml


SOURCE_NAME = "manifest-source.yaml"
PRIOR_RELEASE_MANIFEST_NAME = ".search-space-manifest.prior-release.json"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
TAG_RE = re.compile(r"^search-spaces-(\d{4})\.(\d{2})\.(\d{2})(?:[-.][A-Za-z0-9._-]+)?$")
STANDARD_FILENAME_RE = re.compile(
    r"^(?P<compiler>ptxas|nvcc)(?P<version>\d+(?:\.\d+)+)"
    r"(?:_(?P<variant>[A-Za-z0-9._-]+))?_search_space\.bin$"
)
SOURCE_FIELDS = (
    "compiler",
    "compiler_version",
    "variant",
    "filename",
    "search_space_format",
    "description",
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
    generated_at: str


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and SHA256_RE.fullmatch(value) is not None


def _generated_at_from_tag(tag: str) -> str:
    match = TAG_RE.fullmatch(tag)
    if not match:
        raise ValueError("tag must match search-spaces-YYYY.MM.DD[-suffix]")
    year, month, day = match.groups()
    return f"{year}-{month}-{day}T00:00:00Z"


def _mapping(value: object, context: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{context}: expected object")
    return value


def _list(value: object, context: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{context}: expected list")
    return value


def _load_source(path: pathlib.Path) -> list[dict[str, object]]:
    raw = yaml.safe_load(path.read_text())
    source = _mapping(raw, SOURCE_NAME)
    entries = []
    for index, entry in enumerate(_list(source.get("entries"), f"{SOURCE_NAME}: entries")):
        entry_obj = _mapping(entry, f"{SOURCE_NAME}: entries[{index}]")
        filename = entry_obj.get("filename")
        if not isinstance(filename, str) or not filename:
            raise ValueError(f"{SOURCE_NAME}: entries[{index}] missing filename")
        entries.append(entry_obj)
    return entries


def _load_prior_manifest(input_dir: pathlib.Path) -> dict[str, object]:
    prior_path = input_dir / PRIOR_RELEASE_MANIFEST_NAME
    if not prior_path.is_file():
        raise ValueError(
            f"{input_dir}: missing {PRIOR_RELEASE_MANIFEST_NAME}. "
            "Run make setup-search-space-release before updating the catalog."
        )
    return _mapping(json.loads(prior_path.read_text()), PRIOR_RELEASE_MANIFEST_NAME)


def _entries_by_filename(
    entries: list[dict[str, object]],
    context: str,
) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for index, entry in enumerate(entries):
        filename = entry.get("filename")
        if not isinstance(filename, str) or not filename:
            raise ValueError(f"{context}[{index}] missing filename")
        if filename in result:
            raise ValueError(f"{context}: duplicate filename {filename}")
        result[filename] = entry
    return result


def _prior_entries_by_filename(manifest: Mapping[str, object]) -> dict[str, dict[str, object]]:
    entries = []
    for index, entry in enumerate(_list(manifest.get("entries"), "prior manifest entries")):
        entries.append(_mapping(entry, f"prior manifest entries[{index}]"))
    return _entries_by_filename(entries, "prior manifest entries")


def _fixme(field: str, filename: str) -> str:
    return f"FIXME: review {field} for {filename}"


def _count_fixmes(value: object) -> int:
    if isinstance(value, str):
        return value.count("FIXME")
    if isinstance(value, list):
        return sum(_count_fixmes(item) for item in value)
    if isinstance(value, dict):
        return sum(_count_fixmes(item) for item in value.values())
    return 0


def _inferred_metadata(filename: str) -> dict[str, object]:
    match = STANDARD_FILENAME_RE.fullmatch(filename)
    if not match:
        return {}
    return {
        "compiler": match.group("compiler"),
        "compiler_version": match.group("version"),
        "variant": match.group("variant") or "default",
    }


def _has_value(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, list | dict):
        return bool(value)
    return True


def _field_value(
    field: str,
    filename: str,
    current: Mapping[str, object],
    prior: Mapping[str, object],
    inferred: Mapping[str, object],
) -> object:
    for source in (current, prior, inferred):
        value = source.get(field)
        if _has_value(value):
            return value

    if field == "variant":
        return "default"
    if field == "search_space_format":
        return "1.0.0"
    return _fixme(field, filename)


def _source_entry(
    filename: str,
    current: Mapping[str, object],
    prior: Mapping[str, object],
) -> dict[str, object]:
    inferred = _inferred_metadata(filename)
    entry = {
        "compiler": _field_value("compiler", filename, current, prior, inferred),
        "compiler_version": _field_value("compiler_version", filename, current, prior, inferred),
        "variant": _field_value("variant", filename, current, prior, inferred),
        "filename": filename,
    }
    search_space_format = _field_value("search_space_format", filename, current, prior, inferred)
    if (
        search_space_format != "1.0.0"
        or "search_space_format" in current
        or "search_space_format" in prior
    ):
        entry["search_space_format"] = search_space_format

    description = _field_value("description", filename, current, prior, inferred)
    if _has_value(description) and not str(description).startswith("FIXME:"):
        entry["description"] = description
    return entry


def _ordered_filenames(
    staged_filenames: set[str],
    current_entries: list[dict[str, object]],
) -> list[str]:
    result: list[str] = []
    for entry in current_entries:
        filename = entry.get("filename")
        if isinstance(filename, str) and filename in staged_filenames and filename not in result:
            result.append(filename)
    result.extend(sorted(staged_filenames - set(result)))
    return result


def update_catalog(input_dir: pathlib.Path, tag: str) -> CatalogUpdateSummary:
    """Update manifest-source.yaml from the staged search-space .bin files."""
    input_dir = input_dir.resolve()
    if not input_dir.is_dir():
        raise ValueError(f"input directory does not exist: {input_dir}")

    generated_at = _generated_at_from_tag(tag)
    source_path = input_dir / SOURCE_NAME
    if not source_path.is_file():
        raise ValueError(f"{input_dir}: missing {SOURCE_NAME}")

    current_entries = _load_source(source_path)
    current_by_filename = _entries_by_filename(current_entries, f"{SOURCE_NAME}: entries")
    prior_by_filename = _prior_entries_by_filename(_load_prior_manifest(input_dir))

    bin_paths = sorted(input_dir.glob("*.bin"))
    if not bin_paths:
        raise ValueError(f"{input_dir}: no *.bin files found")

    staged_filenames = {path.name for path in bin_paths}
    kept: list[str] = []
    added: list[str] = []
    replaced: list[str] = []
    unknown: dict[str, str] = {}
    updated_entries: list[dict[str, object]] = []

    for filename in _ordered_filenames(staged_filenames, current_entries):
        bin_path = input_dir / filename
        current_entry = current_by_filename.get(filename, {})
        prior_entry = prior_by_filename.get(filename, {})
        if not prior_entry:
            added.append(filename)
        else:
            prior_sha = prior_entry.get("sha256")
            if not _has_value(prior_sha):
                unknown[filename] = "no prior sha256"
            elif not _is_sha256(prior_sha):
                unknown[filename] = "invalid prior sha256"
            elif _sha256_file(bin_path) != prior_sha:
                replaced.append(filename)
            else:
                kept.append(filename)
        updated_entries.append(_source_entry(filename, current_entry, prior_entry))

    removed = sorted(set(prior_by_filename) - staged_filenames)
    source_path.write_text(yaml.safe_dump({"entries": updated_entries}, sort_keys=False))
    fixme_count = _count_fixmes(updated_entries)
    return CatalogUpdateSummary(
        kept=kept,
        added=added,
        removed=removed,
        replaced=replaced,
        unknown=unknown,
        fixme_count=fixme_count,
        release_tag=tag,
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


def _print_summary(source_path: pathlib.Path, summary: CatalogUpdateSummary) -> None:
    print(f"Updated {source_path}:")
    print(f"release_tag: {summary.release_tag}")
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
        help="Directory containing manifest-source.yaml and staged *.bin files.",
    )
    parser.add_argument(
        "--tag",
        required=True,
        help="New Search Space release tag, e.g. search-spaces-YYYY.MM.DD[-rev1].",
    )
    args = parser.parse_args(argv)

    try:
        summary = update_catalog(args.input_dir, args.tag)
    except (OSError, ValueError, json.JSONDecodeError, yaml.YAMLError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    _print_summary(args.input_dir / SOURCE_NAME, summary)
    if summary.unknown:
        print("WARNING: resolve unknown search-space comparisons before building release assets.")
    if summary.fixme_count:
        print("Resolve every FIXME marker before building the release assets.")
    else:
        print("No FIXME markers found in the updated manifest source.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
