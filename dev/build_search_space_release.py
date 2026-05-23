#!/usr/bin/env python3
"""Build auditable Search Space catalog release assets from local inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import shutil
import sys
from collections.abc import Mapping, Sequence

import yaml

from compileiq.search_spaces.manifest import SearchSpaceEntry, SearchSpaceManifestModel


DEFAULT_DOCS_URL = "https://nvidia.github.io/CompileIQ/stable/compilers_overview.html"
PRIOR_RELEASE_MANIFEST_NAME = ".search-space-manifest.prior-release.json"
SOURCE_NAME = "manifest-source.yaml"
TAG_RE = re.compile(r"^search-spaces-(\d{4})\.(\d{2})\.(\d{2})(?:[-.][A-Za-z0-9._-]+)?$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _date_parts_from_tag(tag: str) -> tuple[str, str]:
    match = TAG_RE.fullmatch(tag)
    if not match:
        raise ValueError("tag must match search-spaces-YYYY.MM.DD[-suffix]")
    year, month, day = match.groups()
    return f"{year}.{month}.{day}", f"{year}-{month}-{day}T00:00:00Z"


def _mapping(value: object, context: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{context}: expected object")
    return value


def _list(value: object, context: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{context}: expected list")
    return value


def _validate_no_fixme(value: object, context: str) -> None:
    if isinstance(value, str):
        if "FIXME" in value:
            raise ValueError(f"{context}: contains unresolved FIXME")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_no_fixme(item, f"{context}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _validate_no_fixme(item, f"{context}.{key}")


def _load_source(source_dir: pathlib.Path) -> list[dict[str, object]]:
    source_path = source_dir / SOURCE_NAME
    raw = yaml.safe_load(source_path.read_text())
    _validate_no_fixme(raw, SOURCE_NAME)
    source = _mapping(raw, SOURCE_NAME)
    entries = []
    for index, entry in enumerate(_list(source.get("entries"), f"{SOURCE_NAME}: entries")):
        entries.append(_mapping(entry, f"{SOURCE_NAME}: entries[{index}]"))
    return entries


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2) + "\n").encode("utf-8")


def _load_prior_manifest(source_dir: pathlib.Path) -> dict[str, object] | None:
    prior_path = source_dir / PRIOR_RELEASE_MANIFEST_NAME
    if not prior_path.is_file():
        return None
    return _mapping(json.loads(prior_path.read_text()), PRIOR_RELEASE_MANIFEST_NAME)


def _entries_by_filename(manifest: Mapping[str, object]) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for index, entry in enumerate(_list(manifest.get("entries"), "manifest entries")):
        entry_obj = _mapping(entry, f"manifest entries[{index}]")
        filename = entry_obj.get("filename")
        if not isinstance(filename, str) or not filename:
            raise ValueError(f"manifest entries[{index}]: missing filename")
        if filename in result:
            raise ValueError(f"manifest entries: duplicate filename {filename}")
        result[filename] = entry_obj
    return result


def _entry_label(entry: Mapping[str, object]) -> str:
    compiler = str(entry.get("compiler", "unknown"))
    version = str(entry.get("compiler_version", "unknown"))
    variant = str(entry.get("variant", "default"))
    filename = str(entry.get("filename", "unknown.bin"))
    return f"{compiler} {version}/{variant} (`{filename}`)"


def _release_changes(
    source_dir: pathlib.Path,
    manifest: SearchSpaceManifestModel,
    prior_manifest: Mapping[str, object] | None,
) -> dict[str, list[str]]:
    current = {entry.filename: entry.model_dump() for entry in manifest.entries}
    if prior_manifest is None:
        return {
            "added": [_entry_label(current[name]) for name in sorted(current)],
            "updated": [],
            "removed": [],
        }

    prior = _entries_by_filename(prior_manifest)
    added = [_entry_label(current[name]) for name in sorted(set(current) - set(prior))]
    removed = [_entry_label(prior[name]) for name in sorted(set(prior) - set(current))]
    updated: list[str] = []
    for filename in sorted(set(current) & set(prior)):
        prior_sha = prior[filename].get("sha256")
        if (
            isinstance(prior_sha, str)
            and SHA256_RE.fullmatch(prior_sha)
            and sha256_file(source_dir / filename) != prior_sha
        ):
            updated.append(_entry_label(current[filename]))
    return {"added": added, "updated": updated, "removed": removed}


def _render_change_section(title: str, labels: Sequence[str]) -> list[str]:
    if not labels:
        return []
    lines = [f"### {title}", ""]
    lines.extend(f"- {label}" for label in labels)
    lines.append("")
    return lines


def _render_catalog_table(entries: Sequence[SearchSpaceEntry]) -> list[str]:
    lines = [
        "| Compiler | Version | Variant | File | Size bytes | SHA256 | Description |",
        "|---|---|---|---|---:|---|---|",
    ]
    for entry in entries:
        description = (entry.description or "").replace("|", "\\|").replace("\n", " ")
        lines.append(
            "| "
            f"{entry.compiler} | "
            f"{entry.compiler_version} | "
            f"{entry.variant} | "
            f"`{entry.filename}` | "
            f"{entry.size_bytes} | "
            f"`{entry.sha256}` | "
            f"{description} |"
        )
    return lines


def _render_release_body(
    manifest: SearchSpaceManifestModel,
    docs_url: str,
    catalog_date: str,
    changes: Mapping[str, Sequence[str]],
) -> str:
    lines = [
        "## Search Space Catalog Release",
        "",
        f"Documentation: {docs_url}",
        "",
        f"This release contains the complete compiler search-space catalog as of {catalog_date}.",
        "",
        "Assets:",
        "",
        "- `manifest.json`",
    ]
    lines.extend(f"- `{entry.filename}`" for entry in manifest.entries)
    lines.extend(["- `SHA256SUMS.txt`", "", "## Catalog", ""])
    lines.extend(_render_catalog_table(manifest.entries))
    lines.extend([""])

    change_lines = (
        _render_change_section("Added", changes.get("added", []))
        + _render_change_section("Updated", changes.get("updated", []))
        + _render_change_section("Removed", changes.get("removed", []))
    )
    lines.extend(["## Changes", ""])
    if change_lines:
        lines.extend(change_lines)
    else:
        lines.extend(["No added, removed, or updated search spaces.", ""])

    return "\n".join(lines).rstrip() + "\n"


def build_release(
    source_dir: pathlib.Path,
    output_dir: pathlib.Path,
    tag: str,
    generated_at: str | None,
    docs_url: str,
    clean_output: bool,
) -> Sequence[pathlib.Path]:
    if not source_dir.is_dir():
        raise ValueError(f"source directory does not exist: {source_dir}")

    catalog_date, default_generated_at = _date_parts_from_tag(tag)
    generated_at = generated_at or default_generated_at

    if output_dir.resolve() == source_dir.resolve():
        raise ValueError("output directory must differ from source directory")
    if output_dir.exists() and any(output_dir.iterdir()):
        if not clean_output:
            raise ValueError(f"output directory is not empty: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_entries = _load_source(source_dir)
    source_filenames = {
        str(entry.get("filename")) for entry in source_entries if entry.get("filename")
    }
    staged_bin_names = {path.name for path in source_dir.glob("*.bin")}
    missing = source_filenames - staged_bin_names
    extra = staged_bin_names - source_filenames
    if missing:
        raise ValueError("manifest source references missing assets: " + ", ".join(sorted(missing)))
    if extra:
        raise ValueError(
            "input directory contains unreferenced .bin assets: " + ", ".join(sorted(extra))
        )

    output_assets: list[pathlib.Path] = []
    entries: list[SearchSpaceEntry] = []
    for src_entry in source_entries:
        filename = str(src_entry["filename"])
        source_bin = source_dir / filename
        output_bin = output_dir / filename
        shutil.copyfile(source_bin, output_bin)
        entry = SearchSpaceEntry(
            **src_entry,
            sha256=sha256_file(output_bin),
            size_bytes=output_bin.stat().st_size,
        )
        entries.append(entry)
        output_assets.append(output_bin)

    manifest = SearchSpaceManifestModel(
        tag=tag,
        generated_at=generated_at,
        entries=entries,
    )
    output_manifest = output_dir / "manifest.json"
    output_manifest.write_text(manifest.model_dump_json(indent=2) + "\n")

    prior_manifest = _load_prior_manifest(source_dir)
    changes = _release_changes(source_dir, manifest, prior_manifest)
    release_body = output_dir / "release-body.md"
    release_body.write_text(_render_release_body(manifest, docs_url, catalog_date, changes))

    checksum_targets = [output_manifest, *output_assets]
    checksums = [f"{sha256_file(path)}  {path.name}" for path in checksum_targets]
    checksum_path = output_dir / "SHA256SUMS.txt"
    checksum_path.write_text("\n".join(checksums) + "\n")

    return [output_manifest, *output_assets, checksum_path, release_body]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=pathlib.Path,
        required=True,
        help="Directory containing manifest-source.yaml and approved .bin files.",
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        required=True,
        help="Clean output directory for generated release assets.",
    )
    parser.add_argument(
        "--tag",
        required=True,
        help="Release tag, e.g. search-spaces-YYYY.MM.DD[-rev1].",
    )
    parser.add_argument(
        "--generated-at",
        help="Manifest generated_at timestamp. Defaults to the deterministic tag date.",
    )
    parser.add_argument(
        "--docs-url",
        default=DEFAULT_DOCS_URL,
        help=f"Public docs URL written into release-body.md. Default: {DEFAULT_DOCS_URL}",
    )
    parser.add_argument(
        "--clean-output",
        action="store_true",
        help="Delete the output directory before writing generated assets.",
    )
    args = parser.parse_args(argv)

    try:
        assets = build_release(
            args.source_dir,
            args.output_dir,
            args.tag,
            args.generated_at,
            args.docs_url,
            args.clean_output,
        )
    except (OSError, ValueError, json.JSONDecodeError, yaml.YAMLError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"Wrote Search Space release assets for {args.tag} to {args.output_dir}.")
    print("Generated files:")
    for asset in assets:
        print(f"- {asset.name}")
    print("Upload manifest.json, .bin assets, and SHA256SUMS.txt. Use release-body.md as notes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
