#!/usr/bin/env python3
"""Build a release catalog ``manifest.json`` from a manifest-source YAML file.

Reads the YAML source-of-truth, computes sha256 + byte size for each declared
binary, and emits a JSON manifest matching ``SearchSpaceManifestModel``. Run
locally with ``make build-search-space-manifest`` for a sanity check.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import pathlib
import sys

import yaml

from compileiq.search_spaces.manifest import (
    SearchSpaceEntry,
    SearchSpaceManifestModel,
    validate_asset_filename,
)

# Stream hashing in 1 MiB chunks so large binaries are never read into memory at once.
_HASH_CHUNK_BYTES = 1 << 20


def _sha256_of(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(_HASH_CHUNK_BYTES), b""):
            h.update(chunk)
    return h.hexdigest()


def build(
    source: pathlib.Path,
    artifacts_dir: pathlib.Path,
    tag: str,
) -> SearchSpaceManifestModel:
    # The source file is plain data. safe_load avoids constructing arbitrary
    # Python objects if a malformed or untrusted YAML file is passed in.
    raw = yaml.safe_load(source.read_text())
    if not isinstance(raw, dict) or "entries" not in raw:
        raise ValueError(f"{source} must be a YAML mapping with an 'entries' key")

    entries: list[SearchSpaceEntry] = []
    for src_entry in raw["entries"]:
        filename = validate_asset_filename(src_entry["filename"])
        bin_path = artifacts_dir / filename
        if not bin_path.exists():
            raise FileNotFoundError(
                f"{bin_path} declared in {source} but not present in {artifacts_dir}"
            )
        entries.append(
            SearchSpaceEntry(
                **src_entry,
                sha256=_sha256_of(bin_path),
                size_bytes=bin_path.stat().st_size,
            )
        )

    return SearchSpaceManifestModel(
        tag=tag,
        generated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        entries=entries,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=pathlib.Path,
        default=pathlib.Path("release/search-spaces/manifest-source.yaml"),
        help="YAML source-of-truth for manifest entries.",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=pathlib.Path,
        required=True,
        help="Directory containing the .bin files referenced by --source.",
    )
    parser.add_argument(
        "--tag",
        required=True,
        help="Release tag this manifest belongs to (e.g. search-spaces-2026.04.27).",
    )
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        default=pathlib.Path("manifest.json"),
        help="Where to write the resulting JSON.",
    )
    args = parser.parse_args(argv)

    manifest = build(args.source, args.artifacts_dir, args.tag)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(manifest.model_dump_json(indent=2))
    print(
        f"Wrote {args.out} with {len(manifest.entries)} entries for tag {args.tag}.",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
