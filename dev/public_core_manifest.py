#!/usr/bin/env python3
"""Render the public CompileIQ core manifest from a Core producer manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from compileiq.core.verify_core import with_core_lock


PUBLIC_MANIFEST_FIELDS = (
    "schema_version",
    "core_commit",
    "core_ref",
    "core_tag",
    "built_at",
    "pipeline_id",
    "files",
)
REQUIRED_PUBLIC_MANIFEST_FIELDS = frozenset({"schema_version", "core_commit", "files"})


def _normalize_sha256(value: str) -> str:
    return value if value.startswith("sha256:") else f"sha256:{value}"


def render_public_manifest(
    source_manifest: dict[str, object],
    source_manifest_sha256: str,
) -> dict[str, object]:
    """Return the public manifest fields with a regenerated `core_lock`."""
    public = {
        key: source_manifest[key]
        for key in PUBLIC_MANIFEST_FIELDS
        if key in source_manifest
    }
    missing = sorted(REQUIRED_PUBLIC_MANIFEST_FIELDS - public.keys())
    if missing:
        raise ValueError(f"source manifest missing required keys: {', '.join(missing)}")
    if not isinstance(public["files"], dict) or not public["files"]:
        raise ValueError("source manifest must contain a non-empty files object")

    public["source_manifest_sha256"] = _normalize_sha256(source_manifest_sha256)
    return with_core_lock(public)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source-manifest-sha256", required=True)
    args = parser.parse_args()

    source = json.loads(args.source.read_text(encoding="utf-8"))
    public = render_public_manifest(source, args.source_manifest_sha256)
    args.output.write_text(json.dumps(public, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
