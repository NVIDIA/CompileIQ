#!/usr/bin/env python3
"""Generate a sanitized provenance manifest for a CompileIQ wheel artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import zipfile
from email.parser import Parser
from pathlib import Path


def sha256(path: Path) -> str:
    """Hash an artifact incrementally so large wheels are not loaded into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    """Write public-safe package, source, and wheel provenance for CI artifacts."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--wheel", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    wheel = args.wheel.resolve()
    with zipfile.ZipFile(wheel) as archive:
        metadata_name = next(
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        )
        metadata = Parser().parsestr(archive.read(metadata_name).decode())
    commit = (
        os.environ.get("CI_COMMIT_SHA")
        or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    )
    manifest = {
        "schema_version": 1,
        "package": metadata["Name"],
        "version": metadata["Version"],
        "source_commit": commit,
        "source_ref": "tag" if os.environ.get("CI_COMMIT_TAG") else "feature-branch",
        "wheel": {
            "filename": wheel.name,
            "sha256": f"sha256:{sha256(wheel)}",
            "size": wheel.stat().st_size,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
