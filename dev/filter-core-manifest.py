#!/usr/bin/env python3
"""Filter bundled core-manifest.json to one wheel platform."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from compileiq.core.verify_core import with_core_lock


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--platform", required=True, help="Platform root, e.g. linux/x86_64")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    files = manifest.get("files")
    if not isinstance(files, dict):
        print("ERROR: manifest missing files object", file=sys.stderr)
        return 2

    prefix = f"{args.platform}/"
    filtered_files = {
        path: digest
        for path, digest in sorted(files.items())
        if path.startswith(prefix)
    }
    if not filtered_files:
        print(f"ERROR: manifest has no files for {args.platform}", file=sys.stderr)
        return 1
    manifest["files"] = filtered_files

    manifest = with_core_lock(manifest)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"filtered {args.manifest} to {args.platform} ({len(filtered_files)} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
