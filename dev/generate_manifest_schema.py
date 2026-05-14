#!/usr/bin/env python3
"""Generate the checked-in JSON Schema for search-space release manifests."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

from compileiq.search_spaces.manifest import search_space_manifest_json_schema


DEFAULT_OUT = pathlib.Path("schemas/search-space-manifest-v1.schema.json")


def schema_text() -> str:
    return json.dumps(search_space_manifest_json_schema(), indent=2, sort_keys=True) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        default=DEFAULT_OUT,
        help="Path to write the generated schema.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail if --out does not already match the generated schema.",
    )
    args = parser.parse_args(argv)

    generated = schema_text()
    if args.check:
        actual = args.out.read_text() if args.out.exists() else ""
        if actual != generated:
            print(
                f"{args.out} is stale; run dev/generate_manifest_schema.py",
                file=sys.stderr,
            )
            return 1
        return 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(generated)
    print(f"Wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
