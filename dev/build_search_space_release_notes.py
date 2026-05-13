#!/usr/bin/env python3
"""Build auditable Markdown release notes for a search-space catalog release."""

from __future__ import annotations

import argparse
import pathlib
import sys

from compileiq.search_spaces.manifest import SearchSpaceManifestModel


def _escape_table_cell(value: str | None) -> str:
    if value is None:
        return ""
    return value.replace("|", "\\|").replace("\n", " ")


def build_notes(manifest: SearchSpaceManifestModel) -> str:
    """Render release notes for the provided search-space manifest."""
    lines = [
        f"# Search spaces {manifest.tag.removeprefix('search-spaces-')}",
        "",
        "This release publishes the search-space catalog described by `manifest.json`.",
        "",
        f"- Tag: `{manifest.tag}`",
        f"- Manifest format: `{manifest.manifest_format}`",
        f"- Generated at: `{manifest.generated_at}`",
        f"- Entries: {len(manifest.entries)}",
        "",
        "## Catalog",
        "",
        "| Compiler | Version | Variant | File | Size bytes | SHA256 | Description |",
        "|---|---|---|---|---:|---|---|",
    ]

    for entry in manifest.entries:
        lines.append(
            "| "
            f"{_escape_table_cell(entry.compiler)} | "
            f"{_escape_table_cell(entry.compiler_version)} | "
            f"{_escape_table_cell(entry.variant)} | "
            f"`{_escape_table_cell(entry.filename)}` | "
            f"{entry.size_bytes} | "
            f"`{entry.sha256}` | "
            f"{_escape_table_cell(entry.description)} |"
        )

    lines.extend(
        [
            "",
            "## Local Mirror",
            "",
            "To mirror this catalog, download `manifest.json` and each listed `.bin` file "
            "into the same directory, then point CompileIQ at that directory:",
            "",
            "```bash",
            "export CIQ_SEARCH_SPACES_DIR=/path/to/mirrored/search-spaces",
            "```",
            "",
            "CompileIQ verifies each file's size and SHA256 before use.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=pathlib.Path,
        required=True,
        help="Generated search-space release catalog manifest.json.",
    )
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        required=True,
        help="Where to write the Markdown release notes.",
    )
    args = parser.parse_args(argv)

    manifest = SearchSpaceManifestModel.model_validate_json(args.manifest.read_text())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(build_notes(manifest))
    print(
        f"Wrote {args.out} with {len(manifest.entries)} entries for tag {manifest.tag}.",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
