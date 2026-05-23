#!/usr/bin/env python3
"""Seed local Search Space release inputs from a prior release or bootstrap source."""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import subprocess
import sys

import yaml


PRIOR_RELEASE_MANIFEST_NAME = ".search-space-manifest.prior-release.json"
SOURCE_NAME = "manifest-source.yaml"
REPO = "NVIDIA/CompileIQ"


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2) + "\n").encode("utf-8")


def _input_dir_is_empty(path: pathlib.Path) -> bool:
    return not path.exists() or next(path.iterdir(), None) is None


def _write_source_from_manifest(manifest_path: pathlib.Path, source_path: pathlib.Path) -> None:
    manifest = json.loads(manifest_path.read_text())
    entries = []
    for entry in manifest.get("entries", []):
        source_entry = {
            "compiler": entry["compiler"],
            "compiler_version": entry["compiler_version"],
            "variant": entry.get("variant", "default"),
            "filename": entry["filename"],
        }
        if entry.get("search_space_format") and entry.get("search_space_format") != "1.0.0":
            source_entry["search_space_format"] = entry["search_space_format"]
        if entry.get("description"):
            source_entry["description"] = entry["description"]
        entries.append(source_entry)
    source_path.write_text(yaml.safe_dump({"entries": entries}, sort_keys=False))


def setup_release(
    input_dir: pathlib.Path,
    env_file: pathlib.Path,
    release_tag: str,
    manifest_source: pathlib.Path,
    prior_tag: str | None,
    output_dir: pathlib.Path,
) -> None:
    if not _input_dir_is_empty(input_dir):
        raise ValueError(f"input directory is not empty: {input_dir}")

    input_dir.mkdir(parents=True, exist_ok=True)
    env_file.parent.mkdir(parents=True, exist_ok=True)

    prior_manifest = input_dir / PRIOR_RELEASE_MANIFEST_NAME
    staged_source = input_dir / SOURCE_NAME
    if prior_tag:
        subprocess.run(
            [
                "gh",
                "release",
                "download",
                prior_tag,
                "--repo",
                REPO,
                "--dir",
                str(input_dir),
                "--pattern",
                "manifest.json",
                "--pattern",
                "*.bin",
            ],
            check=True,
        )
        downloaded_manifest = input_dir / "manifest.json"
        if not downloaded_manifest.is_file():
            raise ValueError(f"{prior_tag}: downloaded release did not include manifest.json")
        shutil.copyfile(downloaded_manifest, prior_manifest)
        _write_source_from_manifest(downloaded_manifest, staged_source)
        downloaded_manifest.unlink()
        source_message = f"Seeded {input_dir} from {prior_tag}."
    else:
        if not manifest_source.is_file():
            raise ValueError(f"manifest source does not exist: {manifest_source}")
        shutil.copyfile(manifest_source, staged_source)
        prior_manifest.write_bytes(
            _json_bytes(
                {
                    "manifest_format": "1.0.0",
                    "tag": "",
                    "generated_at": "",
                    "entries": [],
                }
            )
        )
        source_message = (
            f"Seeded {input_dir} for the first Search Space release. "
            "No prior search-spaces-* release was found."
        )

    env_file.write_text(
        "\n".join(
            [
                f'export SS_RELEASE_TAG="{release_tag}"',
                f'export SS_RELEASE_ROOT="{input_dir.parent}"',
                f'export SS_INPUT_DIR="{input_dir}"',
                f'export SS_OUTPUT_DIR="{output_dir}"',
                "",
            ]
        )
    )
    print(source_message)
    print(f"Saved prior-release comparison manifest in {prior_manifest}.")
    print(f"Wrote {env_file}. Run: source {env_file}")
    print(f"Review release-inputs as the full catalog contents before building {release_tag}.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=pathlib.Path, required=True)
    parser.add_argument("--env-file", type=pathlib.Path, required=True)
    parser.add_argument("--release-tag", required=True)
    parser.add_argument("--manifest-source", type=pathlib.Path, required=True)
    parser.add_argument("--prior-tag")
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    args = parser.parse_args(argv)

    try:
        setup_release(
            args.input_dir,
            args.env_file,
            args.release_tag,
            args.manifest_source,
            args.prior_tag,
            args.output_dir,
        )
    except (OSError, ValueError, subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
