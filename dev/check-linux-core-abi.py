#!/usr/bin/env python3
"""Check bundled Linux core ELF version requirements."""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path


DEFAULT_ELF_FILES = (
    Path("compileiq/core/executable/linux/x86_64/bin/_core"),
    Path("compileiq/core/executable/linux/x86_64/lib/libciq.so"),
    Path("compileiq/core/executable/linux/aarch64/bin/_core"),
    Path("compileiq/core/executable/linux/aarch64/lib/libciq.so"),
)
DEFAULT_MAX_GLIBC = (2, 34)
DEFAULT_MAX_GLIBCXX = (3, 4, 29)
VERSION_RE = re.compile(r"\b(GLIBCXX|GLIBC)_([0-9]+(?:\.[0-9]+)*)\b")


def parse_version(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in value.split("."))


def version_text(version: tuple[int, ...]) -> str:
    return ".".join(str(part) for part in version)


def run_readelf(path: Path) -> str:
    if shutil.which("readelf") is None:
        raise RuntimeError("readelf not found; install binutils before running this check")

    completed = subprocess.run(
        ["readelf", "--version-info", str(path)],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        details = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"readelf failed for {path}: {details}")
    return completed.stdout


def required_versions(readelf_output: str) -> dict[str, set[tuple[int, ...]]]:
    versions: dict[str, set[tuple[int, ...]]] = {"GLIBC": set(), "GLIBCXX": set()}
    for family, version in VERSION_RE.findall(readelf_output):
        versions[family].add(parse_version(version))
    return versions


def check_file(path: Path, max_glibc: tuple[int, ...], max_glibcxx: tuple[int, ...]) -> list[str]:
    if not path.is_file():
        return [f"{path}: file not found"]

    versions = required_versions(run_readelf(path))
    failures: list[str] = []
    limits = {"GLIBC": max_glibc, "GLIBCXX": max_glibcxx}

    print(path)
    for family in ("GLIBC", "GLIBCXX"):
        seen = sorted(versions[family])
        if seen:
            print(f"  {family}: max {version_text(seen[-1])}")
        else:
            print(f"  {family}: no version requirements")

        too_new = [version for version in seen if version > limits[family]]
        for version in too_new:
            failures.append(
                f"{path}: requires {family}_{version_text(version)} "
                f"> {family}_{version_text(limits[family])}"
            )

    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        default=list(DEFAULT_ELF_FILES),
        help="Linux ELF files to inspect",
    )
    parser.add_argument("--max-glibc", default="2.34", help="Maximum allowed GLIBC version")
    parser.add_argument(
        "--max-glibcxx",
        default="3.4.29",
        help="Maximum allowed GLIBCXX version",
    )
    args = parser.parse_args()

    max_glibc = parse_version(args.max_glibc)
    max_glibcxx = parse_version(args.max_glibcxx)

    failures: list[str] = []
    for path in args.paths:
        failures.extend(check_file(path, max_glibc, max_glibcxx))

    if failures:
        print("\nABI compatibility check failed:", file=sys.stderr)
        for failure in failures:
            print(f"ERROR: {failure}", file=sys.stderr)
        return 1

    print(
        "\nABI compatibility check passed "
        f"(GLIBC <= {args.max_glibc}, GLIBCXX <= {args.max_glibcxx})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
