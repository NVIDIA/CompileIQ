"""Verify packaged core binaries against the core binary manifest."""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

EXECUTABLE_DIR = Path(__file__).resolve().parent / "executable"
MANIFEST_PATH = EXECUTABLE_DIR / "core-manifest.json"
SIDECAR_FILENAMES = frozenset({"core-manifest.json", "core-version.lock"})
REQUIRED_PLATFORMS = ("linux/x86_64", "linux/aarch64", "win32/amd64")


@dataclass
class VerifyCoreResult:
    """Result of checking core binary files against `core-manifest.json`.

    Attributes:
        ok: True when all checked files and required platforms match.
        manifest_path: Path to the core binary manifest used for verification.
        executable_root: Root directory containing platform-specific core files.
        core_commit: Core source commit recorded by the manifest, when present.
        platforms: Platform directories named by the manifest.
        matches: Manifest-relative files whose SHA-256 matched.
        mismatches: Tuples of manifest-relative path, expected hash, and actual hash.
        missing: Manifest-relative files that were listed but absent on disk.
        extra: On-disk files under `executable_root` that are not in the manifest.
        missing_platforms: Required platform directories absent from the manifest.
    """

    ok: bool
    manifest_path: Path
    executable_root: Path
    core_commit: str | None = None
    platforms: list[str] = field(default_factory=list)
    matches: list[str] = field(default_factory=list)
    mismatches: list[tuple[str, str, str]] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    extra: list[str] = field(default_factory=list)
    missing_platforms: list[str] = field(default_factory=list)

    def render(self) -> str:
        """Render a concise CLI report for human review and CI logs."""
        checked = len(self.matches) + len(self.mismatches)
        lines = [
            f"manifest:    {self.manifest_path}",
            f"executable:  {self.executable_root}",
        ]
        if self.core_commit:
            lines.append(f"core_commit: {self.core_commit}")
        if self.platforms:
            lines.append(f"platforms:   {', '.join(self.platforms)}")
        lines.append(f"checked:     {checked} files")

        if self.missing_platforms:
            lines.append(f"  MISSING PLATFORMS ({len(self.missing_platforms)}):")
            lines.extend(f"    {path}" for path in self.missing_platforms)
        if self.mismatches:
            lines.append(f"  MISMATCH ({len(self.mismatches)}):")
            for path, expected, actual in self.mismatches:
                lines.append(f"    {path}")
                lines.append(f"      expected sha256:{expected}")
                lines.append(f"      actual   sha256:{actual}")
        if self.missing:
            lines.append(f"  MISSING ({len(self.missing)}, in manifest but not on disk):")
            lines.extend(f"    {path}" for path in self.missing)
        if self.extra:
            lines.append(f"  EXTRA ({len(self.extra)}, on disk but not in manifest):")
            lines.extend(f"    {path}" for path in self.extra)

        lines.append("OK" if self.ok else "FAIL")
        return "\n".join(lines)


def load_manifest(manifest_path: Path = MANIFEST_PATH) -> dict:
    """Load the core binary manifest JSON from `compileiq/core/executable`."""
    with manifest_path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Return the SHA-256 hex digest for one file."""
    digest = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hex_digest(value: str) -> str:
    """Normalize `sha256:<hex>` manifest values to plain hex."""
    return value.split(":", 1)[1] if ":" in value else value


def _manifest_files(manifest: dict) -> dict[str, str]:
    """Return the core manifest `files` map as relative paths to SHA-256 digests."""
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("core manifest must contain a non-empty 'files' object")
    return {str(path): _hex_digest(str(digest)) for path, digest in files.items()}


def _platforms(paths: Iterable[str]) -> list[str]:
    """Extract platform directory names such as `linux/x86_64` from manifest paths."""
    platforms = set()
    for path in paths:
        parts = Path(path).parts
        if len(parts) >= 2:
            platforms.add(f"{parts[0]}/{parts[1]}")
    return sorted(platforms)


def _on_disk_files(executable_root: Path, manifest_path: Path) -> dict[str, Path]:
    """Return files under the core executable tree, excluding metadata sidecars."""
    files: dict[str, Path] = {}
    root_resolved = executable_root.resolve()
    manifest_resolved = manifest_path.resolve()

    for path in executable_root.rglob("*"):
        if not path.is_file():
            continue
        if path.resolve() == manifest_resolved:
            continue
        if path.parent.resolve() == root_resolved and path.name in SIDECAR_FILENAMES:
            continue
        files[path.relative_to(executable_root).as_posix()] = path

    return files


def verify(
    executable_root: Path = EXECUTABLE_DIR,
    manifest_path: Path = MANIFEST_PATH,
    required_platforms: Iterable[str] = REQUIRED_PLATFORMS,
) -> VerifyCoreResult:
    """Verify every manifest-listed core binary and report drift."""
    manifest = load_manifest(manifest_path)
    expected = _manifest_files(manifest)
    on_disk = _on_disk_files(executable_root, manifest_path)
    platforms = _platforms(expected)
    missing_platforms = sorted(set(required_platforms) - set(platforms))

    result = VerifyCoreResult(
        ok=True,
        manifest_path=manifest_path,
        executable_root=executable_root,
        core_commit=manifest.get("core_commit"),
        platforms=platforms,
        missing_platforms=missing_platforms,
    )

    for rel_path, expected_hash in sorted(expected.items()):
        path = on_disk.pop(rel_path, None)
        if path is None:
            result.missing.append(rel_path)
            continue

        actual_hash = sha256_file(path)
        if actual_hash == expected_hash:
            result.matches.append(rel_path)
        else:
            result.mismatches.append((rel_path, expected_hash, actual_hash))

    result.extra = sorted(on_disk)
    result.ok = not (
        result.mismatches or result.missing or result.extra or result.missing_platforms
    )
    return result


def verify_binary(
    binary_path: Path,
    executable_root: Path = EXECUTABLE_DIR,
    manifest_path: Path = MANIFEST_PATH,
) -> str:
    """Verify a selected core binary against the core manifest and return its manifest path."""
    binary_path = Path(binary_path)
    executable_root = Path(executable_root)
    manifest = load_manifest(manifest_path)
    expected = _manifest_files(manifest)

    try:
        rel_path = binary_path.resolve().relative_to(executable_root.resolve()).as_posix()
    except ValueError as exc:
        raise RuntimeError(
            f"Core binary {binary_path} is not under manifest root {executable_root}."
        ) from exc

    expected_hash = expected.get(rel_path)
    if expected_hash is None:
        raise RuntimeError(f"Core binary {rel_path} is not listed in {manifest_path}.")
    if not binary_path.is_file():
        raise RuntimeError(f"Core binary {binary_path} does not exist.")

    actual_hash = sha256_file(binary_path)
    if actual_hash != expected_hash:
        raise RuntimeError(
            f"Core binary {rel_path} does not match {manifest_path}: "
            f"expected sha256:{expected_hash}, actual sha256:{actual_hash}."
        )

    return rel_path


def verify_binary_platform(
    binary_path: Path,
    executable_root: Path = EXECUTABLE_DIR,
    manifest_path: Path = MANIFEST_PATH,
) -> list[str]:
    """Verify every manifest entry for the platform containing `binary_path`."""
    executable_root = Path(executable_root)
    rel_path = verify_binary(binary_path, executable_root, manifest_path)
    parts = Path(rel_path).parts
    if len(parts) < 2:
        return [rel_path]

    platform_root = f"{parts[0]}/{parts[1]}"
    manifest = load_manifest(manifest_path)
    expected = {
        path: digest
        for path, digest in _manifest_files(manifest).items()
        if path.startswith(f"{platform_root}/")
    }
    if not expected:
        raise RuntimeError(
            f"Core platform {platform_root} is not listed in {manifest_path}."
        )

    verified = []
    for rel_platform_path, expected_hash in sorted(expected.items()):
        path = executable_root / Path(rel_platform_path)
        if not path.is_file():
            raise RuntimeError(f"Core binary dependency {rel_platform_path} does not exist.")

        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"Core binary dependency {rel_platform_path} does not match "
                f"{manifest_path}: expected sha256:{expected_hash}, "
                f"actual sha256:{actual_hash}."
            )
        verified.append(rel_platform_path)

    return verified


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for `python -m compileiq.core.verify_core`."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m compileiq.core.verify_core",
        description="Verify bundled core binaries against core-manifest.json.",
    )
    parser.add_argument(
        "--executable-root",
        type=Path,
        default=EXECUTABLE_DIR,
        help="Directory holding platform core binaries.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=MANIFEST_PATH,
        help="Core manifest path.",
    )
    parser.add_argument(
        "--allow-missing-platforms",
        action="store_true",
        help="Do not require all shipped platform directories to be present.",
    )
    args = parser.parse_args(argv)

    required = () if args.allow_missing_platforms else REQUIRED_PLATFORMS
    try:
        result = verify(args.executable_root, args.manifest, required_platforms=required)
    except FileNotFoundError as exc:
        print(f"manifest load failed: {exc}", file=sys.stderr)
        return 2
    except (json.JSONDecodeError, ValueError) as exc:
        print(f"manifest is invalid: {exc}", file=sys.stderr)
        return 2

    print(result.render())
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
