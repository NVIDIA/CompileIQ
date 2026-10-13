#!/usr/bin/env python3
"""Materialize a verified native payload into an editable developer environment."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from packaging.tags import sys_tags
from packaging.utils import InvalidWheelFilename, parse_wheel_filename

if __package__:
    from dev.compose_compileiq_wheel import verify_file_record
else:
    from compose_compileiq_wheel import verify_file_record


LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1"
REGISTRATION_PROBE = (
    "from compileiq.backends import get_backend_registration; "
    "registration = get_backend_registration('amber'); "
    "assert registration.name == 'amber'"
)


def _reject_lfs_pointer(wheel: Path) -> None:
    """Turn an unresolved LFS checkout into an actionable bootstrap error."""

    if wheel.stat().st_size < 1024 and wheel.read_bytes()[:64].startswith(LFS_POINTER_PREFIX):
        raise RuntimeError(
            f"{wheel} is an unresolved Git LFS pointer. Run `git lfs install` and "
            "`git lfs pull` to fetch the native runtime wheel, then retry."
        )


def _runtime_record(lane_manifest: Path) -> dict[str, object]:
    """Read the single native artifact record authenticated by the lane manifest."""

    lane_manifest = lane_manifest.resolve()
    manifest = json.loads(lane_manifest.read_text())
    if manifest.get("schema_version") != 1:
        raise RuntimeError("unsupported release-lane manifest schema")
    try:
        runtime_record = manifest["outputs"]["native_runtime"]
    except (KeyError, TypeError) as exc:
        raise RuntimeError("release-lane manifest is missing native runtime provenance") from exc
    if not isinstance(runtime_record, dict):
        raise RuntimeError("native runtime provenance must be a JSON object")
    return runtime_record


def _select_runtime_wheel(
    runtime_wheel: Path | None,
    runtime_directories: list[Path],
    lane_manifest: Path,
    runtime_record: dict[str, object],
) -> Path:
    """Select the manifest-named wheel without crossing platform or artifact identities."""

    if runtime_wheel is not None:
        selected = runtime_wheel.resolve()
    else:
        filename = runtime_record.get("filename")
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise RuntimeError("native runtime manifest has an invalid wheel filename")
        candidates: list[Path] = []
        for directory in [lane_manifest.resolve().parent, *runtime_directories]:
            candidate = directory.resolve() / filename
            if candidate not in candidates and candidate.is_file():
                candidates.append(candidate)
        if not candidates:
            searched = ", ".join(
                str(path.resolve()) for path in [lane_manifest.parent, *runtime_directories]
            )
            raise RuntimeError(
                f"native runtime wheel {filename!r} was not found; searched: {searched}"
            )
        selected = candidates[0]

    try:
        _, _, _, wheel_tags = parse_wheel_filename(selected.name)
    except InvalidWheelFilename as exc:
        raise RuntimeError(
            f"native runtime artifact is not a valid wheel: {selected.name}"
        ) from exc
    if wheel_tags.isdisjoint(sys_tags()):
        raise RuntimeError(
            f"{selected.name} is not compatible with this Python interpreter and platform. "
            "Fetch the native runtime wheel for the current Python ABI and operating system."
        )
    return selected


def _select_runtime_artifact(
    runtime_wheel: Path | None,
    runtime_directories: list[Path],
    lane_manifests: list[Path],
) -> tuple[Path, Path, dict[str, object]]:
    """Choose exactly one compatible wheel and its authenticating manifest.

    The checked-in runtime directory can carry one manifest per platform/Python cell.
    Selection remains fail-closed: zero compatible cells gives an actionable
    error, while more than one requires an explicit wheel and manifest rather
    than relying on filesystem order.
    """

    if not lane_manifests:
        raise RuntimeError("at least one native runtime lane manifest is required")

    matches: list[tuple[Path, Path, dict[str, object]]] = []
    advertised: list[str] = []
    for lane_manifest in lane_manifests:
        lane_manifest = lane_manifest.resolve()
        runtime_record = _runtime_record(lane_manifest)
        filename = runtime_record.get("filename")
        if isinstance(filename, str):
            advertised.append(filename)
        if runtime_wheel is not None and filename != runtime_wheel.name:
            continue
        try:
            selected = _select_runtime_wheel(
                runtime_wheel,
                runtime_directories,
                lane_manifest,
                runtime_record,
            )
        except RuntimeError:
            continue
        matches.append((selected, lane_manifest, runtime_record))

    if not matches:
        available = ", ".join(sorted(set(advertised))) or "none"
        raise RuntimeError(
            "no manifest-verified native runtime wheel is compatible with this "
            f"Python interpreter and platform. Available runtime artifacts: {available}"
        )
    if len(matches) != 1:
        choices = ", ".join(f"{wheel.name} ({manifest.name})" for wheel, manifest, _ in matches)
        raise RuntimeError(
            "multiple compatible native runtime artifacts were found: "
            f"{choices}. Set both NATIVE_RUNTIME_WHEEL and NATIVE_RUNTIME_MANIFEST "
            "to select one explicitly."
        )
    return matches[0]


def materialize(
    runtime_wheel: Path | None,
    lane_manifests: list[Path],
    runtime_directories: list[Path],
) -> None:
    """Verify the pinned CI artifact, then install it without resolving dependencies."""

    runtime_wheel, _, runtime_record = _select_runtime_artifact(
        runtime_wheel,
        runtime_directories,
        lane_manifests,
    )
    _reject_lfs_pointer(runtime_wheel)
    verify_file_record(runtime_wheel, runtime_record, "native runtime wheel")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--force-reinstall",
            str(runtime_wheel),
        ],
        check=True,
    )
    try:
        subprocess.run(
            [sys.executable, "-c", REGISTRATION_PROBE],
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "the native runtime wheel installed, but its amber backend could not register "
            "against this CompileIQ frontend. Use a runtime artifact built for this "
            "frontend revision."
        ) from exc


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-wheel", type=Path)
    parser.add_argument("--runtime-directory", action="append", default=[], type=Path)
    parser.add_argument("--lane-manifest", action="append", required=True, type=Path)
    args = parser.parse_args()
    try:
        materialize(args.runtime_wheel, args.lane_manifest, args.runtime_directory)
    except (RuntimeError, OSError) as exc:
        parser.exit(1, f"ERROR: {exc}\n")
    print("verified native runtime materialized for editable development")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
