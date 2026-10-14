#!/usr/bin/env python3
"""Validate checked-in native runtime artifacts for public distribution."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


RUNTIME_DIRECTORY = "native-runtime-wheels"
COMMIT_SHA = re.compile(r"^[0-9a-f]{40}$")
PYTHON_TAGS = ("cp311", "cp312", "cp313", "cp314")
PLATFORM_TAGS = (
    "linux_x86_64",
    "linux_aarch64",
    "win_amd64",
    "macosx_15_0_arm64",
)
EXPECTED_RUNTIME_TAGS = {
    f"{python_tag}-{python_tag}-{platform_tag}"
    for python_tag in PYTHON_TAGS
    for platform_tag in PLATFORM_TAGS
}
# Build-machine path fragments that should never be embedded in a published wheel.
FORBIDDEN_ARTIFACT_BYTES = (
    b"/users/",
    b"/home/",
    b"/builds/",
    b"\\users\\",
    b"\\builds\\",
)
# Producer-specific private markers are not listed in this public script. CI that
# owns them supplies a JSON file, {"public_text": [...], "artifact_bytes": [...]},
# through --markers or this environment variable. Both lists match case-insensitively.
MARKERS_ENVIRONMENT_VARIABLE = "CIQ_PUBLIC_ARTIFACT_MARKERS"
PUBLIC_TEXT_FILES = ("AGENTS.md", "README.md")
PUBLIC_TEXT_ROOTS = ("docs", "agent-skills", RUNTIME_DIRECTORY)
LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _exact_keys(value: object, expected: set[str], label: str) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != expected:
        actual = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise RuntimeError(f"{label} has keys {actual}, expected {sorted(expected)}")
    return value


@dataclass(frozen=True)
class PrivateMarkers:
    """Caller-supplied markers that must not appear in public text or artifacts."""

    public_text: tuple[str, ...] = ()
    artifact_bytes: tuple[str, ...] = ()


def load_private_markers(path: Path) -> PrivateMarkers:
    """Read a private marker file supplied by the CI that owns the markers."""

    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read private marker file: {path}") from exc
    if not isinstance(document, dict) or not set(document) <= {"public_text", "artifact_bytes"}:
        raise RuntimeError(
            "private marker file must be a JSON object with only "
            "'public_text' and 'artifact_bytes' lists"
        )
    lists: dict[str, tuple[str, ...]] = {}
    for key in ("public_text", "artifact_bytes"):
        values = document.get(key, [])
        if not isinstance(values, list) or not all(
            isinstance(value, str) and value for value in values
        ):
            raise RuntimeError(f"private marker file '{key}' must be a list of non-empty strings")
        lists[key] = tuple(values)
    return PrivateMarkers(**lists)


def _artifact_leaks(contents: bytes, markers: PrivateMarkers) -> list[str]:
    lowered = contents.lower()
    forbidden = FORBIDDEN_ARTIFACT_BYTES + tuple(
        marker.lower().encode() for marker in markers.artifact_bytes
    )
    return [marker.decode() for marker in forbidden if marker in lowered]


def _check_wheel(wheel: Path, expected_record: dict[str, object], markers: PrivateMarkers) -> None:
    if wheel.stat().st_size < 1024 and wheel.read_bytes()[:64].startswith(LFS_POINTER_PREFIX):
        raise RuntimeError(
            f"{wheel} is an unresolved Git LFS pointer; fetch LFS objects before validation"
        )
    actual_record = {
        "filename": wheel.name,
        "sha256": f"sha256:{_sha256(wheel)}",
        "size": wheel.stat().st_size,
    }
    if actual_record != expected_record:
        raise RuntimeError(
            "native runtime wheel does not match its public manifest: "
            f"expected {expected_record!r}, found {actual_record!r}"
        )

    try:
        with zipfile.ZipFile(wheel) as archive:
            extensions: list[str] = []
            for member in archive.infolist():
                name = PurePosixPath(member.filename)
                if name.is_absolute() or ".." in name.parts:
                    raise RuntimeError(f"runtime wheel contains an unsafe path: {member.filename}")
                if member.filename.endswith((".pyx", ".cpp", ".c", ".h", ".hpp")):
                    raise RuntimeError(f"runtime wheel contains native source: {member.filename}")
                if member.filename.startswith("compileiq_native_runtime/_native.") and (
                    member.filename.endswith(".so") or member.filename.endswith(".pyd")
                ):
                    extensions.append(member.filename)
                leaks = _artifact_leaks(archive.read(member), markers)
                if leaks:
                    raise RuntimeError(
                        f"runtime wheel member {member.filename} contains private markers: {leaks}"
                    )
    except zipfile.BadZipFile as exc:
        raise RuntimeError(f"native runtime artifact is not a valid wheel: {wheel}") from exc
    if len(extensions) != 1:
        raise RuntimeError(f"runtime wheel contains {len(extensions)} native extensions")


def _check_runtime_matrix(repository: Path, markers: PrivateMarkers) -> None:
    runtime_directory = repository / RUNTIME_DIRECTORY
    if not runtime_directory.is_dir():
        raise RuntimeError(f"required public runtime directory is missing: {RUNTIME_DIRECTORY}/")
    allowed_names = {"README.md"}
    wheel_paths = sorted(runtime_directory.glob("compileiq_native_runtime-*.whl"))
    manifest_paths = sorted(runtime_directory.glob("native-runtime-manifest-*.json"))
    allowed_names.update(path.name for path in wheel_paths)
    allowed_names.update(path.name for path in manifest_paths)
    unexpected = sorted(
        path.name for path in runtime_directory.iterdir() if path.name not in allowed_names
    )
    if unexpected:
        raise RuntimeError(f"public runtime directory contains unexpected entries: {unexpected}")
    expected_count = len(EXPECTED_RUNTIME_TAGS)
    if len(wheel_paths) != expected_count or len(manifest_paths) != expected_count:
        raise RuntimeError(
            f"public runtime matrix must contain exactly {expected_count} wheels and manifests"
        )

    wheels = {path.name: path for path in wheel_paths}
    seen_tags: set[str] = set()
    producer_pipeline_ids: set[int] = set()
    producer_job_ids: set[int] = set()
    producer_source_commits: set[str] = set()
    for manifest_path in manifest_paths:
        try:
            manifest = json.loads(manifest_path.read_text())
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"invalid public runtime manifest: {manifest_path}") from exc
        manifest = _exact_keys(
            manifest,
            {"schema_version", "provenance", "runtime", "outputs"},
            manifest_path.name,
        )
        if manifest["schema_version"] != 1:
            raise RuntimeError(f"unsupported public runtime manifest: {manifest_path}")
        provenance = _exact_keys(
            manifest["provenance"],
            {"pipeline_id", "job_id", "source_commit"},
            f"{manifest_path.name} provenance",
        )
        pipeline_id = provenance["pipeline_id"]
        job_id = provenance["job_id"]
        if type(pipeline_id) is not int or pipeline_id <= 0:
            raise RuntimeError(
                f"{manifest_path.name} has an invalid producer pipeline_id: {pipeline_id!r}"
            )
        if type(job_id) is not int or job_id <= 0:
            raise RuntimeError(f"{manifest_path.name} has an invalid producer job_id: {job_id!r}")
        producer_pipeline_ids.add(pipeline_id)
        producer_job_ids.add(job_id)
        source_commit = provenance["source_commit"]
        if not isinstance(source_commit, str) or COMMIT_SHA.fullmatch(source_commit) is None:
            raise RuntimeError(
                f"{manifest_path.name} has an invalid producer source commit: " f"{source_commit!r}"
            )
        producer_source_commits.add(source_commit)
        runtime = _exact_keys(
            manifest["runtime"],
            {"python_tag", "abi_tag", "platform_tag", "wheel_tag"},
            f"{manifest_path.name} runtime",
        )
        python_tag, abi_tag, platform_tag, wheel_tag = (
            runtime["python_tag"],
            runtime["abi_tag"],
            runtime["platform_tag"],
            runtime["wheel_tag"],
        )
        if not all(isinstance(tag, str) for tag in (python_tag, abi_tag, platform_tag, wheel_tag)):
            raise RuntimeError(f"{manifest_path.name} runtime tags must all be strings")
        if wheel_tag != f"{python_tag}-{abi_tag}-{platform_tag}":
            raise RuntimeError(
                f"{manifest_path.name} wheel_tag {wheel_tag!r} does not match its "
                f"python/abi/platform tags {python_tag!r}-{abi_tag!r}-{platform_tag!r}"
            )
        if wheel_tag not in EXPECTED_RUNTIME_TAGS:
            raise RuntimeError(f"unexpected public runtime tag in {manifest_path}: {wheel_tag!r}")
        if manifest_path.name != f"native-runtime-manifest-{wheel_tag}.json":
            raise RuntimeError(f"public runtime manifest filename does not match {wheel_tag}")
        seen_tags.add(wheel_tag)
        outputs = _exact_keys(
            manifest["outputs"], {"native_runtime"}, f"{manifest_path.name} outputs"
        )
        record = _exact_keys(
            outputs["native_runtime"],
            {"filename", "sha256", "size"},
            f"{manifest_path.name} native runtime record",
        )
        filename = record["filename"]
        if not isinstance(filename, str) or not filename.endswith(f"-{wheel_tag}.whl"):
            raise RuntimeError(f"public runtime record has an invalid filename: {filename!r}")
        wheel = wheels.get(filename)
        if wheel is None:
            raise RuntimeError(f"manifest-named runtime wheel is missing: {filename}")
        _check_wheel(wheel, record, markers)
    if seen_tags != EXPECTED_RUNTIME_TAGS:
        raise RuntimeError(
            f"public runtime tags are {sorted(seen_tags)}, expected {sorted(EXPECTED_RUNTIME_TAGS)}"
        )
    if len(producer_pipeline_ids) != 1 or len(producer_source_commits) != 1:
        raise RuntimeError("public runtime manifests contain mixed producer coordinates")
    if len(producer_job_ids) != expected_count:
        raise RuntimeError("public runtime manifests contain duplicate producer job IDs")


def _check_public_text(repository: Path, markers: PrivateMarkers) -> None:
    public_text_paths = [repository / name for name in PUBLIC_TEXT_FILES]
    for root_name in PUBLIC_TEXT_ROOTS:
        root = repository / root_name
        if not root.is_dir():
            continue
        public_text_paths.extend(path for path in root.rglob("*") if path.is_file())

    leaks: list[str] = []
    for path in public_text_paths:
        if not path.is_file() or path.suffix == ".whl":
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for forbidden in markers.public_text:
            if forbidden.casefold() in text.casefold():
                leaks.append(f"{path.relative_to(repository)}: {forbidden}")
    if leaks:
        raise RuntimeError(
            "public documentation contains private implementation identifiers: "
            + ", ".join(sorted(leaks))
        )


def check(repository: Path, markers: PrivateMarkers | None = None) -> None:
    """Validate the complete checked-in public runtime and documentation surface."""

    markers = markers or PrivateMarkers()

    repository = repository.resolve()
    _check_runtime_matrix(repository, markers)
    _check_public_text(repository, markers)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", default=Path.cwd(), type=Path)
    parser.add_argument(
        "--markers",
        default=os.environ.get(MARKERS_ENVIRONMENT_VARIABLE) or None,
        type=Path,
        help=f"JSON private marker file (default: ${MARKERS_ENVIRONMENT_VARIABLE})",
    )
    args = parser.parse_args(argv)
    try:
        markers = load_private_markers(args.markers) if args.markers else PrivateMarkers()
        check(args.repository, markers)
    except RuntimeError as exc:
        parser.exit(1, f"ERROR: {exc}\n")
    if args.markers:
        print("public artifact guard passed")
    else:
        print("public artifact guard passed (generic checks only; no private marker file)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
