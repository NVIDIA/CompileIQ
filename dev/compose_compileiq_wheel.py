#!/usr/bin/env python3
"""Compose a complete platform CompileIQ wheel from verified release inputs.

The separately built ``compileiq-native-runtime`` wheel is an intermediate build
artifact. This tool copies its import package and backend entry point into the
public ``compileiq`` wheel, producing the only wheel intended for publication.
"""

from __future__ import annotations

import argparse
import base64
import configparser
import csv
import hashlib
import io
import json
import re
import zipfile
from copy import copy
from email import policy
from email.parser import BytesParser
from pathlib import Path

from packaging.requirements import InvalidRequirement, Requirement


EXPECTED_ENTRY_POINT = (
    "compileiq.backends",
    "amber",
    "compileiq_native_runtime:register_backend_artifact",
)

MANYLINUX_PLATFORM_TAGS = {
    "linux_aarch64": "manylinux_2_34_aarch64",
    "linux_x86_64": "manylinux_2_34_x86_64",
}


def sha256(path: Path) -> str:
    """Return one artifact's lowercase SHA-256 digest."""

    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, str | int]:
    """Return the filename, digest, and size used in release manifests."""

    return {
        "filename": path.name,
        "sha256": f"sha256:{sha256(path)}",
        "size": path.stat().st_size,
    }


def _canonical_distribution(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _normalize_wheel_tag(tag: str) -> str:
    parts = tag.split("-", 2)
    if len(parts) != 3:
        raise RuntimeError(f"invalid wheel tag {tag!r}")
    interpreter, abi, platform = parts
    return "-".join((interpreter, abi, MANYLINUX_PLATFORM_TAGS.get(platform, platform)))


def _normalized_wheel_metadata(contents: bytes, filename_tag: str) -> bytes:
    text = contents.decode("utf-8")
    tags = [line.removeprefix("Tag: ") for line in text.splitlines() if line.startswith("Tag: ")]
    if tags != [filename_tag]:
        raise RuntimeError(
            "native runtime WHEEL tag does not match its filename: "
            f"filename={filename_tag!r}, metadata={tags!r}"
        )
    normalized_tag = _normalize_wheel_tag(filename_tag)
    return text.replace(f"Tag: {filename_tag}", f"Tag: {normalized_tag}").encode("utf-8")


def _exactly_one(names: list[str], suffix: str) -> str:
    matches = [name for name in names if name.endswith(suffix)]
    if len(matches) != 1:
        raise RuntimeError(f"expected one {suffix!r} file, found {matches}")
    return matches[0]


def _metadata(archive: zipfile.ZipFile) -> tuple[str, str, str]:
    names = archive.namelist()
    metadata_name = _exactly_one(names, ".dist-info/METADATA")
    metadata = BytesParser(policy=policy.default).parsebytes(archive.read(metadata_name))
    return metadata_name, str(metadata["Name"]), str(metadata["Version"])


def _merged_metadata(frontend_contents: bytes, runtime_contents: bytes) -> bytes:
    """Merge runtime-only dependencies into the one published distribution."""

    frontend = BytesParser(policy=policy.default).parsebytes(frontend_contents)
    runtime = BytesParser(policy=policy.default).parsebytes(runtime_contents)
    existing: set[str] = set()
    for value in frontend.get_all("Requires-Dist") or []:
        try:
            existing.add(_canonical_distribution(Requirement(value).name))
        except InvalidRequirement as exc:
            raise RuntimeError(f"frontend wheel has an invalid dependency: {value!r}") from exc

    for value in runtime.get_all("Requires-Dist") or []:
        try:
            distribution = _canonical_distribution(Requirement(value).name)
        except InvalidRequirement as exc:
            raise RuntimeError(
                f"native runtime wheel has an invalid dependency: {value!r}"
            ) from exc
        if distribution == "compileiq" or distribution in existing:
            continue
        frontend["Requires-Dist"] = value
        existing.add(distribution)

    return frontend.as_bytes(policy=policy.default.clone(linesep="\n", max_line_length=0))


def verify_file_record(path: Path, record: dict[str, object], label: str) -> None:
    expected_filename = record.get("filename")
    expected_hash = str(record.get("sha256", "")).removeprefix("sha256:")
    expected_size = record.get("size")
    if path.name != expected_filename:
        raise RuntimeError(
            f"{label} filename {path.name!r} does not match manifest {expected_filename!r}"
        )
    if sha256(path) != expected_hash:
        raise RuntimeError(f"{label} SHA-256 does not match the release manifest")
    if path.stat().st_size != expected_size:
        raise RuntimeError(f"{label} size does not match the release manifest")


def verify_lane_manifest(
    manifest_path: Path,
    frontend_wheel: Path,
    runtime_wheel: Path,
    *,
    frontend_from_this_build: bool = False,
) -> dict[str, object]:
    """Verify the composition inputs against the native-runtime lane manifest.

    A release-lane manifest binds both input wheels. The checked-in public
    manifests under ``native-runtime-wheels/`` bind only the runtime wheel; with
    ``frontend_from_this_build`` the caller attests that it built the frontend
    wheel from this checkout, and the composition manifest records its digest.
    A frontend record, when present, is always verified.
    """

    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != 1:
        raise RuntimeError("unsupported release-lane manifest schema")
    try:
        public_record = (
            None
            if frontend_from_this_build and "inputs" not in manifest
            else manifest["inputs"]["public_compileiq"]["wheel"]
        )
        runtime_record = manifest["outputs"]["native_runtime"]
    except (KeyError, TypeError) as exc:
        raise RuntimeError("release-lane manifest is missing wheel provenance") from exc
    if (public_record is not None and not isinstance(public_record, dict)) or not isinstance(
        runtime_record, dict
    ):
        raise RuntimeError("release-lane wheel provenance must be JSON objects")
    if public_record is not None:
        verify_file_record(frontend_wheel, public_record, "frontend wheel")
    verify_file_record(runtime_wheel, runtime_record, "native runtime wheel")
    return manifest


def _parse_entry_points(contents: bytes) -> configparser.ConfigParser:
    parser = configparser.ConfigParser(interpolation=None)
    parser.optionxform = str
    parser.read_string(contents.decode("utf-8"))
    return parser


def _entry_points_bytes(
    frontend: zipfile.ZipFile,
    frontend_names: list[str],
    runtime: zipfile.ZipFile,
    runtime_names: list[str],
) -> bytes:
    runtime_name = _exactly_one(runtime_names, ".dist-info/entry_points.txt")
    runtime_points = _parse_entry_points(runtime.read(runtime_name))
    group, name, expected = EXPECTED_ENTRY_POINT
    if runtime_points.get(group, name, fallback=None) != expected:
        raise RuntimeError("native runtime wheel has an unexpected backend entry point")

    matches = [item for item in frontend_names if item.endswith(".dist-info/entry_points.txt")]
    if len(matches) > 1:
        raise RuntimeError(f"frontend wheel has multiple entry-point files: {matches}")
    points = (
        _parse_entry_points(frontend.read(matches[0]))
        if matches
        else configparser.ConfigParser(interpolation=None)
    )
    points.optionxform = str
    if not points.has_section(group):
        points.add_section(group)
    existing = points.get(group, name, fallback=None)
    if existing not in (None, expected):
        raise RuntimeError(f"frontend wheel already defines {group}/{name} as {existing!r}")
    points.set(group, name, expected)
    output = io.StringIO()
    points.write(output, space_around_delimiters=False)
    return output.getvalue().encode("utf-8")


def _renamed_info(source: zipfile.ZipInfo, filename: str) -> zipfile.ZipInfo:
    info = copy(source)
    info.filename = filename
    info.orig_filename = filename
    return info


def _record_bytes(files: dict[str, tuple[zipfile.ZipInfo, bytes]], record_name: str) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    for name in sorted(files):
        if name == record_name:
            continue
        contents = files[name][1]
        digest = base64.urlsafe_b64encode(hashlib.sha256(contents).digest()).rstrip(b"=").decode()
        writer.writerow((name, f"sha256={digest}", len(contents)))
    writer.writerow((record_name, "", ""))
    return output.getvalue().encode("utf-8")


def compose_wheel(
    frontend_wheel: Path,
    runtime_wheel: Path,
    lane_manifest: Path,
    output_dir: Path,
    *,
    frontend_from_this_build: bool = False,
) -> Path:
    """Return a manifest-verified, platform-specific complete CompileIQ wheel."""

    frontend_wheel = frontend_wheel.resolve()
    runtime_wheel = runtime_wheel.resolve()
    lane_manifest = lane_manifest.resolve()
    verify_lane_manifest(
        lane_manifest,
        frontend_wheel,
        runtime_wheel,
        frontend_from_this_build=frontend_from_this_build,
    )

    with zipfile.ZipFile(frontend_wheel) as frontend, zipfile.ZipFile(runtime_wheel) as runtime:
        frontend_names = frontend.namelist()
        runtime_names = runtime.namelist()
        frontend_metadata_name, frontend_distribution, frontend_version = _metadata(frontend)
        runtime_metadata_name, runtime_distribution, runtime_version = _metadata(runtime)
        if _canonical_distribution(frontend_distribution) != "compileiq":
            raise RuntimeError(f"unexpected frontend distribution {frontend_distribution!r}")
        if _canonical_distribution(runtime_distribution) != "compileiq-native-runtime":
            raise RuntimeError(f"unexpected runtime distribution {runtime_distribution!r}")

        frontend_dist_info = frontend_metadata_name.rsplit("/", 1)[0]
        frontend_wheel_metadata = f"{frontend_dist_info}/WHEEL"
        frontend_record = f"{frontend_dist_info}/RECORD"
        frontend_entry_points = f"{frontend_dist_info}/entry_points.txt"
        runtime_wheel_metadata = _exactly_one(runtime_names, ".dist-info/WHEEL")
        runtime_entry_points = _exactly_one(runtime_names, ".dist-info/entry_points.txt")

        runtime_prefix = f"compileiq_native_runtime-{runtime_version}"
        if not runtime_wheel.name.startswith(runtime_prefix + "-"):
            raise RuntimeError(
                f"runtime wheel filename does not start with expected {runtime_prefix!r}"
            )
        platform_suffix = runtime_wheel.name[len(runtime_prefix) :]
        if platform_suffix.endswith("-none-any.whl"):
            raise RuntimeError("native runtime wheel cannot use a platform-independent tag")
        if not platform_suffix.startswith("-") or not platform_suffix.endswith(".whl"):
            raise RuntimeError(f"native runtime wheel has invalid suffix {platform_suffix!r}")
        runtime_filename_tag = platform_suffix[1:-4]
        normalized_platform_suffix = f"-{_normalize_wheel_tag(runtime_filename_tag)}.whl"

        files = {
            info.filename: (info, frontend.read(info.filename))
            for info in frontend.infolist()
            if not info.is_dir() and info.filename not in {frontend_record, frontend_entry_points}
        }
        frontend_metadata_info = frontend.getinfo(frontend_metadata_name)
        files[frontend_metadata_name] = (
            frontend_metadata_info,
            _merged_metadata(
                frontend.read(frontend_metadata_name),
                runtime.read(runtime_metadata_name),
            ),
        )
        runtime_payload = [
            info
            for info in runtime.infolist()
            if not info.is_dir() and info.filename.startswith("compileiq_native_runtime/")
        ]
        extensions = [
            info.filename
            for info in runtime_payload
            if info.filename.startswith("compileiq_native_runtime/_native.")
            and info.filename.endswith((".so", ".pyd"))
        ]
        if len(extensions) != 1:
            raise RuntimeError(f"expected one native runtime extension, found {extensions}")
        for info in runtime_payload:
            if info.filename in files:
                raise RuntimeError(f"wheel payload collision at {info.filename}")
            files[info.filename] = (info, runtime.read(info.filename))

        wheel_info = runtime.getinfo(runtime_wheel_metadata)
        files[frontend_wheel_metadata] = (
            _renamed_info(wheel_info, frontend_wheel_metadata),
            _normalized_wheel_metadata(
                runtime.read(runtime_wheel_metadata),
                runtime_filename_tag,
            ),
        )
        entry_info = runtime.getinfo(runtime_entry_points)
        files[frontend_entry_points] = (
            _renamed_info(entry_info, frontend_entry_points),
            _entry_points_bytes(frontend, frontend_names, runtime, runtime_names),
        )
        record_info = frontend.getinfo(frontend_record)
        files[frontend_record] = (
            record_info,
            _record_bytes(files, frontend_record),
        )

        output_dir.mkdir(parents=True, exist_ok=True)
        output = output_dir / f"compileiq-{frontend_version}{normalized_platform_suffix}"
        with zipfile.ZipFile(output, "w") as composed:
            for name in sorted(files):
                info, contents = files[name]
                composed.writestr(info, contents)

    print(f"wrote complete wheel {output}")
    return output


def write_composition_manifest(
    output: Path,
    frontend_wheel: Path,
    runtime_wheel: Path,
    lane_manifest: Path,
    manifest_path: Path,
) -> None:
    """Record the verified inputs and single publishable output."""

    manifest = {
        "schema_version": 1,
        "product": "complete-compileiq-wheel",
        "lane_manifest": file_record(lane_manifest.resolve()),
        "inputs": {
            "frontend": file_record(frontend_wheel.resolve()),
            "native_runtime": file_record(runtime_wheel.resolve()),
        },
        "output": {"compileiq": file_record(output.resolve())},
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"wrote composition manifest {manifest_path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frontend-wheel", required=True, type=Path)
    parser.add_argument("--runtime-wheel", required=True, type=Path)
    parser.add_argument("--lane-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--output-manifest", required=True, type=Path)
    parser.add_argument(
        "--frontend-from-this-build",
        action="store_true",
        help=(
            "accept a lane manifest that binds only the runtime wheel, such as the "
            "checked-in native-runtime-wheels/ manifests, because the caller built "
            "the frontend wheel from this checkout"
        ),
    )
    args = parser.parse_args()
    output = compose_wheel(
        args.frontend_wheel,
        args.runtime_wheel,
        args.lane_manifest,
        args.output_dir,
        frontend_from_this_build=args.frontend_from_this_build,
    )
    write_composition_manifest(
        output,
        args.frontend_wheel,
        args.runtime_wheel,
        args.lane_manifest,
        args.output_manifest,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
