#!/usr/bin/env python3
"""Verify one platform's complete CompileIQ wheel contract."""

from __future__ import annotations

import argparse
import base64
import configparser
import csv
import hashlib
import io
import json
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from email import policy
from email.parser import BytesParser
from pathlib import Path


EXPECTED_ENTRY_POINT = "compileiq_native_runtime:register_backend_artifact"
SUPPORTED_PLATFORM_TAGS = {
    "manylinux_2_34_aarch64",
    "manylinux_2_34_x86_64",
    "win_amd64",
    "macosx_15_0_arm64",
}
# CompileIQ 1.1 removed the original core. A complete wheel that still carries
# anything under this prefix was composed from a stale frontend.
REMOVED_CORE_PREFIX = "compileiq/core/"


def _exactly_one(names: list[str], suffix: str) -> str:
    matches = [name for name in names if name.endswith(suffix)]
    if len(matches) != 1:
        raise RuntimeError(f"expected one {suffix!r} file, found {matches}")
    return matches[0]


def _requirement_name(value: str) -> str:
    match = re.match(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)", value)
    if match is None:
        raise RuntimeError(f"could not parse requirement {value!r}")
    return re.sub(r"[-_.]+", "-", match.group(1)).lower()


def _filename_tags(wheel: Path) -> set[str]:
    if not wheel.name.endswith(".whl"):
        raise RuntimeError(f"complete wheel has invalid filename {wheel.name!r}")
    parts = wheel.name[:-4].rsplit("-", 3)
    if len(parts) != 4:
        raise RuntimeError(f"complete wheel has invalid filename {wheel.name!r}")
    _, interpreters, abis, platforms = parts
    return {
        f"{interpreter}-{abi}-{platform}"
        for interpreter in interpreters.split(".")
        for abi in abis.split(".")
        for platform in platforms.split(".")
    }


def _verify_platform(names: list[str], tags: list[str]) -> None:
    platforms = {tag.rsplit("-", 1)[-1] for tag in tags}
    unsupported = sorted(platforms - SUPPORTED_PLATFORM_TAGS)
    if unsupported:
        raise RuntimeError(f"complete wheel has unsupported platforms: {unsupported}")
    if len(platforms) != 1:
        raise RuntimeError(f"complete wheel spans more than one platform: {sorted(platforms)}")
    removed_core = [name for name in names if name.startswith(REMOVED_CORE_PREFIX)]
    if removed_core:
        raise RuntimeError(
            f"complete wheel contains files from the removed bundled core: {removed_core}"
        )


def _verify_linux_native_abi(
    archive: zipfile.ZipFile,
    extension: str,
    tags: list[str],
) -> None:
    """Apply the public ELF version ceiling to the embedded native extension."""

    if not any("manylinux" in tag or "-linux_" in tag for tag in tags):
        return
    if shutil.which("readelf") is None:
        raise RuntimeError(
            "readelf not found; install binutils to verify the Linux native runtime ABI"
        )
    checker = Path(__file__).with_name("check-linux-abi.py")
    with tempfile.TemporaryDirectory(prefix="ciq-native-abi-") as directory:
        extracted = Path(directory) / Path(extension).name
        extracted.write_bytes(archive.read(extension))
        completed = subprocess.run(
            [sys.executable, str(checker), str(extracted)],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            details = completed.stderr.strip() or completed.stdout.strip()
            raise RuntimeError(f"native runtime ABI compatibility check failed: {details}")


def verify_complete_wheel(wheel: Path) -> None:
    """Fail unless ``wheel`` is the complete, single-distribution product."""

    wheel = wheel.resolve()
    if wheel.name.endswith("-none-any.whl"):
        raise RuntimeError("complete CompileIQ wheel must be platform-specific")
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        metadata_name = _exactly_one(names, ".dist-info/METADATA")
        metadata = BytesParser(policy=policy.default).parsebytes(archive.read(metadata_name))
        distribution = re.sub(r"[-_.]+", "-", str(metadata["Name"])).lower()
        if distribution != "compileiq":
            raise RuntimeError(f"complete wheel owns unexpected distribution {distribution!r}")
        requirements = {
            _requirement_name(value) for value in (metadata.get_all("Requires-Dist") or [])
        }
        if "pymoo" not in requirements:
            raise RuntimeError("complete CompileIQ wheel must depend directly on pymoo")

        extensions = [
            name
            for name in names
            if name.startswith("compileiq_native_runtime/_native.")
            and name.endswith((".so", ".pyd"))
        ]
        if len(extensions) != 1:
            raise RuntimeError(f"complete wheel has unexpected native extensions: {extensions}")
        if any("compileiq_native_runtime-" in name and ".dist-info/" in name for name in names):
            raise RuntimeError("complete wheel must not install a second runtime distribution")

        entry_points_name = _exactly_one(names, ".dist-info/entry_points.txt")
        entry_points = configparser.ConfigParser(interpolation=None)
        entry_points.optionxform = str
        entry_points.read_string(archive.read(entry_points_name).decode("utf-8"))
        actual = entry_points.get("compileiq.backends", "amber", fallback=None)
        if actual != EXPECTED_ENTRY_POINT:
            raise RuntimeError(f"complete wheel has unexpected native entry point {actual!r}")

        wheel_metadata_name = _exactly_one(names, ".dist-info/WHEEL")
        wheel_metadata = archive.read(wheel_metadata_name).decode("utf-8")
        if "Root-Is-Purelib: false" not in wheel_metadata:
            raise RuntimeError("complete wheel metadata still marks the product as pure Python")
        tags = [
            line.removeprefix("Tag: ")
            for line in wheel_metadata.splitlines()
            if line.startswith("Tag: ")
        ]
        if not tags or any(tag.endswith("-any") for tag in tags):
            raise RuntimeError(f"complete wheel has invalid platform tags: {tags}")
        filename_tags = _filename_tags(wheel)
        if set(tags) != filename_tags:
            raise RuntimeError(
                "complete wheel filename and WHEEL metadata tags differ: "
                f"filename={sorted(filename_tags)}, metadata={sorted(tags)}"
            )
        _verify_platform(names, tags)
        _verify_linux_native_abi(archive, extensions[0], tags)

        record_name = _exactly_one(names, ".dist-info/RECORD")
        rows = list(csv.reader(io.StringIO(archive.read(record_name).decode("utf-8"))))
        recorded = {row[0] for row in rows}
        if recorded != set(names):
            missing = sorted(set(names) - recorded)
            extra = sorted(recorded - set(names))
            raise RuntimeError(f"wheel RECORD mismatch (missing={missing}, extra={extra})")
        for name, digest, size in rows:
            if name == record_name:
                if digest or size:
                    raise RuntimeError("RECORD must leave its own hash and size empty")
                continue
            contents = archive.read(name)
            algorithm, encoded = digest.split("=", 1)
            if algorithm != "sha256":
                raise RuntimeError(f"RECORD uses unsupported digest for {name}")
            actual_digest = (
                base64.urlsafe_b64encode(hashlib.sha256(contents).digest()).rstrip(b"=").decode()
            )
            if encoded != actual_digest or int(size) != len(contents):
                raise RuntimeError(f"RECORD does not match {name}")

    print(f"complete wheel verification passed: {wheel.name}")


def verify_composition_manifest(wheel: Path, manifest_path: Path) -> None:
    """Bind the audited wheel to the release-composition output record."""

    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != 1 or manifest.get("product") != (
        "complete-compileiq-wheel"
    ):
        raise RuntimeError("unsupported complete-wheel composition manifest")
    try:
        record = manifest["output"]["compileiq"]
    except (KeyError, TypeError) as exc:
        raise RuntimeError("composition manifest is missing its CompileIQ output") from exc
    expected_digest = str(record.get("sha256", "")).removeprefix("sha256:")
    actual_digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
    if (
        record.get("filename") != wheel.name
        or record.get("size") != wheel.stat().st_size
        or expected_digest != actual_digest
    ):
        raise RuntimeError("complete wheel does not match its composition manifest")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("wheel", type=Path)
    parser.add_argument("--composition-manifest", type=Path)
    args = parser.parse_args()
    verify_complete_wheel(args.wheel)
    if args.composition_manifest is not None:
        verify_composition_manifest(args.wheel.resolve(), args.composition_manifest.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
