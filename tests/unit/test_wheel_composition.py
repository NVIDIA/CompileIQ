import hashlib
import json
import shutil
import subprocess
import zipfile
from pathlib import Path

import pytest

from dev.compose_compileiq_wheel import compose_wheel, write_composition_manifest
from dev.verify_complete_wheel import (
    _verify_core_platform,
    _verify_linux_native_abi,
    verify_complete_wheel,
    verify_composition_manifest,
)


@pytest.fixture
def _stub_native_abi_scan(monkeypatch):
    """The synthetic wheel payload is not an ELF file; ABI behavior is integration-tested."""

    monkeypatch.setattr("dev.verify_complete_wheel._verify_linux_native_abi", lambda *args: None)


def _write_wheel(path: Path, files: dict[str, bytes]) -> Path:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, contents in files.items():
            archive.writestr(name, contents)
    return path


def _file_record(path: Path) -> dict[str, str | int]:
    return {
        "filename": path.name,
        "sha256": f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}",
        "size": path.stat().st_size,
    }


def _composition_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    core_manifest = json.dumps(
        {
            "schema_version": 2,
            "files": {
                "linux/x86_64/bin/_core": "sha256:" + "0" * 64,
            },
        }
    ).encode()
    frontend = _write_wheel(
        tmp_path / "compileiq-1.1.0b0-py3-none-any.whl",
        {
            "compileiq/__init__.py": b"",
            "compileiq/core/executable/linux/x86_64/bin/_core": b"legacy-core",
            "compileiq/core/executable/core-manifest.json": core_manifest,
            "compileiq-1.1.0b0.dist-info/METADATA": (
                b"Metadata-Version: 2.3\n"
                b"Name: compileiq\n"
                b"Version: 1.1.0b0\n"
                b"Requires-Dist: pydantic (>=2.9.2,<3.0.0)\n\n"
            ),
            "compileiq-1.1.0b0.dist-info/WHEEL": (
                b"Wheel-Version: 1.0\n" b"Root-Is-Purelib: true\n" b"Tag: py3-none-any\n"
            ),
            "compileiq-1.1.0b0.dist-info/RECORD": b"",
        },
    )
    runtime = _write_wheel(
        tmp_path / "compileiq_native_runtime-1.1.0.dev0-cp311-cp311-linux_x86_64.whl",
        {
            "compileiq_native_runtime/__init__.py": b"def register_backend_artifact(): pass\n",
            "compileiq_native_runtime/_native.cpython-311-x86_64-linux-gnu.so": b"native",
            "compileiq_native_runtime-1.1.0.dev0.dist-info/METADATA": (
                b"Metadata-Version: 2.3\n"
                b"Name: compileiq-native-runtime\n"
                b"Version: 1.1.0.dev0\n"
                b"Requires-Dist: compileiq (>=1.1.0.dev0,<2.0.0)\n"
                b"Requires-Dist: pydantic (>=2.9.2)\n"
                b"Requires-Dist: pymoo (>=0.6.1,<0.7)\n\n"
            ),
            "compileiq_native_runtime-1.1.0.dev0.dist-info/entry_points.txt": (
                b"[compileiq.backends]\n"
                b"amber=compileiq_native_runtime:register_backend_artifact\n"
            ),
            "compileiq_native_runtime-1.1.0.dev0.dist-info/WHEEL": (
                b"Wheel-Version: 1.0\n"
                b"Root-Is-Purelib: false\n"
                b"Tag: cp311-cp311-linux_x86_64\n"
            ),
            "compileiq_native_runtime-1.1.0.dev0.dist-info/RECORD": b"",
        },
    )
    manifest = tmp_path / "native-runtime-lane-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "inputs": {"public_compileiq": {"wheel": _file_record(frontend)}},
                "outputs": {"native_runtime": _file_record(runtime)},
            }
        )
    )
    return frontend, runtime, manifest


def test_compose_complete_wheel_uses_one_distribution_and_platform_tag(
    tmp_path,
    _stub_native_abi_scan,
):
    """Publish one CompileIQ wheel containing both legacy and native providers."""

    frontend, runtime, manifest = _composition_inputs(tmp_path)
    output = compose_wheel(frontend, runtime, manifest, tmp_path / "complete")
    composition_manifest = tmp_path / "complete" / "composition-manifest.json"
    write_composition_manifest(
        output,
        frontend,
        runtime,
        manifest,
        composition_manifest,
    )

    assert output.name == "compileiq-1.1.0b0-cp311-cp311-manylinux_2_34_x86_64.whl"
    verify_complete_wheel(output)
    verify_composition_manifest(output, composition_manifest)
    with zipfile.ZipFile(output) as archive:
        names = archive.namelist()
        assert "compileiq/core/executable/linux/x86_64/bin/_core" in names
        assert "compileiq_native_runtime/_native.cpython-311-x86_64-linux-gnu.so" in names
        assert not any("compileiq_native_runtime-" in name for name in names)
        assert b"Tag: cp311-cp311-manylinux_2_34_x86_64" in archive.read(
            "compileiq-1.1.0b0.dist-info/WHEEL"
        )
        assert archive.read("compileiq-1.1.0b0.dist-info/METADATA").startswith(
            b"Metadata-Version: 2.3\nName: compileiq\n"
        )
        metadata = archive.read("compileiq-1.1.0b0.dist-info/METADATA").decode()
        assert "Requires-Dist: pymoo" in metadata
        assert metadata.count("Requires-Dist: pydantic") == 1
        assert "Requires-Dist: compileiq " not in metadata


def test_linux_native_abi_scan_requires_readelf(tmp_path, monkeypatch):
    wheel = _write_wheel(tmp_path / "runtime.whl", {"runtime/_native.so": b"not-elf"})
    monkeypatch.setattr("dev.verify_complete_wheel.shutil.which", lambda executable: None)

    with zipfile.ZipFile(wheel) as archive:
        with pytest.raises(RuntimeError, match="readelf not found"):
            _verify_linux_native_abi(
                archive,
                "runtime/_native.so",
                ["cp311-cp311-manylinux_2_34_x86_64"],
            )


def test_linux_native_abi_scan_reports_checker_failure(tmp_path, monkeypatch):
    wheel = _write_wheel(tmp_path / "runtime.whl", {"runtime/_native.so": b"not-elf"})
    monkeypatch.setattr(
        "dev.verify_complete_wheel.shutil.which",
        lambda executable: "/usr/bin/readelf",
    )
    monkeypatch.setattr(
        "dev.verify_complete_wheel.subprocess.run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args=args,
            returncode=1,
            stdout="",
            stderr="GLIBCXX requirement exceeds the supported ceiling",
        ),
    )

    with zipfile.ZipFile(wheel) as archive:
        with pytest.raises(RuntimeError, match="GLIBCXX requirement exceeds"):
            _verify_linux_native_abi(
                archive,
                "runtime/_native.so",
                ["cp311-cp311-manylinux_2_34_x86_64"],
            )


def test_compose_complete_wheel_rejects_unverified_runtime(tmp_path):
    """Refuse payload substitution after the runtime artifact was manifested."""

    frontend, runtime, manifest = _composition_inputs(tmp_path)
    with runtime.open("ab") as destination:
        destination.write(b"tampered")

    with pytest.raises(RuntimeError, match="SHA-256"):
        compose_wheel(frontend, runtime, manifest, tmp_path / "complete")


def test_verify_complete_wheel_rejects_filename_metadata_tag_mismatch(tmp_path):
    frontend, runtime, manifest = _composition_inputs(tmp_path)
    output = compose_wheel(frontend, runtime, manifest, tmp_path / "complete")
    mismatched = tmp_path / output.name.replace("x86_64", "aarch64")
    shutil.copyfile(output, mismatched)

    with pytest.raises(RuntimeError, match="filename and WHEEL metadata tags differ"):
        verify_complete_wheel(mismatched)


def test_verify_complete_wheel_rejects_missing_core_for_tagged_platform(tmp_path):
    frontend, runtime, manifest = _composition_inputs(tmp_path)
    output = compose_wheel(frontend, runtime, manifest, tmp_path / "complete")
    mismatched = tmp_path / output.name.replace("x86_64", "aarch64")
    with zipfile.ZipFile(output) as source, zipfile.ZipFile(mismatched, "w") as destination:
        for info in source.infolist():
            contents = source.read(info.filename)
            if info.filename.endswith(".dist-info/WHEEL"):
                contents = contents.replace(b"x86_64", b"aarch64")
            destination.writestr(info, contents)

    with pytest.raises(RuntimeError, match="does not have a bundled core for linux/aarch64"):
        verify_complete_wheel(mismatched)


def test_macos_complete_wheel_allows_no_legacy_core(tmp_path):
    wheel = _write_wheel(tmp_path / "frontend.whl", {"compileiq/__init__.py": b""})

    with zipfile.ZipFile(wheel) as archive:
        _verify_core_platform(
            archive.namelist(),
            ["cp313-cp313-macosx_15_0_arm64"],
            archive,
        )


def test_macos_complete_wheel_rejects_a_foreign_legacy_core(tmp_path):
    wheel = _write_wheel(
        tmp_path / "frontend.whl",
        {"compileiq/core/executable/linux/x86_64/bin/_core": b"core"},
    )

    with zipfile.ZipFile(wheel) as archive:
        with pytest.raises(RuntimeError, match="coreless complete wheel unexpectedly contains"):
            _verify_core_platform(
                archive.namelist(),
                ["cp313-cp313-macosx_15_0_arm64"],
                archive,
            )
