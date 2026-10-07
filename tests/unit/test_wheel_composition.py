import hashlib
import json
import shutil
import subprocess
import zipfile
from pathlib import Path

import pytest

from dev.compose_compileiq_wheel import compose_wheel, write_composition_manifest
from dev.verify_complete_wheel import (
    _verify_linux_native_abi,
    _verify_platform,
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


def _composition_inputs(
    tmp_path: Path,
    extra_frontend_files: dict[str, bytes] | None = None,
) -> tuple[Path, Path, Path]:
    frontend = _write_wheel(
        tmp_path / "compileiq-1.1.0b0-py3-none-any.whl",
        {
            "compileiq/__init__.py": b"",
            **(extra_frontend_files or {}),
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
    """Publish one CompileIQ wheel containing the frontend and the native provider."""

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
        assert not any(name.startswith("compileiq/core/") for name in names)
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


def _checked_in_manifest(tmp_path: Path, runtime: Path) -> Path:
    """Write the runtime-only manifest shape kept under native-runtime-wheels/."""

    manifest = tmp_path / "native-runtime-manifest-cp311-cp311-linux_x86_64.json"
    manifest.write_text(
        json.dumps({"schema_version": 1, "outputs": {"native_runtime": _file_record(runtime)}})
    )
    return manifest


def test_compose_from_checked_in_manifest_with_a_frontend_built_here(
    tmp_path,
    _stub_native_abi_scan,
):
    """Public CI composes its own frontend wheel with the checked-in runtime manifests."""

    frontend, runtime, _ = _composition_inputs(tmp_path)
    manifest = _checked_in_manifest(tmp_path, runtime)

    with pytest.raises(RuntimeError, match="missing wheel provenance"):
        compose_wheel(frontend, runtime, manifest, tmp_path / "complete")

    output = compose_wheel(
        frontend,
        runtime,
        manifest,
        tmp_path / "complete",
        frontend_from_this_build=True,
    )
    composition_manifest = tmp_path / "complete" / "composition-manifest.json"
    write_composition_manifest(output, frontend, runtime, manifest, composition_manifest)
    verify_complete_wheel(output)
    verify_composition_manifest(output, composition_manifest)
    recorded = json.loads(composition_manifest.read_text())
    assert recorded["inputs"]["frontend"] == _file_record(frontend)


def test_frontend_from_this_build_still_verifies_runtime_and_any_frontend_record(tmp_path):
    frontend, runtime, lane_manifest = _composition_inputs(tmp_path)
    with frontend.open("ab") as destination:
        destination.write(b"rebuilt")

    with pytest.raises(RuntimeError, match="frontend wheel SHA-256"):
        compose_wheel(
            frontend,
            runtime,
            lane_manifest,
            tmp_path / "complete",
            frontend_from_this_build=True,
        )

    checked_in = _checked_in_manifest(tmp_path, runtime)
    with runtime.open("ab") as destination:
        destination.write(b"tampered")
    with pytest.raises(RuntimeError, match="native runtime wheel SHA-256"):
        compose_wheel(
            frontend,
            runtime,
            checked_in,
            tmp_path / "complete",
            frontend_from_this_build=True,
        )


def test_verify_complete_wheel_rejects_filename_metadata_tag_mismatch(tmp_path):
    frontend, runtime, manifest = _composition_inputs(tmp_path)
    output = compose_wheel(frontend, runtime, manifest, tmp_path / "complete")
    mismatched = tmp_path / output.name.replace("x86_64", "aarch64")
    shutil.copyfile(output, mismatched)

    with pytest.raises(RuntimeError, match="filename and WHEEL metadata tags differ"):
        verify_complete_wheel(mismatched)


def test_verify_complete_wheel_rejects_files_from_the_removed_core(
    tmp_path,
    _stub_native_abi_scan,
):
    """A frontend built before the core was removed must not compose into a release."""

    frontend, runtime, manifest = _composition_inputs(
        tmp_path,
        {"compileiq/core/executable/linux/x86_64/bin/_core": b"legacy-core"},
    )
    output = compose_wheel(frontend, runtime, manifest, tmp_path / "complete")

    with pytest.raises(RuntimeError, match="removed bundled core"):
        verify_complete_wheel(output)


@pytest.mark.parametrize(
    "platform_tag",
    ["manylinux_2_34_x86_64", "manylinux_2_34_aarch64", "win_amd64", "macosx_15_0_arm64"],
)
def test_every_supported_platform_ships_without_a_bundled_core(platform_tag):
    _verify_platform(["compileiq/__init__.py"], [f"cp313-cp313-{platform_tag}"])


def test_verify_platform_rejects_an_unsupported_platform():
    with pytest.raises(RuntimeError, match="unsupported platforms"):
        _verify_platform(["compileiq/__init__.py"], ["cp313-cp313-linux_riscv64"])
