from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest
from packaging.tags import Tag, sys_tags

from dev import materialize_native_runtime


RUNTIME_MATRIX_TAGS = tuple(
    f"{python_tag}-{python_tag}-{platform_tag}"
    for python_tag in ("cp311", "cp312", "cp313", "cp314")
    for platform_tag in (
        "linux_x86_64",
        "linux_aarch64",
        "win_amd64",
        "macosx_15_0_arm64",
    )
)


def _manifest(
    path: Path,
    wheel: Path,
    name: str = "native-runtime-manifest.json",
) -> Path:
    manifest = path / name
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "outputs": {
                    "native_runtime": {
                        "filename": wheel.name,
                        "sha256": f"sha256:{hashlib.sha256(wheel.read_bytes()).hexdigest()}",
                        "size": wheel.stat().st_size,
                    }
                },
            }
        )
    )
    return manifest


def test_rejects_an_unresolved_lfs_pointer_before_artifact_verification(tmp_path: Path) -> None:
    """Explain the missing LFS pull instead of passing pointer text to pip as a wheel."""

    compatible_tag = next(sys_tags())
    wheel = tmp_path / f"compileiq_native_runtime-1.1.0-{compatible_tag}.whl"
    wheel.write_bytes(
        b"version https://git-lfs.github.com/spec/v1\n"
        b"oid sha256:0123456789abcdef\n"
        b"size 123456\n"
    )

    with pytest.raises(RuntimeError, match="unresolved Git LFS pointer.*git lfs pull"):
        materialize_native_runtime._reject_lfs_pointer(wheel)


def test_selects_the_manifest_named_wheel_from_the_manifest_directory(
    tmp_path: Path,
) -> None:
    """Keep one manifest and wheel paired even when other wheel files are nearby."""

    compatible_tag = next(sys_tags())
    wheel = tmp_path / f"compileiq_native_runtime-1.1.0-{compatible_tag}.whl"
    wheel.write_bytes(b"wheel")
    manifest = _manifest(tmp_path, wheel)
    (tmp_path / "compileiq_native_runtime-1.1.0-cp311-cp311-linux_x86_64.whl").write_bytes(
        b"other wheel"
    )
    record = materialize_native_runtime._runtime_record(manifest)

    selected = materialize_native_runtime._select_runtime_wheel(
        None,
        [tmp_path],
        manifest,
        record,
    )

    assert selected == wheel


def test_rejects_a_manifest_named_wheel_for_an_incompatible_platform(tmp_path: Path) -> None:
    """Fail before pip when the runtime matrix has no artifact for the active platform."""

    wheel = tmp_path / "compileiq_native_runtime-1.1.0-cp999-cp999-any.whl"
    wheel.write_bytes(b"wheel")
    manifest = _manifest(tmp_path, wheel)
    record = materialize_native_runtime._runtime_record(manifest)

    with pytest.raises(RuntimeError, match="not compatible"):
        materialize_native_runtime._select_runtime_wheel(None, [], manifest, record)


def test_selects_the_only_compatible_cell_from_multiple_generated_manifests(
    tmp_path: Path,
) -> None:
    """Let one beta checkout carry platform cells without relying on file ordering."""

    compatible_tag = next(sys_tags())
    compatible_wheel = tmp_path / f"compileiq_native_runtime-1.1.0-{compatible_tag}.whl"
    compatible_wheel.write_bytes(b"compatible wheel")
    compatible_manifest = _manifest(
        tmp_path,
        compatible_wheel,
        f"native-runtime-manifest-{compatible_tag}.json",
    )
    incompatible_wheel = tmp_path / "compileiq_native_runtime-1.1.0-cp999-cp999-any.whl"
    incompatible_wheel.write_bytes(b"incompatible wheel")
    incompatible_manifest = _manifest(
        tmp_path,
        incompatible_wheel,
        "native-runtime-manifest-cp999-cp999-any.json",
    )

    selected_wheel, selected_manifest, _ = materialize_native_runtime._select_runtime_artifact(
        None,
        [tmp_path],
        [incompatible_manifest, compatible_manifest],
    )

    assert selected_wheel == compatible_wheel
    assert selected_manifest == compatible_manifest


@pytest.mark.parametrize("compatible_wheel_tag", RUNTIME_MATRIX_TAGS)
def test_twelve_cell_runtime_matrix_has_exactly_one_compatible_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    compatible_wheel_tag: str,
) -> None:
    manifests = []
    wheels = {}
    for wheel_tag in RUNTIME_MATRIX_TAGS:
        wheel = tmp_path / f"compileiq_native_runtime-1.1.0.dev0-{wheel_tag}.whl"
        wheel.write_bytes(wheel_tag.encode())
        wheels[wheel_tag] = wheel
        manifests.append(
            _manifest(
                tmp_path,
                wheel,
                f"native-runtime-manifest-{wheel_tag}.json",
            )
        )
    python_tag, abi_tag, platform_tag = compatible_wheel_tag.split("-", 2)
    monkeypatch.setattr(
        materialize_native_runtime,
        "sys_tags",
        lambda: iter((Tag(python_tag, abi_tag, platform_tag),)),
    )

    selected_wheel, selected_manifest, _ = materialize_native_runtime._select_runtime_artifact(
        None,
        [tmp_path],
        manifests,
    )

    assert selected_wheel == wheels[compatible_wheel_tag]
    assert selected_manifest.name == f"native-runtime-manifest-{compatible_wheel_tag}.json"


def test_rejects_ambiguous_compatible_bridge_cells(tmp_path: Path) -> None:
    """Require explicit provenance selection if two bridge manifests match this host."""

    compatible_tag = next(sys_tags())
    manifests = []
    for version in ("1.1.0", "1.1.1"):
        wheel = tmp_path / f"compileiq_native_runtime-{version}-{compatible_tag}.whl"
        wheel.write_bytes(version.encode())
        manifests.append(
            _manifest(
                tmp_path,
                wheel,
                f"native-runtime-manifest-{version}-{compatible_tag}.json",
            )
        )

    with pytest.raises(RuntimeError, match="multiple compatible native runtime artifacts"):
        materialize_native_runtime._select_runtime_artifact(None, [tmp_path], manifests)


def test_reports_available_cells_when_none_match_the_host(tmp_path: Path) -> None:
    """Name the staged cells when clone-and-run lacks one for the active platform."""

    wheel = tmp_path / "compileiq_native_runtime-1.1.0-cp999-cp999-any.whl"
    wheel.write_bytes(b"incompatible wheel")
    manifest = _manifest(tmp_path, wheel)

    with pytest.raises(
        RuntimeError,
        match=r"no manifest-verified.*Available runtime artifacts: .*cp999",
    ):
        materialize_native_runtime._select_runtime_artifact(None, [tmp_path], [manifest])


def test_materialize_probes_amber_registration_after_install(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Treat frontend/runtime registration as part of artifact compatibility."""

    compatible_tag = next(sys_tags())
    wheel = tmp_path / f"compileiq_native_runtime-1.1.0-{compatible_tag}.whl"
    wheel.write_bytes(b"wheel payload" * 200)
    manifest = _manifest(tmp_path, wheel)
    calls: list[list[str]] = []

    def record(command, **kwargs):
        assert kwargs["check"] is True
        calls.append(command)

    monkeypatch.setattr(materialize_native_runtime.subprocess, "run", record)

    materialize_native_runtime.materialize(wheel, [manifest], [])

    assert len(calls) == 2
    assert calls[0][1:4] == ["-m", "pip", "install"]
    assert calls[1][-1] == materialize_native_runtime.REGISTRATION_PROBE


def test_materialize_rejects_a_runtime_that_cannot_register(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Surface a stale runtime/frontend API mismatch immediately after installation."""

    compatible_tag = next(sys_tags())
    wheel = tmp_path / f"compileiq_native_runtime-1.1.0-{compatible_tag}.whl"
    wheel.write_bytes(b"wheel payload" * 200)
    manifest = _manifest(tmp_path, wheel)
    attempt = 0

    def fail_probe(command, **kwargs):
        nonlocal attempt
        attempt += 1
        if attempt == 2:
            raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(materialize_native_runtime.subprocess, "run", fail_probe)

    with pytest.raises(RuntimeError, match="could not register against this CompileIQ frontend"):
        materialize_native_runtime.materialize(wheel, [manifest], [])


@pytest.mark.parametrize("missing", ["lane-manifest", "runtime-wheel"])
def test_main_reports_missing_inputs_without_a_traceback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    missing: str,
) -> None:
    """A nonexistent --lane-manifest or --runtime-wheel is an ERROR line, not a traceback."""

    compatible_tag = next(sys_tags())
    wheel = tmp_path / f"compileiq_native_runtime-1.1.0-{compatible_tag}.whl"
    wheel.write_bytes(b"wheel payload" * 200)
    manifest = _manifest(tmp_path, wheel)
    if missing == "lane-manifest":
        manifest = tmp_path / "absent-manifest.json"
    else:
        wheel = tmp_path / "absent.whl"
    monkeypatch.setattr(
        "sys.argv",
        [
            "materialize_native_runtime.py",
            "--runtime-wheel",
            str(wheel),
            "--lane-manifest",
            str(manifest),
        ],
    )

    with pytest.raises(SystemExit) as excinfo:
        materialize_native_runtime.main()

    assert excinfo.value.code == 1
    assert capsys.readouterr().err.startswith("ERROR: ")
