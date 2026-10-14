from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from dev.check_public_artifacts import (
    EXPECTED_RUNTIME_TAGS,
    MARKERS_ENVIRONMENT_VARIABLE,
    PrivateMarkers,
    check,
    load_private_markers,
    main,
)


ROOT = Path(__file__).resolve().parents[2]


def _write_runtime_matrix(root: Path) -> Path:
    runtime = root / "native-runtime-wheels"
    runtime.mkdir()
    (runtime / "README.md").write_text("Public compiled runtime artifacts.\n")
    for index, wheel_tag in enumerate(sorted(EXPECTED_RUNTIME_TAGS)):
        wheel = runtime / f"compileiq_native_runtime-1.1.0.dev0-{wheel_tag}.whl"
        extension = "_native.pyd" if "win_amd64" in wheel_tag else "_native.so"
        with zipfile.ZipFile(wheel, "w") as archive:
            archive.writestr("compileiq_native_runtime/__init__.py", "")
            archive.writestr(f"compileiq_native_runtime/{extension}", b"compiled payload")
        python_tag, abi_tag, platform_tag = wheel_tag.split("-", 2)
        manifest = {
            "schema_version": 1,
            "provenance": {
                "pipeline_id": 700,
                "job_id": 800 + index,
                "source_commit": "1" * 40,
            },
            "runtime": {
                "python_tag": python_tag,
                "abi_tag": abi_tag,
                "platform_tag": platform_tag,
                "wheel_tag": wheel_tag,
            },
            "outputs": {
                "native_runtime": {
                    "filename": wheel.name,
                    "sha256": f"sha256:{hashlib.sha256(wheel.read_bytes()).hexdigest()}",
                    "size": wheel.stat().st_size,
                }
            },
        }
        (runtime / f"native-runtime-manifest-{wheel_tag}.json").write_text(json.dumps(manifest))
    return runtime


def test_public_artifact_guard_accepts_exact_public_runtime_matrix(tmp_path: Path) -> None:
    _write_runtime_matrix(tmp_path)

    check(tmp_path)


def test_public_artifact_guard_defines_exact_sixteen_cell_cross_product() -> None:
    assert len(EXPECTED_RUNTIME_TAGS) == 16
    assert EXPECTED_RUNTIME_TAGS == {
        f"{python_tag}-{python_tag}-{platform_tag}"
        for python_tag in ("cp311", "cp312", "cp313", "cp314")
        for platform_tag in (
            "linux_x86_64",
            "linux_aarch64",
            "win_amd64",
            "macosx_15_0_arm64",
        )
    }


def test_public_artifact_guard_rejects_private_identifiers_in_docs(tmp_path: Path) -> None:
    _write_runtime_matrix(tmp_path)
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "development.md").write_text("Build this from Example-Private-Repo.")
    markers = PrivateMarkers(public_text=("example-private-repo",))

    check(tmp_path)
    with pytest.raises(RuntimeError, match="private implementation identifiers"):
        check(tmp_path, markers)


def test_public_artifact_guard_rejects_private_binary_provenance(tmp_path: Path) -> None:
    runtime = _write_runtime_matrix(tmp_path)
    manifest_path = next(runtime.glob("native-runtime-manifest-cp311-cp311-win_amd64.json"))
    manifest = json.loads(manifest_path.read_text())
    wheel = runtime / manifest["outputs"]["native_runtime"]["filename"]
    with zipfile.ZipFile(wheel, "a") as archive:
        archive.writestr("compileiq_native_runtime/private.txt", "example-private-toolchain")
    manifest["outputs"]["native_runtime"].update(
        sha256=f"sha256:{hashlib.sha256(wheel.read_bytes()).hexdigest()}",
        size=wheel.stat().st_size,
    )
    manifest_path.write_text(json.dumps(manifest))

    markers = PrivateMarkers(artifact_bytes=("Example-Private-Toolchain",))

    check(tmp_path)
    with pytest.raises(RuntimeError, match="private markers.*example-private-toolchain"):
        check(tmp_path, markers)


def test_public_artifact_guard_rejects_build_machine_paths_without_markers(
    tmp_path: Path,
) -> None:
    runtime = _write_runtime_matrix(tmp_path)
    manifest_path = next(runtime.glob("native-runtime-manifest-cp311-cp311-linux_x86_64.json"))
    manifest = json.loads(manifest_path.read_text())
    wheel = runtime / manifest["outputs"]["native_runtime"]["filename"]
    with zipfile.ZipFile(wheel, "a") as archive:
        archive.writestr("compileiq_native_runtime/build.txt", "/home/builder/src/module.c")
    manifest["outputs"]["native_runtime"].update(
        sha256=f"sha256:{hashlib.sha256(wheel.read_bytes()).hexdigest()}",
        size=wheel.stat().st_size,
    )
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(RuntimeError, match="private markers.*/home/"):
        check(tmp_path)


def test_private_markers_load_from_json_file(tmp_path: Path) -> None:
    path = tmp_path / "markers.json"
    path.write_text(json.dumps({"public_text": ["a"], "artifact_bytes": ["b", "c"]}))

    assert load_private_markers(path) == PrivateMarkers(
        public_text=("a",), artifact_bytes=("b", "c")
    )


@pytest.mark.parametrize(
    "document",
    [[], {"public_text": "a"}, {"public_text": [""]}, {"unknown": []}],
)
def test_private_markers_reject_malformed_files(tmp_path: Path, document: object) -> None:
    path = tmp_path / "markers.json"
    path.write_text(json.dumps(document))

    with pytest.raises(RuntimeError, match="private marker file"):
        load_private_markers(path)


def test_guard_command_reads_markers_from_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_runtime_matrix(tmp_path)
    (tmp_path / "README.md").write_text("Mentions example-private-repo.")
    markers = tmp_path / "markers.json"
    markers.write_text(json.dumps({"public_text": ["example-private-repo"]}))

    monkeypatch.delenv(MARKERS_ENVIRONMENT_VARIABLE, raising=False)
    assert main(["--repository", str(tmp_path)]) == 0
    assert "generic checks only" in capsys.readouterr().out

    monkeypatch.setenv(MARKERS_ENVIRONMENT_VARIABLE, str(markers))
    with pytest.raises(SystemExit) as exited:
        main(["--repository", str(tmp_path)])
    assert exited.value.code == 1
    assert "README.md: example-private-repo" in capsys.readouterr().err


def test_public_artifact_guard_rejects_unexpected_private_provenance(
    tmp_path: Path,
) -> None:
    runtime = _write_runtime_matrix(tmp_path)
    manifest_path = next(runtime.glob("native-runtime-manifest-*.json"))
    manifest = json.loads(manifest_path.read_text())
    manifest["pipeline_id"] = "123"
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(RuntimeError, match="has keys"):
        check(tmp_path)


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("pipeline_id", 0, "invalid producer pipeline_id"),
        ("job_id", "123", "invalid producer job_id"),
        ("source_commit", "local-build", "invalid producer source commit"),
    ],
)
def test_public_artifact_guard_rejects_invalid_opaque_provenance(
    tmp_path: Path, key: str, value: object, message: str
) -> None:
    runtime = _write_runtime_matrix(tmp_path)
    manifest_path = next(runtime.glob("native-runtime-manifest-*.json"))
    manifest = json.loads(manifest_path.read_text())
    manifest["provenance"][key] = value
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(RuntimeError, match=message):
        check(tmp_path)


def test_public_artifact_guard_rejects_mixed_producer_coordinates(tmp_path: Path) -> None:
    runtime = _write_runtime_matrix(tmp_path)
    manifest_path = next(runtime.glob("native-runtime-manifest-*.json"))
    manifest = json.loads(manifest_path.read_text())
    manifest["provenance"]["pipeline_id"] = 701
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(RuntimeError, match="mixed producer coordinates"):
        check(tmp_path)


def test_public_artifact_guard_rejects_duplicate_producer_jobs(tmp_path: Path) -> None:
    runtime = _write_runtime_matrix(tmp_path)
    manifest_paths = sorted(runtime.glob("native-runtime-manifest-*.json"))
    first = json.loads(manifest_paths[0].read_text())
    second = json.loads(manifest_paths[1].read_text())
    second["provenance"]["job_id"] = first["provenance"]["job_id"]
    manifest_paths[1].write_text(json.dumps(second))

    with pytest.raises(RuntimeError, match="duplicate producer job IDs"):
        check(tmp_path)


def test_public_workflow_fetches_lfs_and_gates_ci_pass() -> None:
    workflow = (ROOT / ".github/workflows/ci.yml").read_text()

    assert "public-artifact-guard:" in workflow
    guard = workflow.split("  public-artifact-guard:", maxsplit=1)[1].split(
        "\n  lint:", maxsplit=1
    )[0]
    assert "lfs: true" in guard
    assert "python dev/check_public_artifacts.py" in guard
    ci_pass = workflow.split("  ci-pass:", maxsplit=1)[1]
    assert "public-artifact-guard" in ci_pass.split("steps:", maxsplit=1)[0]


def test_public_runtime_wheels_are_lfs_tracked_without_capturing_manifests() -> None:
    attributes = (ROOT / ".gitattributes").read_text()

    assert ("native-runtime-wheels/*.whl " "filter=lfs diff=lfs merge=lfs -text") in attributes
    assert "native-runtime-manifest-*.json filter=lfs" not in attributes


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("platform_tag", "bogus_platform", "does not match its python/abi/platform tags"),
        ("python_tag", 311, "runtime tags must all be strings"),
    ],
)
def test_public_artifact_guard_rejects_inconsistent_runtime_tags(
    tmp_path: Path, key: str, value: object, message: str
) -> None:
    runtime = _write_runtime_matrix(tmp_path)
    manifest_path = next(runtime.glob("native-runtime-manifest-*.json"))
    manifest = json.loads(manifest_path.read_text())
    manifest["runtime"][key] = value
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(RuntimeError, match=message):
        check(tmp_path)
