from __future__ import annotations

import re
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PYTHON_VERSIONS = ("3.11", "3.12", "3.13", "3.14")
RAY_PLATFORMS = (
    "manylinux2014_x86_64",
    "manylinux2014_aarch64",
    "macosx_12_0_arm64",
    "win_amd64",
)


def _job(workflow: Path, name: str) -> str:
    text = workflow.read_text()
    match = re.search(rf"(?ms)^  {re.escape(name)}:\n(.*?)(?=^  [a-zA-Z0-9_-]+:|\Z)", text)
    assert match is not None, f"missing {name} in {workflow}"
    return match.group(1)


def test_beta_metadata_and_ray_lock_cover_the_release_matrix() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert project["tool"]["poetry"]["version"] == "1.1.0b2"
    assert project["tool"]["poetry"]["dependencies"]["python"] == ">=3.11,<3.15"
    assert project["tool"]["poetry"]["dependencies"]["ray"] == {
        "extras": ["default"],
        "version": "^2.58.0",
    }

    lock = tomllib.loads((ROOT / "poetry.lock").read_text())
    ray = next(package for package in lock["package"] if package["name"] == "ray")
    assert ray["version"] == "2.58.0"
    filenames = {record["file"] for record in ray["files"]}
    expected = {
        f"ray-2.58.0-cp{minor}-cp{minor}-{platform}.whl"
        for minor in ("311", "312", "313", "314")
        for platform in RAY_PLATFORMS
    }
    assert expected <= filenames


def test_all_supported_windows_pythons_are_present_in_test_matrices() -> None:
    ci = ROOT / ".github/workflows/ci.yml"
    nightly = ROOT / ".github/workflows/nightly.yml"
    for workflow, job_name in (
        (ci, "unit-test"),
        (ci, "integration-test"),
        (nightly, "fuzz-test-deep"),
    ):
        job = _job(workflow, job_name)
        assert "os: [ubuntu-latest, windows-latest]" in job
        assert 'python: ["3.11", "3.12", "3.13", "3.14"]' in job
        assert "exclude:" not in job


def test_known_beta_flakes_remain_explicitly_tracked() -> None:
    ci_job = _job(ROOT / ".github/workflows/ci.yml", "integration-test")
    nightly_job = _job(ROOT / ".github/workflows/nightly.yml", "fuzz-test-deep")
    assert "https://github.com/NVIDIA/CompileIQ/issues/17" in ci_job
    assert "known_broken: true" in ci_job
    assert "continue-on-error: ${{ matrix.known_broken || false }}" in ci_job
    assert "https://github.com/NVIDIA/CompileIQ/issues/15" in nightly_job
    assert "known_broken: true" in nightly_job
    assert "continue-on-error: ${{ matrix.known_broken || false }}" in nightly_job


def test_complete_wheel_acceptance_forwards_the_exact_expected_version(
    tmp_path, monkeypatch
) -> None:
    from dev import test_complete_wheel_install

    wheel = tmp_path / "compileiq-1.1.0b2-cp314-cp314-macosx_15_0_arm64.whl"
    wheel.write_bytes(b"test wheel")
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))

    monkeypatch.setattr(test_complete_wheel_install.subprocess, "run", fake_run)
    test_complete_wheel_install.test_install(
        wheel,
        requested_venv=tmp_path / "venv",
        expected_version="1.1.0b2",
    )

    assert len(calls) == 4
    assert all(
        kwargs["env"]["CIQ_EXPECTED_VERSION"] == "1.1.0b2"
        for _, kwargs in calls
    )
    assert 'metadata.version("compileiq")' in test_complete_wheel_install.ACCEPTANCE
    assert "assert installed_version == expected_version" in (
        test_complete_wheel_install.ACCEPTANCE
    )


def test_release_upload_publishes_only_accepted_complete_wheels() -> None:
    ci = ROOT / ".github/workflows/ci.yml"
    upload = _job(ci, "upload-github-release-wheels")
    assert "startsWith(github.ref, 'refs/tags/v')" in upload
    assert "needs: [ci-pass, verify-and-test-complete-wheel]" in upload
    assert "pattern: compileiq-complete-wheel-*" in upload
    assert "draft: true" in upload

    accept = _job(ci, "verify-and-test-complete-wheel")
    assert "name: compileiq-complete-wheel-${{ matrix.wheel_tag }}" in accept
    assert accept.index("dev/verify_complete_wheel.py") < accept.index(
        "dev/test_complete_wheel_install.py"
    ) < accept.index("Upload accepted complete wheel")
    # Unverified composed wheels must never match the release download pattern.
    compose = _job(ci, "compose-complete-wheels")
    assert "name: compileiq-composed-wheels-unverified" in compose

    matrix = _job(ci, "complete-wheel-matrix")
    for runner in ("ubuntu-latest", "ubuntu-24.04-arm", "windows-latest", "macos-15"):
        assert f'"os": "{runner}"' in matrix
    assert 'for python in ("3.11", "3.12", "3.13", "3.14")' in matrix


def test_ci_pass_fails_closed_over_every_job() -> None:
    ci = ROOT / ".github/workflows/ci.yml"
    text = ci.read_text()
    jobs = set(re.findall(r"(?m)^  ([a-zA-Z0-9_-]+):\n", text.split("\njobs:\n", 1)[1]))
    gate = _job(ci, "ci-pass")
    assert "name: CI Pass" in gate  # the repository rules require this check name
    assert "if: always()" in gate
    needs = set(re.findall(r"(?m)^      - ([a-zA-Z0-9_-]+)$", gate.split("steps:", 1)[0]))
    assert needs == jobs - {"ci-pass", "upload-github-release-wheels"}
    # The only job allowed to be skipped is per-platform acceptance, and only off
    # main and release tags, where compose-complete-wheels accepts the Linux wheels.
    assert (
        "ALLOWED_SKIPS: ${{ needs.complete-wheel-matrix.outputs.full != 'true' "
        "&& 'verify-and-test-complete-wheel' || '' }}"
    ) in gate

    accept = _job(ci, "verify-and-test-complete-wheel")
    assert "if: needs.complete-wheel-matrix.outputs.full == 'true'" in accept
    compose = _job(ci, "compose-complete-wheels")
    assert "if: needs.complete-wheel-matrix.outputs.full != 'true'" in compose
    assert compose.index("dev/verify_complete_wheel.py") < compose.index(
        "dev/test_complete_wheel_install.py"
    )
