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
