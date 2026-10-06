#!/usr/bin/env python3
"""Install one complete CompileIQ wheel into a clean environment and accept it."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path


ACCEPTANCE = r"""
import importlib.util
import tempfile
import os
from importlib import metadata
from pathlib import Path

import compileiq
import compileiq.search_spaces.base as ss
import compileiq_native_runtime
import pymoo

from compileiq.backends import BackendNotFoundError, get_backend_registration
from compileiq.ciq import Search
from compileiq.types import SearchConfiguration, Worker
from compileiq.utils.validation import Score


class InlineWorker(Worker):
    @classmethod
    def create(cls, cache_folder, normalize, tracker):
        return cls(cache_folder=cache_folder, normalize=normalize, tracker=tracker)

    def run(
        self,
        *,
        function,
        params_pool,
        params_ids,
        num_function_returns=1,
        **kwargs,
    ):
        del kwargs
        return [
            Score(
                score=function(params),
                params=params,
                param_id=param_id,
                num_objectives=num_function_returns,
            )
            for params, param_id in zip(params_pool, params_ids, strict=True)
        ]


def scores(candidates):
    return [
        Score(
            score=float(index),
            params=candidate.params,
            param_id=candidate.param_id,
            num_objectives=1,
        )
        for index, candidate in enumerate(candidates)
    ]

installed_version = metadata.version("compileiq")
expected_version = os.environ.get("CIQ_EXPECTED_VERSION")
if expected_version:
    assert installed_version == expected_version, (installed_version, expected_version)
else:
    assert installed_version
try:
    metadata.version("compileiq-native-runtime")
except metadata.PackageNotFoundError:
    pass
else:
    raise AssertionError("runtime must be embedded, not installed as a second distribution")

registration = get_backend_registration("amber")
assert registration.capabilities.checkpoint
assert registration.capabilities.sealed_checkpoint
assert registration.artifact.backend_version == compileiq_native_runtime.__version__

with tempfile.TemporaryDirectory(prefix="ciq-complete-wheel-semantic-") as directory:
    root = Path(directory)
    search_space = {"x": ss.range(0, 100)}
    # The default engine, with no engine= argument, is the native runtime.
    search = Search(
        objective_function=lambda params: float(params["x"]),
        search_space=search_space,
        search_config=SearchConfiguration(generations=2, pool_size=6),
        worker_type=InlineWorker,
        cache_folder=root / "initial",
        disable_progress_bar=True,
    )
    assert search.provider == "amber", search.provider
    first = search.ask()
    assert len(first) == 6
    search.tell(scores(first))
    checkpoint = search.save_checkpoint(
        root / "runtime.checkpoint",
        require_no_pending=True,
    )

    restored = Search.from_checkpoint(
        checkpoint,
        objective_function=lambda params: float(params["x"]),
        search_space=search_space,
        worker_type=InlineWorker,
        cache_folder=root / "restored",
        disable_progress_bar=True,
    )
    assert len(restored.result.get_results()) == 6
    second = restored.ask()
    assert len(second) == 6
    assert {str(candidate.param_id) for candidate in first}.isdisjoint(
        str(candidate.param_id) for candidate in second
    )
    restored.tell(scores(second))
    assert len(restored.result.get_results()) == 12

print("complete CompileIQ wheel clean-install acceptance passed")
"""


def _venv_python(venv: Path) -> Path:
    if os.name == "nt":
        return venv / "Scripts" / "python.exe"
    return venv / "bin" / "python"


@contextmanager
def _environment(requested: Path | None):
    if requested is not None:
        requested = requested.resolve()
        requested.parent.mkdir(parents=True, exist_ok=True)
        yield requested
        return
    with tempfile.TemporaryDirectory(prefix="ciq-complete-wheel-") as directory:
        yield Path(directory) / "venv"


def test_install(
    wheel: Path,
    requested_venv: Path | None = None,
    expected_version: str | None = None,
) -> None:
    """Create a clean venv and install only the complete public distribution."""

    wheel = wheel.resolve()
    if not wheel.is_file():
        raise RuntimeError(f"complete wheel does not exist: {wheel}")
    with _environment(requested_venv) as venv:
        environment = os.environ.copy()
        environment.pop("PYTHONHOME", None)
        environment.pop("PYTHONPATH", None)
        environment["PYTHONNOUSERSITE"] = "1"
        environment["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
        if expected_version:
            environment["CIQ_EXPECTED_VERSION"] = expected_version
        else:
            environment.pop("CIQ_EXPECTED_VERSION", None)
        workdir = venv.parent / "work"
        workdir.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [sys.executable, "-m", "venv", "--clear", str(venv)],
            cwd=workdir,
            env=environment,
            check=True,
        )
        python = _venv_python(venv)
        subprocess.run(
            [
                str(python),
                "-m",
                "pip",
                "install",
                "--quiet",
                "--no-cache-dir",
                str(wheel),
            ],
            cwd=workdir,
            env=environment,
            check=True,
        )
        subprocess.run(
            [str(python), "-m", "pip", "check"],
            cwd=workdir,
            env=environment,
            check=True,
        )
        subprocess.run(
            [str(python), "-c", ACCEPTANCE],
            cwd=workdir,
            env=environment,
            check=True,
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wheel", required=True, type=Path)
    parser.add_argument("--venv", type=Path)
    parser.add_argument("--expected-version")
    args = parser.parse_args()
    test_install(args.wheel, args.venv, args.expected_version)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
