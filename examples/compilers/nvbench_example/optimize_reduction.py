#!/usr/bin/env python3
"""
CompileIQ NVBench Example: Optimize CUDA reduction kernel with PTXAS controls.

Uses NVBench for statistically rigorous benchmarking and CompileIQ search
over the PTXAS search space to find compiler configurations for a CUDA
reduction kernel.

Usage:
    # Run optimization for chosen architecture
    python optimize_reduction.py --arch sm_120

    # Benchmark-only (no optimization):
    python optimize_reduction.py --arch sm_120 --benchmark-only

    # With saved config:
    python optimize_reduction.py --arch sm_120 --benchmark-only \
        --nvbench-path /path/to/nvbench/install \
        --nvcc-options "-Xptxas --apply-controls=best_reduction.acf"
"""

import argparse
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

from compileiq.ciq import Search, SearchConfiguration
from compileiq.search_spaces.compilers import PtxasSearchSpace
from compileiq.types import INVALID_SCORE, ProblemType
from compileiq.utils.helpers import save_compiler_config
from nvbench_utils import parse_nvbench_result

SCRIPT_DIR = Path(__file__).parent.resolve()


# ---------------------------------------------------------------------------
# CUDA / NVBench path discovery
# ---------------------------------------------------------------------------

def get_nvcc_path() -> Path:
    """Find nvcc from CUDACXX or PATH."""
    nvcc_path = os.environ.get("CUDACXX") or shutil.which("nvcc")
    if not nvcc_path:
        raise RuntimeError("nvcc not found in PATH")
    return Path(nvcc_path)


def discover_nvbench_prefix(nvbench_path: Path | None) -> Path:
    """Return an installed NVBench CMake prefix."""
    if nvbench_path is not None:
        return Path(nvbench_path)

    env_path = os.environ.get("NVBENCH_PATH")
    if env_path:
        return Path(env_path)

    try:
        import cuda.bench as bench
    except ImportError as e:
        raise RuntimeError(
            "NVBench install prefix was not provided. Pass --nvbench-path, set "
            "NVBENCH_PATH, or install cuda-bench so cuda.bench.get_nvbench_prefix() "
            "is available."
        ) from e

    return bench.get_nvbench_prefix()


def normalize_cmake_arch(arch: str) -> str:
    """Convert nvcc-style architecture strings to CMake CUDA architectures."""
    if arch.startswith("sm_"):
        return arch.removeprefix("sm_")
    if arch.startswith("compute_"):
        return arch.removeprefix("compute_")
    return arch


# ---------------------------------------------------------------------------
# Configure, build, and run
# ---------------------------------------------------------------------------

def quote_cmd(cmd: list[str]) -> str:
    return shlex.join(str(part) for part in cmd)


def print_failed_command(cmd: list[str], *, stdout: str = "", stderr: str = "") -> None:
    print(f"Command failed: {quote_cmd(cmd)}")
    if stdout:
        print("stdout:")
        print(stdout.rstrip())
    if stderr:
        print("stderr:")
        print(stderr.rstrip())


def cmake_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("CUDACXX", str(get_nvcc_path()))
    return env


def cmake_configure(
    build_dir: Path,
    arch: str,
    nvbench_path: Path,
    controls_file: Path | None = None,
    extra_nvcc_opts: list[str] | None = None,
) -> bool:
    """Configure a CMake build tree for the benchmark."""
    build_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "cmake",
        "-S", str(SCRIPT_DIR),
        "-B", str(build_dir),
        "-G", "Ninja",
        f"-DCMAKE_PREFIX_PATH={nvbench_path}",
        f"-DCMAKE_CUDA_ARCHITECTURES={normalize_cmake_arch(arch)}",
    ]
    if controls_file is not None:
        cmd.append(f"-DCOMPILEIQ_PTXAS_CONTROLS={controls_file}")
    if extra_nvcc_opts:
        cmd.append(f"-DCOMPILEIQ_EXTRA_NVCC_OPTIONS={shlex.join(extra_nvcc_opts)}")

    try:
        subprocess.run(
            cmd,
            env=cmake_env(),
            capture_output=True,
            text=True,
            check=True,
            timeout=120,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        stdout = getattr(e, "stdout", "") or ""
        stderr = getattr(e, "stderr", "")
        print_failed_command(cmd, stdout=stdout, stderr=stderr)
        return False

    return True


def cmake_build(build_dir: Path) -> Path | None:
    """Build the configured benchmark target."""
    exe = build_dir / "reduction_bench"

    try:
        cmd = ["cmake", "--build", str(build_dir), "--target", "reduction_bench"]
        subprocess.run(
            cmd,
            env=cmake_env(),
            capture_output=True,
            text=True,
            check=True,
            timeout=120,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        stdout = getattr(e, "stdout", "") or ""
        stderr = getattr(e, "stderr", "") or ""
        print_failed_command(cmd, stdout=stdout, stderr=stderr)
        return None

    return exe


def run_nvbench(
    exe_path: Path, elements_pow2: int, tmpdir: str, timeout: int = 360,
) -> float | None:
    """Run the NVBench benchmark and return P75 latency (seconds).

    Returns None on failure or timeout.
    """
    result_path = os.path.join(tmpdir, "result.json")
    cmd = [
        str(exe_path), "-d", "0", "-b", "reduction",
        "-a", f"Elements[pow2]={elements_pow2}",
        "--no-batch", "--stopping-criterion", "entropy",
        "--jsonbin", result_path,
    ]

    try:
        p = subprocess.Popen(
            cmd, start_new_session=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        stdout, stderr = p.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(p.pid), signal.SIGTERM)
        stdout, stderr = p.communicate()
        print("NVBench benchmark timed out")
        print_failed_command(cmd, stdout=stdout or "", stderr=stderr or "")
        return None

    if p.returncode != 0:
        print(f"NVBench benchmark failed with exit code {p.returncode}")
        print_failed_command(cmd, stdout=stdout or "", stderr=stderr or "")
        return None

    score = parse_nvbench_result(Path(result_path))
    if score is None:
        print("NVBench result parsing failed")
        print(f"Result file: {result_path}")
        print_failed_command(cmd, stdout=stdout or "", stderr=stderr or "")
    return score


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

def run_benchmark_only(args) -> int:
    """Benchmark-only mode: build with optional nvcc options and report timing."""
    extra_opts = shlex.split(args.nvcc_options) if args.nvcc_options else None
    nvbench_path = discover_nvbench_prefix(args.nvbench_path)

    with tempfile.TemporaryDirectory(prefix="ciq_nvbench_") as tmpdir:
        build_dir = Path(tmpdir) / "build"
        if not cmake_configure(
            build_dir,
            args.arch,
            nvbench_path,
            extra_nvcc_opts=extra_opts,
        ):
            print("Configure failed")
            return 1

        exe = cmake_build(build_dir)
        if exe is None:
            print("Build failed")
            return 1

        score = run_nvbench(exe, args.elements_pow2, tmpdir)
        if score is None:
            print("Benchmark failed")
            return 1

        print(f"P75 latency: {score * 1000:.4f} ms  ({score:.6f} s)")
        return 0


def run_optimization(args, cuda_version: str):
    """Run optimization to find the best PTXAS compiler config."""
    nvbench_path = discover_nvbench_prefix(args.nvbench_path)

    with tempfile.TemporaryDirectory(prefix="ciq_nvbench_") as tmpdir:
        tmp_path = Path(tmpdir)
        baseline_build_dir = tmp_path / "baseline_build"
        search_build_dir = tmp_path / "search_build"
        controls_path = search_build_dir / "controls.acf"

        # Run baseline (no compiler controls)
        print("Running baseline...")
        if not cmake_configure(baseline_build_dir, args.arch, nvbench_path):
            print("Baseline configure failed")
            return 1
        baseline_exe = cmake_build(baseline_build_dir)
        if baseline_exe is None:
            print("Baseline build failed")
            return 1
        baseline = run_nvbench(baseline_exe, args.elements_pow2, tmpdir)

        if baseline is None:
            print("Baseline benchmark failed")
            return 1
        print(f"Baseline P75: {baseline * 1000:.4f} ms\n")

        # Configure the candidate build once. The controls file is declared as
        # an object dependency in CMakeLists.txt, so rewriting it is enough to
        # trigger recompilation with Ninja on each candidate.
        search_build_dir.mkdir(parents=True, exist_ok=True)
        controls_path.write_text("", encoding="utf-8")
        if not cmake_configure(
            search_build_dir, args.arch, nvbench_path, controls_file=controls_path
        ):
            print("Search configure failed")
            return 1

        # Objective function: compile with PTXAS config, measure with NVBench
        def objective(config_str: str) -> float:
            save_compiler_config(str(controls_path), config_str)
            controls_path.touch()

            exe = cmake_build(search_build_dir)
            if exe is None:
                return INVALID_SCORE

            score = run_nvbench(exe, args.elements_pow2, tmpdir)
            if score is None:
                return INVALID_SCORE

            return score

        # Configure and run search
        search_space = args.search_space or PtxasSearchSpace(version=cuda_version)
        config = SearchConfiguration(
            problem_type=ProblemType.MIN,
            generations=args.generations,
            pool_size=args.pool_size,
        )
        tuner = Search(
            objective_function=objective,
            search_space=search_space,
            search_config=config,
            dump_results=SCRIPT_DIR / "optimization_results.csv",
        )

        print(f"Starting optimization ({args.generations} generations, pool={args.pool_size})...")
        print("Using PtxasSearchSpace with NVBench measurement (P75 latency)\n")
        results = tuner.start(num_workers=1)
        best = results.get_best_result()

        # Report results
        if best:
            best_time = best.get("score_1", best.get("score"))
            speedup = baseline / best_time if best_time > 0 else 0

            print(f"\nBaseline:  {baseline * 1000:.4f} ms")
            print(f"Optimized: {best_time * 1000:.4f} ms")
            print(f"Speedup:   {speedup:.2f}x")

            # Save best config
            config_path = SCRIPT_DIR / "best_reduction.acf"
            save_compiler_config(str(config_path), best["params"])
            print(f"\nConfig saved: {config_path}")
            print(
                "Usage: cmake -S . -B build -G Ninja "
                f"-DCOMPILEIQ_PTXAS_CONTROLS={config_path} ..."
            )

        return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="CompileIQ NVBench optimization with PTXAS controls"
    )
    parser.add_argument("--arch", default="sm_100",
                        help="GPU architecture (default: sm_100)")
    parser.add_argument("--nvbench-path", type=Path,
                        default=os.environ.get("NVBENCH_PATH"),
                        help="NVBench install directory (or set NVBENCH_PATH env var)")
    parser.add_argument("--elements-pow2", type=int, default=26,
                        help="Problem size as power of 2 (default: 26, i.e. 2^26)")

    # Optimization args
    parser.add_argument("--generations", type=int, default=10,
                        help="Search generations (default: 10)")
    parser.add_argument("--pool-size", type=int, default=15,
                        help="Population size (default: 15)")
    parser.add_argument("--search-space", type=Path, default=None,
                        help="Local search space file (skip auto-download)")

    # Benchmark-only args
    parser.add_argument("--benchmark-only", action="store_true",
                        help="Skip optimization, just benchmark")
    parser.add_argument("--nvcc-options", default="",
                        help="Additional NVCC options (benchmark-only mode)")
    args = parser.parse_args()

    # Check CUDA version
    version_output = subprocess.run(
        [str(get_nvcc_path()), "--version"], capture_output=True, text=True, check=True
    ).stdout
    cuda_version_match = re.search(r"release (\d+\.\d+),", version_output)
    if cuda_version_match is None:
        parser.error("Could not determine CUDA version from nvcc --version output")
    cuda_version = cuda_version_match.group(1)
    assert float(cuda_version) >= 13.3, "CompileIQ requires CUDA 13.3+"

    if args.benchmark_only:
        return run_benchmark_only(args)
    return run_optimization(args, cuda_version)


if __name__ == "__main__":
    sys.exit(main())
