#!/usr/bin/env python3
"""
CompileIQ NVCC Example: Optimize CUDA reduction kernel.

Uses evolutionary search to find optimal NVCC compiler configurations
via the --apply-controls feature.

Usage:
    python optimize_reduction.py [--arch sm_100] [--generations 10]

    # Benchmark-only (no optimization):
    python optimize_reduction.py --benchmark-only
    python optimize_reduction.py --benchmark-only --nvcc-options "--apply-controls config.bin"
"""

import argparse
import os
import re
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path
from statistics import mean, stdev

from compileiq.ciq import Search, SearchConfiguration
from compileiq.search_spaces.compilers import NvccSearchSpace
from compileiq.types import INVALID_SCORE, ProblemType

SCRIPT_DIR = Path(__file__).parent.resolve()
REDUCTION_CU = SCRIPT_DIR / "reduction.cu"


def get_cuda_paths():
    """Find CUDA installation from nvcc in PATH."""
    nvcc_path = shutil.which("nvcc")
    if not nvcc_path:
        raise RuntimeError("nvcc not found in PATH")
    cuda_root = Path(nvcc_path).parent.parent
    cuda_lib = cuda_root / "lib64"
    if not cuda_lib.exists():
        cuda_lib = cuda_root / "lib"
    return cuda_root, cuda_lib


def get_include_flags(cuda_root: Path) -> list:
    """Find CCCL include paths needed for cooperative_groups."""
    flags = []
    for candidate in [cuda_root / "include", cuda_root / "include" / "cccl",
                      cuda_root.parent / "include", cuda_root.parent / "include" / "cccl"]:
        if candidate.exists():
            flags.append(f"-I{candidate}")
    return flags


def build_and_run(
    arch: str, config_file: Path = None, extra_nvcc_opts: list = None, num_runs: int = 3,
) -> dict:
    """Build reduction kernel and return benchmark results.

    Returns dict with 'success', 'times_ms', 'mean_ms', 'std_ms' keys.
    """
    cuda_root, cuda_lib = get_cuda_paths()
    nvcc = str(cuda_root / "bin" / "nvcc")

    with tempfile.TemporaryDirectory(prefix="ciq_") as tmpdir:
        exe = Path(tmpdir) / "reduction"

        flags = [
            f"-arch={arch}",
            *get_include_flags(cuda_root),
            "-O3", "-std=c++17", "-DNDEBUG",
        ]
        if config_file:
            flags += ["--apply-controls", str(config_file)]
        if extra_nvcc_opts:
            flags += extra_nvcc_opts

        env = os.environ.copy()
        env["LIBRARY_PATH"] = f"{cuda_lib}:{env.get('LIBRARY_PATH', '')}"
        env["LD_LIBRARY_PATH"] = f"{cuda_lib}:{env.get('LD_LIBRARY_PATH', '')}"

        try:
            subprocess.run(
                [nvcc, *flags, str(REDUCTION_CU), "-o", str(exe)],
                env=env, capture_output=True, check=True, timeout=120,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return {"success": False, "error": "compilation_failed"}

        times = []
        for _ in range(num_runs):
            try:
                result = subprocess.run(
                    [str(exe), "-n=67108864"],
                    env=env, capture_output=True, text=True, timeout=60,
                )
                output = result.stdout + result.stderr
                if "Test passed" not in output:
                    return {"success": False, "error": "test_failed"}
                match = re.search(r'Time\s*=\s*([\d.]+)\s*s', output)
                if match:
                    times.append(float(match.group(1)) * 1000.0)
                else:
                    return {"success": False, "error": "parse_failed"}
            except Exception:
                return {"success": False, "error": "runtime_error"}

        return {
            "success": True,
            "times_ms": times,
            "mean_ms": mean(times),
            "std_ms": stdev(times) if len(times) > 1 else 0.0,
        }


def run_benchmark(args):
    """Benchmark-only mode: build with optional nvcc options and report timing."""
    extra_opts = shlex.split(args.nvcc_options) if args.nvcc_options else None
    result = build_and_run(args.arch, extra_nvcc_opts=extra_opts, num_runs=args.runs)

    if result["success"]:
        print(f"Mean: {result['mean_ms']:.3f} ms (+/- {result['std_ms']:.3f})")
        print(f"Runs: {[f'{t:.3f}' for t in result['times_ms']]}")
    else:
        print(f"Failed: {result.get('error')}")
        exit(1)


def run_optimization(args, cuda_version: str):
    """Run evolutionary optimization to find best compiler config."""
    # Run baseline
    print("Running baseline...")
    baseline_result = build_and_run(args.arch, num_runs=args.runs)
    if not baseline_result["success"]:
        print(f"Baseline failed: {baseline_result.get('error')}")
        return 1
    baseline = baseline_result["mean_ms"]
    print(f"Baseline: {baseline:.3f} ms\n")

    # Create objective that writes config to temp file
    def objective(config_blob: str) -> float:
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            f.write(bytes.fromhex(config_blob))
            config_path = Path(f.name)
        try:
            result = build_and_run(args.arch, config_path, num_runs=args.runs)
            return result["mean_ms"] if result["success"] else INVALID_SCORE
        finally:
            config_path.unlink(missing_ok=True)

    # Configure and run search
    search_space = args.search_space if args.search_space else NvccSearchSpace(version=cuda_version)
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
    results = tuner.start(num_workers=1)
    best = results.get_best_result()

    # Report results
    if best:
        best_time = best.get("score_1", best.get("score"))
        speedup = baseline / best_time if best_time > 0 else 0

        print(f"\nBaseline:  {baseline:.3f} ms")
        print(f"Optimized: {best_time:.3f} ms")
        print(f"Speedup:   {speedup:.2f}x")

        # Save best config
        config_path = SCRIPT_DIR / "reduction_best_config.bin"
        config_path.write_bytes(bytes.fromhex(best["params"]))
        print(f"\nConfig saved: {config_path}")
        print(f"Usage: nvcc --apply-controls {config_path} -arch={args.arch} ...")


def main():
    parser = argparse.ArgumentParser(description="CompileIQ NVCC optimization")
    parser.add_argument("--arch", default="sm_100", help="GPU architecture (default: sm_100)")
    parser.add_argument("--runs", type=int, default=3, help="Number of benchmark runs")

    # Optimization args
    parser.add_argument("--generations", type=int, default=10, help="Search generations")
    parser.add_argument("--pool-size", type=int, default=15, help="Population size")
    parser.add_argument("--search-space", type=Path, default=None, help="Custom search space")

    # Benchmark-only args
    parser.add_argument("--benchmark-only", action="store_true",
                        help="Skip optimization, just benchmark with optional --nvcc-options")
    parser.add_argument("--nvcc-options", default="",
                        help="Additional NVCC options (benchmark-only mode)")
    args = parser.parse_args()

    # Check CUDA version
    version_output = subprocess.run(
        ["nvcc", "--version"], capture_output=True, text=True, check=True
    ).stdout
    cuda_version = re.search(r"release (\d+\.\d+),", version_output).group(1)
    assert float(cuda_version) >= 13.3, "CompileIQ requires CUDA 13.3+"

    if args.benchmark_only:
        run_benchmark(args)
    else:
        run_optimization(args, cuda_version)


if __name__ == "__main__":
    main()
