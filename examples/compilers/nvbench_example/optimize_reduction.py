#!/usr/bin/env python3
"""
CompileIQ NVBench Example: Optimize CUDA reduction kernel with PTXAS controls.

Uses NVBench for statistically rigorous benchmarking and CompileIQ search
over the PTXAS search space to find compiler configurations for a CUDA
reduction kernel.

Usage:
    python optimize_reduction.py --nvbench-path /path/to/nvbench/install

    # Benchmark-only (no optimization):
    python optimize_reduction.py --nvbench-path /path/to/nvbench/install --benchmark-only

    # With saved config:
    python optimize_reduction.py --nvbench-path /path/to/nvbench/install \
        --benchmark-only --nvcc-options "--apply-controls best_reduction.acf"
"""

import argparse
import os
import re
import shlex
import shutil
import signal
import subprocess
import tempfile
from pathlib import Path

from compileiq.ciq import Search, SearchConfiguration
from compileiq.search_spaces.compilers import PtxasSearchSpace
from compileiq.types import INVALID_SCORE, ProblemType
from compileiq.utils.helpers import save_compiler_config
from nvbench_utils import parse_nvbench_result

SCRIPT_DIR = Path(__file__).parent.resolve()
REDUCTION_CU = SCRIPT_DIR / "reduction_bench.cu"


# ---------------------------------------------------------------------------
# CUDA / NVBench path discovery
# ---------------------------------------------------------------------------

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


def get_nvbench_paths(nvbench_path: Path):
    """Locate NVBench include, lib, and main.cu.o from install directory.

    Returns (include_dir, lib_dir, main_obj) or raises RuntimeError.
    """
    nvbench_path = Path(nvbench_path)
    include_dir = nvbench_path / "include"
    lib_dir = nvbench_path / "lib"
    main_obj = lib_dir / "objects-Release" / "nvbench.main" / "main.cu.o"

    if not include_dir.exists():
        raise RuntimeError(f"NVBench include dir not found: {include_dir}")
    if not lib_dir.exists():
        raise RuntimeError(f"NVBench lib dir not found: {lib_dir}")
    if not main_obj.exists():
        raise RuntimeError(
            f"NVBench main.cu.o not found: {main_obj}\n"
            "Ensure NVBench was built with cmake --build ... --target install"
        )
    return include_dir, lib_dir, main_obj


# ---------------------------------------------------------------------------
# Build and run
# ---------------------------------------------------------------------------

def build_benchmark(
    arch: str,
    nvbench_path: Path,
    tmpdir: str,
    config_file: Path = None,
    extra_nvcc_opts: list = None,
) -> Path | None:
    """Compile reduction_bench.cu linked with NVBench.

    Returns path to the compiled executable, or None on failure.
    """
    cuda_root, cuda_lib = get_cuda_paths()
    nvcc = str(cuda_root / "bin" / "nvcc")
    include_dir, lib_dir, main_obj = get_nvbench_paths(nvbench_path)

    exe = Path(tmpdir) / "reduction_bench"

    flags = [
        f"-arch={arch}",
        "-O3", "-std=c++17", "-DNDEBUG",
        f"-I{include_dir}",
        f"-L{lib_dir}",
        f"-Xlinker=-rpath,{lib_dir}",
    ]
    if config_file:
        # Forward controls to ptxas only (not libnvvm) via -Xptxas
        flags += ["-Xptxas", f"--apply-controls={config_file}"]
    if extra_nvcc_opts:
        flags += extra_nvcc_opts

    env = os.environ.copy()
    env["LIBRARY_PATH"] = f"{cuda_lib}:{env.get('LIBRARY_PATH', '')}"
    env["LD_LIBRARY_PATH"] = f"{cuda_lib}:{lib_dir}:{env.get('LD_LIBRARY_PATH', '')}"

    cmd = [
        nvcc, *flags,
        str(REDUCTION_CU),
        str(main_obj),
        "-lnvbench", "-lcudart_static", "-lcuda",
        "-o", str(exe),
    ]

    try:
        subprocess.run(cmd, env=env, capture_output=True, check=True, timeout=120)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        stderr = getattr(e, "stderr", b"")
        if stderr:
            print(f"Build failed: {stderr.decode(errors='replace')[:500]}")
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
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        p.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(p.pid), signal.SIGTERM)
        print("NVBench benchmark timed out")
        return None

    if p.returncode != 0:
        return None

    return parse_nvbench_result(Path(result_path))


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

def run_benchmark_only(args):
    """Benchmark-only mode: build with optional nvcc options and report timing."""
    extra_opts = shlex.split(args.nvcc_options) if args.nvcc_options else None

    with tempfile.TemporaryDirectory(prefix="ciq_nvbench_") as tmpdir:
        exe = build_benchmark(
            args.arch, args.nvbench_path, tmpdir, extra_nvcc_opts=extra_opts,
        )
        if not exe:
            print("Build failed")
            exit(1)

        score = run_nvbench(exe, args.elements_pow2, tmpdir)
        if score is None:
            print("Benchmark failed")
            exit(1)

        print(f"P75 latency: {score * 1000:.4f} ms  ({score:.6f} s)")


def run_optimization(args, cuda_version: str):
    """Run optimization to find the best PTXAS compiler config."""
    nvbench_path = args.nvbench_path

    # Run baseline (no compiler controls)
    print("Running baseline...")
    with tempfile.TemporaryDirectory(prefix="ciq_nvbench_base_") as tmpdir:
        exe = build_benchmark(args.arch, nvbench_path, tmpdir)
        if not exe:
            print("Baseline build failed")
            return 1
        baseline = run_nvbench(exe, args.elements_pow2, tmpdir)

    if baseline is None:
        print("Baseline benchmark failed")
        return 1
    print(f"Baseline P75: {baseline * 1000:.4f} ms\n")

    # Objective function: compile with PTXAS config, measure with NVBench
    def objective(config_str: str) -> float:
        with tempfile.TemporaryDirectory(prefix="ciq_nvbench_") as tmpdir:
            acf_path = Path(tmpdir) / "controls.acf"
            save_compiler_config(str(acf_path), config_str)

            exe = build_benchmark(args.arch, nvbench_path, tmpdir, config_file=acf_path)
            if not exe:
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
        print(f"Usage: nvcc -Xptxas --apply-controls={config_path} -arch={args.arch} ...")


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

    # Validate NVBench path
    if not args.nvbench_path:
        parser.error(
            "--nvbench-path is required (or set NVBENCH_PATH environment variable)"
        )

    # Check CUDA version
    version_output = subprocess.run(
        ["nvcc", "--version"], capture_output=True, text=True, check=True
    ).stdout
    cuda_version = re.search(r"release (\d+\.\d+),", version_output).group(1)
    assert float(cuda_version) >= 13.3, "CompileIQ requires CUDA 13.3+"

    if args.benchmark_only:
        run_benchmark_only(args)
    else:
        run_optimization(args, cuda_version)


if __name__ == "__main__":
    main()
