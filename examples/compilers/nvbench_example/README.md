# NVBench CMake Optimization Example

Optimize a CUDA reduction kernel with CompileIQ using NVBench for accurate
benchmarking. This variant builds the benchmark with CMake and links against the
NVBench installation embedded in the `cuda-bench` wheel.

## Why NVBench?

NVBench provides statistically rigorous kernel runtime measurements and should
be preferred to naive `cudaEvent` timing:

- **Cold measurements** — L2 cache is flushed between samples, preventing
  artificially warm cache from skewing results
- **Entropy-based convergence** — automatically collects enough samples until
  the timing distribution stabilizes
- **Throttling detection** — automatically discards measurements when thermal
  throttling is detected
- **Access to raw samples** — supports custom metrics such as P75 latency

## Prerequisites

- Linux
- CUDA 13.3+
- `nvcc` available through `PATH` or `CUDACXX`
- CMake >= 3.30.4
- Ninja
- Blackwell GPU (sm_100) or adjust `--arch`
- `pip install compileiq`
- `pip install "cuda-bench[cu13]>=0.3.0"`

The `cuda-bench` wheel contains a CUDA-versioned NVBench CMake installation.

## Quick Start

```bash
# Run optimization (auto-downloads PTXAS search space)
python optimize_reduction.py

# Benchmark with optimized config
python optimize_reduction.py --benchmark-only \
    --nvcc-options "-Xptxas --apply-controls=best_reduction.acf"
```

## Options

```bash
# Custom GPU architecture
python optimize_reduction.py --arch sm_90a

# More thorough search
python optimize_reduction.py --generations 20 --pool-size 30

# Smaller problem size (2^22 elements)
python optimize_reduction.py --elements-pow2 22

# Standalone benchmark (no optimization)
python optimize_reduction.py --benchmark-only --arch sm_100
```

## Files

- `reduction_bench.cu` — NVBench-instrumented CUDA reduction kernel
- `optimize_reduction.py` — Optimization and benchmarking script
- `nvbench_utils.py` — NVBench result parsing utilities
- `CMakeLists.txt` — standalone CMake project for the reduction benchmark
- `best_reduction.acf` — Best PTXAS config (generated)
- `optimization_results.csv` — Search history (generated)
