# NVBench Optimization Example

Optimize a CUDA reduction kernel with CompileIQ using NVBench for accurate
benchmarking and the PTXAS search space for compiler control tuning.

## Why NVBench?

NVBench provides statistically rigorous and accurate kernel runtime measurements
and should be preferred to naive use of cudaEvent timing (as used in
`../nvcc_example/`):

- **Cold measurements** — L2 cache is flushed between samples, preventing
  artificially warm cache from skewing results
- **Entropy-based convergence** — automatically collects enough samples until
  the timing distribution stabilizes, rather than using a fixed iteration count
- **Throttling detection** — automatically discards measurements when thermal
  throttling is detected
- **Access to full sample** — allows implementing custom metrics, e.g.
  75-percentile, which are more robust and less sensitive to outliers than
  the mean

## Why PTXAS Controls on CUDA Source?

The existing `ptxas_example/` applies PTXAS controls to pre-compiled `.ptx`
files. This example shows that PTXAS controls also work on `.cu` source files
compiled through `nvcc` — using `-Xptxas --apply-controls=<file>` to forward
controls to the internal PTXAS invocation. This opens up PTXAS-level tuning
for any CUDA kernel without requiring a separate PTX compilation step.

## Prerequisites

- CUDA 13.3+
- NVBench (built and installed — see below)
- Blackwell GPU (sm_100) or adjust `--arch`
- `pip install compileiq`

### NVBench Installation

```bash
git clone https://github.com/NVIDIA/nvbench.git
cd nvbench
cmake -B build -G Ninja --preset nvbench-ci \
  -DCMAKE_INSTALL_PREFIX=$(pwd)/build/nvbench_install \
  -DCMAKE_CUDA_COMPILER=$(which nvcc) \
  -DCMAKE_CUDA_ARCHITECTURES=100 \
  -DNVBench_ENABLE_CUPTI=OFF \
  -DNVBench_ENABLE_NVML=OFF
cmake --build build --target install
export NVBENCH_PATH=$(pwd)/build/nvbench_install
```

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
- `best_reduction.acf` — Best PTXAS config (generated)
- `optimization_results.csv` — Search history (generated)
