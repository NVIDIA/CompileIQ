# Benchmarking Approaches

CompileIQ's compiler tuning requires accurate, repeatable kernel timing.
Different measurement tools suit different workflows. This page summarizes
the approaches used across CompileIQ's examples and when to choose each one.

## NVBench (C++ CUDA kernels)

[NVBench](https://github.com/NVIDIA/nvbench) is a CUDA kernel benchmarking
library that provides cold measurements (L2 cache flushed between samples),
entropy-based convergence, and full timing distributions.

NVBench is the recommended tool for benchmarking C++ CUDA kernels. See the
[NVBench optimization example](https://github.com/NVIDIA/CompileIQ/blob/main/examples/compilers/nvbench_example/)
for a complete walkthrough.

**When to use**: C++ CUDA kernels where you want statistically rigorous,
cache-cold measurements without writing your own timing harness.

**Key features**:

- L2 cache flushed between samples for cold measurements
- Entropy-based stopping criterion adapts sample count to noise
- Full sample distribution available for custom aggregation (P75, median, etc.)
- Automatic discarding of measurements when thermal throttling is detected

## Triton `do_bench` (Python / Triton kernels)

Triton's built-in `triton.testing.do_bench` provides a convenient Python-native
benchmarking function with configurable warmup and repetitions. It is used in
the [Triton optimization example](triton_example.md).

```python
runtime = triton.testing.do_bench(
    lambda: matmul(a, b, acf_file),
    warmup=100,
    rep=1000,
    return_mode="mean",
)
```

**When to use**: Triton kernels or any GPU kernel callable from Python. Simple
to integrate and avoids compilation of a separate benchmark binary.

## Manual CUDA Events (simple C++ timing)

For lightweight benchmarking without external dependencies, `cudaEventRecord`
and `cudaEventElapsedTime` can bracket a kernel launch. This approach is used
in the [NVCC optimization example](nvcc_example.md).

**When to use**: Self-contained prototype CUDA programs where you want minimal
dependencies and full control over warmup iterations. For production use-cases
use of NVBench is strongly recommended.

## Choosing a benchmarking approach

| Approach | Language | Cache-cold | Statistical rigor | Setup complexity |
|----------|----------|------------|-------------------|------------------|
| NVBench | C++ | Yes | High (entropy-driven) | Medium (requires NVBench install) |
| `do_bench` | Python | No | Medium (fixed warmup + reps) | Low (part of Triton) |
| CUDA Events | C++ | No | Basic (manual runs) | Low (no dependencies) |

## Profiling with NVIDIA Nsight Compute

Benchmarking measures *how long* a kernel takes. **Profiling** answers
*why* it takes that long -- identifying memory bottlenecks, occupancy
limits, instruction mix, and warp stalls.

[NVIDIA Nsight Compute](https://developer.nvidia.com/nsight-compute)
is the recommended profiling tool for CUDA kernels. While CompileIQ
examples focus on benchmarking (timing is the optimization objective),
Nsight Compute is invaluable for understanding *what* to optimize
before or after a CompileIQ search.

> Nsight Compute integration examples are planned for a future release.
