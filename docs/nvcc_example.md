# Tuning NVCC compiler knobs for a CUDA kernel

In this section, we walk through optimizing the runtime of a CUDA reduction kernel by tuning NVCC compiler knobs with CompileIQ.

> The example code can be found [in our repo here](https://github.com/NVIDIA/CompileIQ/blob/main/examples/compilers/nvcc_example/optimize_reduction.py).

## CUDA Reduction Example

This example uses a self-contained CUDA reduction kernel (`reduction.cu`) that sums 64M integers using shared memory and warp shuffle. CompileIQ tunes the NVCC compiler knobs via `--apply-controls` to minimize runtime.

What you'll need:

* A Python environment with CompileIQ installed
* CUDA Toolkit (CTK) 13.3+
* A GPU (Blackwell sm_100 by default, adjustable via `--arch`)

### Building the objective function

The objective function receives a hex-encoded compiler configuration from CompileIQ, writes it to a temporary file, compiles the kernel with `--apply-controls`, runs it, and returns the execution time:

```python
from compileiq.types import INVALID_SCORE

def objective(config_blob: str) -> float:
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        f.write(bytes.fromhex(config_blob))
        config_path = Path(f.name)
    try:
        result = build_and_run(arch, config_path)
        return result["mean_ms"] if result["success"] else INVALID_SCORE
    finally:
        config_path.unlink(missing_ok=True)
```

The `build_and_run` helper compiles in a single step and benchmarks:

```python
subprocess.run([nvcc, "-arch=sm_100", "-O3", "-std=c++17",
                "--apply-controls", str(config_file),
                "reduction.cu", "-o", str(exe)], ...)
```

As described in our [Safety Section](compilers_overview.md#safety--correctness-read-this-first), the objective includes timeouts, correctness checking (`"Test passed"` validation), and returns `INVALID_SCORE` on any failure.

### Configuring the search

The search space is fetched automatically based on the detected CUDA version:

```python
from compileiq.search_spaces.compilers import NvccSearchSpace

cuda_version = re.search(r"release (\d+\.\d+),", version_output).group(1)
search_space = NvccSearchSpace(version=cuda_version)

config = SearchConfiguration(
    problem_type=ProblemType.MIN,
    generations=10,
    pool_size=15,
)

tuner = Search(
    objective_function=objective,
    search_space=search_space,
    search_config=config,
)
results = tuner.start(num_workers=1)
```

The script automatically runs a baseline (no compiler controls), then reports the speedup achieved by the best configuration found.

### Running the example

```bash
# Run optimization
python optimize_reduction.py --arch sm_100

# Benchmark the saved config
python optimize_reduction.py --benchmark-only \
    --nvcc-options "--apply-controls reduction_best_config.bin"
```

## Comparison with the PTXAS example

This NVCC example differs from the [PTXAS spill example](ptx_spill_example.md) in two key ways:

* **Full compilation**: NVCC compiles `.cu` source to a runnable binary, so each evaluation is slower but measures actual runtime performance.
* **Runtime metric**: The objective minimizes execution time rather than register spills.

Both examples follow the same pattern: define an objective, fetch a compiler search space, and run the evolutionary search.
