# Autotuning a cuTile kernel

This walkthrough uses CompileIQ to autotune a
[cuTile](https://github.com/NVIDIA/cutile-python) (CudaTile) matrix multiplication
kernel. The complete example is available in
[`examples/compilers/cutile_example`](https://github.com/NVIDIA/CompileIQ/tree/main/examples/compilers/cutile_example).

CompileIQ is normally used to tune NVIDIA internal compiler parameters, but it
can also tune user-level parameters. This example focuses on user-level cuTile
parameters. Compiler-parameter tuning for cuTile will be showcased in a future
version.

## What the example searches

The example searches four kinds of cuTile configuration:

| Parameter | cuTile surface | Effect |
|-----------|----------------|--------|
| `tile_idx` | Compile-time constants | Selects a known-valid `(tm, tn, tk)` tile shape. |
| `num_ctas` | `kernel.replace_hints(...)` | Sets the function-level CTA hint. |
| `occupancy` | `kernel.replace_hints(...)` | Sets the function-level occupancy hint. |
| `latency` | `ct.load(...)` | Sets the latency hint on every load. |
| `allow_tma` | `ct.load(...)` | Allows or forbids TMA on every load. |

The fixed problem shape is `1024 x 1024 x 1024`. The tile configurations are
curated so that every shape divides the problem dimensions evenly:

```python
TILE_CONFIGS = [
    {"tm": 64, "tn": 64, "tk": 32},
    {"tm": 128, "tn": 64, "tk": 32},
    {"tm": 128, "tn": 128, "tk": 32},
    {"tm": 128, "tn": 128, "tk": 64},
]

SEARCH_SPACE = {
    "tile_idx": ss.choice(list(range(len(TILE_CONFIGS)))),
    "num_ctas": ss.choice([1, 2]),
    "occupancy": ss.choice([1, 2, 3, 4]),
    "latency": ss.choice([1, 4, 8]),
    "allow_tma": ss.choice([0, 1]),
}
```

Using `choice` for the tile index tells CompileIQ that the configurations are
categorical. It also prevents the search from constructing unsupported
combinations by varying the three tile dimensions independently.

## Applying each candidate

The objective applies the function-level hints by creating a kernel variant for
the sampled candidate:

```python
def build_kernel(config):
    return matmul_kernel.replace_hints(
        num_ctas=config["num_ctas"],
        occupancy=config["occupancy"],
    )
```

The launch passes the selected tile dimensions and per-load hints as
compile-time constants. Each candidate therefore causes cuTile to JIT-compile
the corresponding kernel variant.

## Correctness and benchmarking

Every candidate is checked against `torch.matmul` before it is timed. A compile
failure, launch failure, or numerical mismatch returns `INVALID_SCORE` instead
of terminating the search:

```python
try:
    kernel = build_kernel(config)
    out = run_matmul(kernel, a, b, c, config)
    ref = torch.matmul(a.float(), b.float()).to(torch.float16)
    torch.cuda.synchronize()
    if not torch.allclose(out, ref, atol=1e-1, rtol=1e-1):
        return INVALID_SCORE
    return bench_ms(lambda: run_matmul(kernel, a, b, c, config))
except Exception:
    return INVALID_SCORE
```

The benchmark uses CUDA events and reports the mean kernel runtime in
milliseconds. Passing `--clock-mhz` also uses `gpu_benchmark_mode` to stabilize
GPU clocks for the duration of the search.

## Running the example

Install CompileIQ with the examples dependency group, then run the script on a
CUDA 13.3 or newer system with a supported NVIDIA GPU:

```bash
poetry install --with examples
cd examples/compilers/cutile_example
poetry run python cutile_autotune.py --generations 5 --pool-size 16
```

To lock GPU clocks during the run, add `--clock-mhz` with a clock supported by
your GPU:

```bash
poetry run python cutile_autotune.py \
    --generations 5 --pool-size 16 --clock-mhz 1965
```

The example prints the best runtime, tile configuration, and hint assignment:

```text
Best runtime: 0.0421 ms
Best tile config: {'tm': 128, 'tn': 128, 'tk': 64}
Best hints: num_ctas=2 occupancy=2 latency=4 allow_tma=True
```

The result is specific to the kernel, input shape, software stack, and GPU used
for the search. Re-run correctness and performance validation in the intended
deployment environment before using a configuration in production.
