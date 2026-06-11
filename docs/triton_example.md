# Tuning PTXAS for your Triton kernel

This section shows how to apply CompileIQ-generated PTXAS ACFs to a Triton
kernel. The basic example uses a fixed-config illustrative matmul kernel and
searches only PTXAS controls. The companion mixed example shows how to search
Triton launch/configuration knobs and PTXAS controls together.

> The example code and supporting files can be found [in our repo here](https://github.com/NVIDIA/CompileIQ/blob/main/examples/compilers/triton_example/triton_ptx.py).

> Examples may become stale as triton or the compiler improve, and these examples are simple in nature, meant to teach you how to incorporate CompileIQ into your existing code

## Fixed-config matmul example

The basic `triton_ptx.py` example keeps the Triton kernel configuration fixed:

```python
BLOCK_M = 32
BLOCK_N = 64
BLOCK_K = 32
NUM_WARPS = 4
NUM_STAGES = 3
```

It then tunes only the PTXAS compiler controls for that kernel and matrix
shape. The example uses exact-tile `4096 x 4096` inputs, so `M`, `N`, and `K`
must be divisible by the block sizes.

## Specific changes for Triton

Triton is JIT-compiled, so force recompilation for each ACF:

```python
os.environ["TRITON_ALWAYS_COMPILE"] = "1"
```

Triton ships its own PTXAS binary. CompileIQ ACF support requires
PTXAS 13.3 or newer, so point Triton at the expected local `ptxas`. Set both
environment variables when targeting systems where Triton may select a
Blackwell-specific PTXAS path:

```python
ptxas_path = shutil.which("ptxas")
if ptxas_path is None:
    raise RuntimeError("ptxas not found in PATH.")

os.environ["TRITON_PTXAS_PATH"] = ptxas_path
os.environ["TRITON_PTXAS_BLACKWELL_PATH"] = ptxas_path
```

Pass the ACF to Triton through the `ptx_options` kernel-launch keyword:

```python
ptxas_options = f"--apply-controls={controls_path}" if controls_path else None

matmul_kernel[grid](
    a,
    b,
    c,
    M,
    N,
    K,
    a.stride(0),
    a.stride(1),
    b.stride(0),
    b.stride(1),
    c.stride(0),
    c.stride(1),
    BLOCK_M=BLOCK_M,
    BLOCK_N=BLOCK_N,
    BLOCK_K=BLOCK_K,
    num_warps=NUM_WARPS,
    num_stages=NUM_STAGES,
    ptx_options=ptxas_options,
)
```

> You can optionally use the `PTX_OPTIONS` environment variable to pass in the `--apply-controls` flag instead of explicitly adding it to your kernel calls. Note that this will be applied globally

### Building the objective function

The objective writes each sampled CompileIQ config to a temporary `.acf` file,
passes that file to the Triton launch, checks correctness against Torch, and
only then benchmarks runtime:

```python
def objective(config) -> float:
    ptxas_path = shutil.which("ptxas")
    if ptxas_path is None:
        raise RuntimeError("ptxas not found in PATH.")

    os.environ["TRITON_PTXAS_PATH"] = ptxas_path
    os.environ["TRITON_PTXAS_BLACKWELL_PATH"] = ptxas_path
    os.environ["TRITON_ALWAYS_COMPILE"] = "1"

    a = torch.rand((4096, 4096), device=DEVICE, dtype=torch.float16) - 0.5
    b = torch.rand((4096, 4096), device=DEVICE, dtype=torch.float16) - 0.5

    with tempfile.NamedTemporaryFile(suffix=".acf", delete=True) as f:
        save_compiler_config(f.name, config)
        triton_out = matmul(a, b, f.name)
        torch_out = torch.matmul(a, b)

        if not torch.allclose(triton_out, torch_out, atol=1e-2, rtol=0):
            return INVALID_SCORE

        return triton.testing.do_bench(
            lambda: matmul(a, b, f.name),
            warmup=100,
            rep=1000,
            return_mode="mean",
        )
```

This objective follows the guardrails in the [Safety Section](compilers_overview.md#safety--correctness-read-this-first):

* Force recompilation so each ACF can affect generated code.
* Use an explicit PTXAS 13.3+ path.
* Validate correctness before timing.
* Return `INVALID_SCORE` for wrong answers.

### Expanding the search to different matrix sizes

In the basic example, the search is specific to `4096 x 4096` matmul with the
fixed block sizes above. If you want to support multiple sizes, you have a few
options:

* Run a separate search for each size.
* Benchmark and validate all matrix sizes inside the objective and return an aggregate score.
* Use a [multi-objective search](getting_started.md#multi-objective-searches) and return one score per size.

### A note on performance

Reliable latency measurements are important during the search. CompileIQ
provides a helper to lock clocks:

```python
from compileiq.utils.gpu import gpu_benchmark_mode

with gpu_benchmark_mode(clock_mhz=1965, raise_on_failure=False):
    results = tuner.start(task_timeout=20)
```

Because the default multiprocessing worker runs locally, you can lock clocks
once before starting the search. If you use Ray across multiple machines, lock
clocks before the run or use `gpu_benchmark_mode` inside the objective function
so each remote evaluation runs under the same clock policy.

## Expanding to mixed search spaces

Besides tuning PTXAS, CompileIQ can co-tune application parameters. The
`mixed_triton.py` example searches a user-defined index into a list of Triton
configs plus the PTXAS search space:

```python
TRITON_CONFIGS = [
    {"block_m": 32, "block_n": 64, "block_k": 32, "stages": 3, "warps": 4},
    {"block_m": 64, "block_n": 64, "block_k": 32, "stages": 3, "warps": 4},
    {"block_m": 64, "block_n": 128, "block_k": 32, "stages": 4, "warps": 4},
    {"block_m": 128, "block_n": 128, "block_k": 32, "stages": 4, "warps": 4},
    {"block_m": 128, "block_n": 256, "block_k": 64, "stages": 3, "warps": 8},
]

search_space = [
    {"config_idx": ss.range(0, len(TRITON_CONFIGS) - 1)},
    PtxasSearchSpace(version=cuda_version),
]
```

The objective receives a list with one entry per search-space component:

```python
def objective(mixed_config: list) -> float:
    user_space, ptxas_config = mixed_config
    cfg = TRITON_CONFIGS[user_space["config_idx"]]

    ...

    with tempfile.NamedTemporaryFile(suffix=".acf", delete=True) as f:
        save_compiler_config(f.name, ptxas_config)
        triton_output = matmul(a, b, f.name, cfg)
```

Mixed-search results keep that same list shape in `best["params"]`. Unpack it
before saving the ACF:

```python
best = results.get_best_result()
user_space, ptxas_config = best["params"]

print(f"Best Triton config index: {user_space['config_idx']}")
save_compiler_config("best_matmul.acf", ptxas_config)
```

```python
tuner = Search(
    objective_function=objective,
    search_space=search_space,
    search_config=config,
)
```

### Expanding even further

The example above indexes into a curated list of supported Triton configs.
Alternatively, you can search over block and launch parameters directly:

```python
user_space = {
    "block_m": ss.range(16, 128, 16),
    "block_n": ss.range(16, 256, 16),
    "block_k": ss.range(16, 128, 16),
    "stages": ss.choice([2, 3, 4, 5]),
    "warps": ss.choice([2, 4, 8]),
}

search_space = [
    user_space,
    PtxasSearchSpace(version=cuda_version),
]
```

This approach may require longer searches, but it gives CompileIQ visibility
into each parameter combination.

