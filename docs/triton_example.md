# Tuning PTXAS controls in your Triton kernel

In this section, we will build on an existing Triton tutorial kernel and apply PTXAS ACFs.

> The example code and supporting files can be found [in our repo here](https://github.com/NVIDIA/CompileIQ/blob/main/examples/compilers/triton_example/triton_ptx.py).

## Matmul Triton Tutorial

The original kernel for the Triton tutorial can be found in [the Triton repository](https://github.com/triton-lang/triton/blob/v3.6.0/python/tutorials/03-matrix-multiplication.py).

Our goal is to apply CompileIQ on top of an existing matmul kernel to optimize runtime. Our changes are minimal: we do not modify the kernel code itself. Let’s walk through the integration before looking at the full objective function.

### Specific Changes to Triton

First, Triton is JIT-compiled, so we need to force recompilation for each ACF we try. An easy way to do this is by setting the environment variable `TRITON_ALWAYS_COMPILE`.

Second, at the time of writing, Triton ships with its own PTXAS binary for ease of use. However, we need PTXAS 13.3 or higher to use CompileIQ’s ACF support. Therefore, we also override Triton’s default PTXAS with a local version via `TRITON_PTXAS_PATH`.

You can do this programmatically with:

```python
import os
os.environ["TRITON_ALWAYS_COMPILE"] = "1"
os.environ["TRITON_PTXAS_PATH"] = "path/to/bin/ptxas"
```

> For blackwell overwrites use: `TRITON_PTXAS_BLACKWELL_PATH` instead of `TRITON_PTXAS_PATH`

Finally, we need to pass the ACF file to Triton so it can forward it to PTXAS. Triton already supports this via `ptx_options` when launching your kernel:

```python
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
    ACTIVATION=activation,
    ptx_options=f"--apply-controls={ptx_acf_filename}",
)
```

> You can optionally use the `PTX_OPTIONS` environment variable to pass in the `--apply-controls` flag instead of explicitly adding it to your kernel calls.

### Building the objective function

With everything in place, we can now build our objective function:

```python
def objective(ptx_acf: str) -> float:
    """
    Our objective will minimize runtime. It applies the given
    ACF to the kernel, verifies output correctness, and
    benchmarks the runtime.
    """

    ptxas_path = shutil.which("ptxas")  # Ensures Triton picks up the expected ptxas
    if ptxas_path is None:
        raise RuntimeError("ptxas not found in PATH.")

    os.environ["TRITON_PTXAS_PATH"] = ptxas_path
    os.environ["TRITON_ALWAYS_COMPILE"] = "1"

    a = torch.rand((512, 512), device=DEVICE, dtype=torch.float16) - 0.5
    b = torch.rand((512, 512), device=DEVICE, dtype=torch.float16) - 0.5

    with tempfile.NamedTemporaryFile(suffix=".acf", delete=True) as tmp_acf_file:
        save_compiler_config(tmp_acf_file.name, ptx_acf)

        triton_output = matmul(a, b, tmp_acf_file.name)
        torch_output = torch.matmul(a, b)

        # Validating output
        if not torch.allclose(triton_output, torch_output, atol=1e-2, rtol=0):
            runtime = INVALID_SCORE
        else:
            # Benchmarking Advanced Controls File
            runtime = triton.testing.do_bench(
                lambda: matmul(a, b, tmp_acf_file.name),
                warmup=100,
                rep=1000,
                return_mode="mean",
            )

    return runtime
```

Here’s what the function does:

* Sets environment variables for Triton to avoid caching and to use the expected PTXAS.
* Creates a temporary file containing the sampled ACF from CompileIQ.
* Passes that file into the `matmul` function from the [original tutorial](https://github.com/triton-lang/triton/blob/v3.6.0/python/tutorials/03-matrix-multiplication.py). This function is modified to pass the PTXAS CLI option `--apply-controls`, as described above.
* Validating result correctness against Torch's implementation.
* Performing benchmarking to get the runtime, only if the correctness test passed.

This objective is following all guidelines from our [Safety Section](compilers_overview.md#safety--correctness-read-this-first).

#### Expanding the search to different matrix sizes

In this example, we tune PTXAS controls for a specific matrix size of 512×512. If you want to support multiple sizes, you have a few options:

* Perform one different search for each size
* Benchmark and validate all matrix sizes inside the objective and return the mean, or only return a score if the ACF showed gains on most or all of the benchmarks
* Perform a [multi-objective search](getting_started.md#multi-objective-searches) where you return one score for each of the sizes you want to support.

#### A Note on Performance

It is important that all measurements performed during the search are reliable. We provide helper functionality to lock clocks which will help with reproducibility and runtime stability.

```python
from compileiq.utils.gpu import gpu_benchmark_mode

with gpu_benchmark_mode(clock_mhz=1965, raise_on_failure=False):
    results = tuner.start()
```

Because we are using the multiprocess worker that only works locally, we can lock the clocks once before starting the search. If you are using Ray on multiple machines, you may want to either lock the clocks beforehand or use `gpu_benchmark_mode` inside the objective function to make sure the clocks are locked for each individual evaluation independent on where it will execute.

## Expanding to Mixed-Search Spaces

Besides tuning PTXAS, CompileIQ often finds its best results when co-tuning other hyperparameters that matter to the application. In this example, Triton already does a good job with its own autotuner. As an example, we can disable the autotuner and expose those parameters to CompileIQ as part of the search space.

First, remove the autotune decorator and keep the configs in an accessible location:

```python
TRITON_CONFIGS = [
    {
        "block_size_m": 128,
        "block_size_n": 256,
        "block_size_k": 64,
        "group_size_m": 8,
        "num_stages": 3,
        "num_warps": 8,
    },
    {
        "block_size_m": 64,
        "block_size_n": 256,
        "block_size_k": 32,
        "group_size_m": 8,
        "num_stages": 4,
        "num_warps": 4,
    }, 
    ... 
    ]
```

We can now define a mixed search space containing the user space and PTX space.

```python
user_space = {"config_idx": ss.range(0, len(TRITON_CONFIGS) - 1)}
ptx_space = PtxasSearchSpace(version=cuda_version)

dna_config = [user_space, ptx_space]

tuner = Search(
    objective_function=objective,
    search_space=dna_config,
    search_config=main_config,
)
```

Here we create a range gene that will have the index to access `TRITON_CONFIGS`.

In the objective function, we now receive a list with the user space and PTX space sampled separately. We adjust the objective to read the index and pass the selected config to the kernel:

```python

def objective(mixed_ss: list[dict, str]) -> float:

    user_space, ptx_acf = mixed_ss
    config_idx = user_space["config_idx"]

    ...

    triton_output = matmul(a, b, tmp_acf_file.name, **TRITON_CONFIGS[config_idx])

```

With this adjustment, CompileIQ can tune both PTX and user-space parameters together !

### Expanding even further

The example above is illustrative and reuses the pre-defined configurations for the Triton matmul example. Alternatively, you can search over block and group sizes independently.

As an example, you could expose each option as a CompileIQ parameter:

```python

user_space = {
    "block_size_m": ss.range(16, 128, 16),
    "block_size_n": ss.range(16, 256, 16),
    "block_size_k": ss.range(16, 128, 16),
    "group_size_m": ss.choice([4, 8, 16]),
    "num_stages": ss.choice([2, 3, 4, 5]),
    "num_warps": ss.choice([2, 4, 8]),
}

ptx_space = PtxasSearchSpace(version=cuda_version)

dna_config = [user_space, ptx_space]

```

Although this approach might require longer searches, it provides better visibility into how each combination behaves. You might even find good solutions where you least expect them.
