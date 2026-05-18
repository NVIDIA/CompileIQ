# Tuning NVIDIA Compilers

Starting in CUDA Toolkit (CTK) 13.3, NVCC and PTXAS expose the __Advanced Controls Interface__ through the `--apply-controls` option. This interface lets you pass an __Advanced Controls File__ (ACF) to the compiler and change the compiler optimization choices used for that compilation.

CompileIQ uses this interface to generate ACFs for a given workload. In practice, this lets you adapt the compiler to the workload without changing the kernel source. If you want to try curated ACFs before running a full search, see [Booster Packs](booster_packs.md).

## What an ACF Changes

An ACF changes compiler optimization and control decisions for a compilation. It does not change your kernel source. The practical result can be different SASS, different register allocation, different scheduling choices, different memory behavior, or a different compile outcome.

This is what we mean by changing the compiler heuristics profile for a workload: the source stays fixed, but the compiler decisions can change.

## Leveraging the Controls Interface with CompileIQ

One of CompileIQ’s key features is support for NVIDIA compiler tuning. Currently supported compilers are NVCC and PTXAS.

To run a compiler search, select a curated compiler search space with `PtxasSearchSpace` or `NvccSearchSpace`. CompileIQ maps the compiler, compiler version, and variant to a published `.bin` release asset, verifies its `sha256` and size, caches it locally, and passes the local path into the normal `Search` flow.

```python
from compileiq.search_spaces.compilers import NvccSearchSpace, PtxasSearchSpace

ptxas_space = PtxasSearchSpace(version="13.3", variant="att")
nvcc_space = NvccSearchSpace(version="13.3")
```

For reproducible runs, pin a search-space release tag:

```python
ptxas_space = PtxasSearchSpace(
    version="13.3",
    variant="att",
    tag="search-spaces-2026.05.05",
)
```

For offline or mirrored environments, set `CIQ_SEARCH_SPACES_DIR` to a directory containing `manifest.json` and the referenced `.bin` files. For staging or a future dedicated artifact repository, set `CIQ_SEARCH_SPACES_REPO`.

This retrieval flow is for search-space `.bin` files used to generate new ACFs during a CompileIQ search. Curated Booster Packs are different artifacts: they are zip bundles of already-generated `.acf` candidates that you download and validate directly against a workload.

These files contain search-space information for CompileIQ to sample from. Each file includes a curated set of compiler controls that modify compiler behavior and interact with each other, producing different SASS.

> Each sample of the Search-Space makes an ACF

## Safety & correctness (read this first)

Before we dive into an example, this section outlines the most important things to know about tuning compiler controls.

> __TL;DR__: Treat ACFs as *per-kernel* and *per-environment*. Expect failures. Add timeouts, correctness checks, and good logging.

### ACFs rarely generalize

Do __NOT__ assume that the performance you find by applying an ACF to a specific kernel (with specific inputs) will generalize to other environments. In practice, an ACF can be sensitive to:

* The kernel and input shapes/workload mix.
* GPU architecture and GPU model.
* Driver + CUDA Toolkit version (CTK).
* Build flags, compiler versions, and runtime environment.

In rare cases, improvements found in one search extend to other code and GPUs, but those cases typically require the search code to perform validation and be intentionally designed for cross-environment verification.

### Expect failures (and handle them)

Because ACFs affect compiler behavior, and because the search space has an intractable number of combinations and dependencies, __applying ACFs may trigger compilation failures, compile hangs, numeric instability, and other unexpected behaviors__. Your objective function should handle these gracefully to avoid crashing the search.

Common failure modes include:

* Compile-time errors.
* Compile-time hangs.
* Runtime crashes.
* Silent wrong answers (numerical instability).
* Performance regressions.
* Higher variance/non-determinism (thermal and scheduling effects).

As CompileIQ progresses through generations, it will weed out invalid regions of the search space and reduce the failure rate.

### Minimum guardrails checklist

With these warnings in mind, it is the user’s responsibility to implement guardrails such as:

* __Timeouts__: hard timeouts for compilation and for the benchmark run.
* __Fail closed__: if compilation fails, a timeout triggers, or correctness fails, mark the sample invalid (don’t crash the search).
* __Correctness validation__: compare outputs against a known-good reference (with appropriate tolerances) across multiple inputs.
* __Repeatability__: run multiple trials and aggregate (e.g., median) if the benchmark is noisy.
* __Logging__: record the ACF, compiler versions, CTK/driver, GPU model, and benchmark configuration so results are reproducible.

When selecting a successful sample from the search, test it in the target deployment environment before shipping (same GPU model/arch, driver, CTK, clocks, and workload). Different systems may produce surprising results for the same kernel.

## Standalone Examples

Each example below follows this pattern: define an objective function, fetch a compiler search space, and run a CompileIQ search.

| Example | Compiler | Metric | Docs |
|---------|----------|--------|------|
| [NVCC reduction](https://github.com/NVIDIA/CompileIQ/blob/main/examples/compilers/nvcc_example/) | NVCC | Runtime (ms) | [NVCC example](nvcc_example.md) |
| [PTXAS spill reduction](https://github.com/NVIDIA/CompileIQ/blob/main/examples/compilers/ptxas_example/) | PTXAS | Spill bytes | [PTXAS example](ptx_spill_example.md) |
| [Triton matmul](https://github.com/NVIDIA/CompileIQ/blob/main/examples/compilers/triton_example/) | PTXAS via Triton | Runtime (ms) | [Triton example](triton_example.md) |
| [NVBench reduction](https://github.com/NVIDIA/CompileIQ/blob/main/examples/compilers/nvbench_example/) | PTXAS via NVCC | Runtime (P75, NVBench) | [Benchmarking](benchmarking.md) |

The PTXAS example is the simplest starting point — it only requires `ptxas` and runs on CPU (no GPU needed). The NVCC and Triton examples require a GPU to measure runtime.
