# CompileIQ and Normalization

Real-world benchmarks are noisy, and that noise can easily be the same size as the improvements you are trying to find.
This is common for highly optimized GPU kernels (e.g., FlashAttention, GEMM), where a “good” improvement might be only 1–9%, while run-to-run variance can be similar due to scheduling, memory pressure, and temperature.

This page is a practical guide to getting more stable signals into CompileIQ, with a focus on CompileIQ’s built-in normalization.

> Note: CompileIQ’s core algorithm already includes some noise-mitigation logic, but benchmarking variance can still dominate in practice.

> IMPORTANT: For best results, lock memory and SM clocks to reduce variation.

## When should I use normalization?

Enable normalization when:

* You expect small improvements and your benchmark is noisy.
* You run across multiple machines/GPUs and want a per-machine baseline.
* You want CompileIQ to compare candidates relative to a consistent reference point.

Normalization is not a replacement for good benchmarking hygiene (warmups, consistent inputs, stable environment), but it can make a big difference when the absolute numbers drift.

## Built-in Normalization

1. Turn on normalization in `SearchConfiguration`.
2. Make your objective function handle a special “baseline run” (CompileIQ passes an empty dict).
3. Confirm your objective returns the same number of values as `num_objectives`.

> Note: If the baseline measurement fails with `normalize=True`, CompileIQ will error out and exit the search early.

### Step 1: Enable `normalize=True`

```python
SearchConfiguration(
    normalize=True,
    ...
)
```

With `normalize=True`, CompileIQ periodically runs a baseline measurement and uses it to normalize subsequent objective measurements.
Normalization behavior is worker-specific (details below).

### Step 2: Make your objective baseline-safe

When normalization is enabled, CompileIQ will sometimes call your objective with an empty dictionary (`BASELINE_DNA`) instead of a sampled configuration. Your objective must treat this as “run the baseline” and return a valid measurement.

Below you can find a sample pseudo code with an example on how to handle baseline runs:

```python
from compileiq.types import BASELINE_DNA

def objective(sample: dict | str | list):
    if sample == BASELINE_DNA:
        # Run your reference configuration (the "no-tuning" setup)
        return run_benchmark_reference()

    # Run the candidate configuration
    return run_benchmark_candidate(sample)
```

### Step 3: Return the right shape

The baseline return value must match `num_objectives`. CompileIQ normalizes each objective against its corresponding baseline.

## What happens under the hood? (by worker)

### Async & Multiprocess Workers

Baselining is measured once at the start of the search.
It is the first object in the queue and the first evaluation to start.

### RayWorker

Ray normalizes per node.
If you are requesting GPU resources, the worker also ensures it baselines each GPU individually.
Baselines are stored in a matrix keyed by (node, GPU id), so tasks will be normalized against the same node/gpu-id pair.

> When normalizing with GPU requirements, we only support normalization for `num_gpus=1`.

If a new node joins mid-search, it will be baselined in the next generation.
If a task lands on a newly joined node before that generation starts (meaning the baseline matrix has no entry yet), the task will compute a local baseline and normalize itself.
Once the next generation begins, a new baseline is measured and stored globally to avoid repeating local baselines.

This flow assumes that baseline measurements for a given (node, GPU id) are stable enough that a local baseline will not diverge significantly from the next generation’s global baseline.

## For GPU Measurements

When performing measurements on GPU it is recommended to lock clocks to reduce variability. CompileIQ provides functionality to perform these operations through `nvidia-smi` calls under the hood.

```python
from compileiq.utils.gpu import gpu_benchmark_mode

with gpu_benchmark_mode(clock_mhz=1965, raise_on_failure=False):
    ... # everything inside here runs with locked clocks

# After exit we reset the clocks back to the default
```

Note that if your search runs locally, you may only need to set these before the search starts, but if you are running a distributed search you may need to lock the clocks in each machine first, or use `gpu_benchmark_mode` inside your objective function to make sure every eval locks the clock before running the benchmark.

## Other strategies (when built-in normalization is not enough)

If you already know your target hardware, it is often worth running your benchmark beforehand to understand the distribution.
Different systems can vary substantially even with the same GPU model.
Knowing whether your measurements are normal, bimodal, or “mostly stable” helps you choose a strategy.

### Option A: Pre-baked normalization factors

Compute a baseline once and normalize inside your objective.
In this setup, CompileIQ receives normalized scores and does not need to know how you computed them.

```python
# pseudo code
BASELINE = 300

def objective(dna):
    score = run_benchmark(dna)
    return norm(score, BASELINE)
```

### Option B: Normalize every evaluation

Measure a baseline each time, then normalize the candidate against it.
This can improve robustness to machine drift, but it increases overall search time.

```python
# pseudo code
def objective(dna):
    baseline_score = run_benchmark_reference()
    score = run_benchmark(dna)
    return norm(score, baseline_score)
```

### Option C: Reduce noise with repeated measurements

* Run multiple trials (and/or multiple inputs) and aggregate results (median/trimmed mean are often good defaults).
* Measure the baseline multiple times and take a conservative value (e.g., best-case baseline for minimization problems), then measure candidates multiple times and take a conservative value (e.g., worst case).
* Remove outliers when you can justify it (e.g., known thermal throttling events).
* Validate across multiple systems/GPUs before returning a score: dispatch runs in a fork-and-join pattern and only return “improved” if it is consistently improved across all tested nodes.
