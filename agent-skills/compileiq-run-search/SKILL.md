---
name: compileiq-run-search
description: >
  Use when composing the Search(...) call and calling .start(). Covers the
  four worker classes (MultiProcessWorker / IsoMultiProcessWorker / RayWorker
  / AsyncWorker) and when to pick each, SearchConfiguration sizing rules,
  dump_results checkpointing, tracker_config choice (Disabled / Loguru /
  MLflow), num_workers/task_timeout semantics, and GPU clock locking for
  stable measurements. Triggers on "Search()", "tuner.start()", "pool_size",
  "num_workers", "task_timeout", "IsoMultiProcessWorker", "RayWorker",
  "dump_results", "MLflow", "GPU clocks".
when_to_use: |
  - About to instantiate Search() and call .start().
  - Search converges too slow / too fast and the user is unsure how to size.
  - Search is hanging on a few configs (need IsoMultiProcessWorker + timeout).
  - Migrating from local to a Ray cluster.
license: Apache-2.0
metadata:
  version: "1.0.0"
  author: NVIDIA CompileIQ
  domain: compiler-optimization
allowed-tools: Bash Read
paths: ["**/*.py"]
---

# compileiq-run-search

After you have an objective function (from `compileiq-author-objective`) and
a search space (from `compileiq-search-space`), this skill helps you choose
the worker, size the configuration, and run the search safely.

## When

- About to instantiate `Search(...)` and call `.start()`.
- Search is converging too fast or too slow and the user is unsure how to
  re-size pool/generations.
- Search hangs on individual configs and the worker doesn't kill them.
- Scaling out from one GPU to a Ray cluster.

## Worker selection

Pass either a built-in `WorkerTypes` enum value or the worker class itself to
`Search(worker_type=...)`:

```python
from compileiq.types import WorkerTypes
from compileiq.worker import (
    MultiProcessWorker,    # default
    IsoMultiProcessWorker, # spawns fresh process per task; kill-safe
    RayWorker,             # distributed
    AsyncWorker,           # asyncio for async def objectives
)
```

| Situation | Worker class | Why |
|---|---|---|
| GPU kernel that may hang, OOM, or leak CUDA context | **`IsoMultiProcessWorker`** | One fresh process per task; parent kills on `task_timeout`. Defaults to `fork`. (`docs/workers.md:42`) |
| Triton mixed example on Blackwell-class GPUs | `WorkerTypes.ISOLATED` + `CIQ_PROCESS_MODE=spawn` | Isolates each evaluation and avoids leaking illegal memory access state across runs. |
| Fast (<100ms), stateless objective | `MultiProcessWorker` (default) | Reuses a pool; lower overhead. Defaults to `forkserver`. |
| Multi-node / multi-GPU cluster | `RayWorker` | User must set up Ray cluster + install compileiq on every worker. Both `num_workers` and `task_timeout` are ignored. (`docs/workers.md:79-91`) |
| I/O-bound `async def` objective | `AsyncWorker` | Concurrency, not parallelism. Rare for GPU work. |

**Default recommendation for compiler tuning of GPU kernels:**
`IsoMultiProcessWorker` with `task_timeout` between **30s** (small kernels) and
**180s** (large attention / XLA HLO).

## SearchConfiguration sizing

Reference: `compileiq/types.py:473-615`. Defaults auto-derive; only set what
you must.

```python
from compileiq.types import SearchConfiguration, ProblemType

config = SearchConfiguration(
    problem_type=ProblemType.MIN,   # MIN for latency; MAX for throughput
    generations=10,                  # required, > 0
    pool_size=15,                    # > 5; auto-derives if omitted
    # cull_size auto-derives to 75% of pool, rounded down to even
    # mutate_rate defaults to 0.25
    # num_objectives defaults to 1
    # normalize defaults to False (set True for cross-GPU runs)
)
```

| Knob | Default | When to override |
|---|---|---|
| `generations` | required | 10 for initial exploration; 20-40 for a deep run. |
| `pool_size` | auto (≥32) | 15 for tiny spaces; 32 for ≥1k design points; 64-128 for ≥10k. |
| `cull_size` | 75% of pool, even | Almost never override directly. |
| `mutate_rate` | 0.25 | Raise to 0.3-0.5 only if convergence stalls in early gens. |
| `num_objectives` | 1 | Must equal `len(return_tuple)` from the objective. |
| `normalize` | False | True when running across heterogeneous nodes or GPUs. |

**Sanity rule of thumb:** if `pool_size * generations < 50`, you are exploring,
not optimizing. If `> 2000`, you are probably overfitting to measurement noise
— `compileiq-validate-result` will earn its keep there.

## Search(...) constructor — every relevant kwarg

```python
from pathlib import Path
from compileiq.ciq import Search
from compileiq.search_spaces.compilers import PtxasSearchSpace
from compileiq.tracker import LoguruTrackerConfig

tuner = Search(
    objective_function=objective,
    search_space=PtxasSearchSpace(version="13.3", variant="att"),
    search_config=config,
    worker_type=IsoMultiProcessWorker,                 # or WorkerTypes.ISOLATED
    tracker_config=LoguruTrackerConfig(sink="optimization.log"),
    dump_results=Path("results.csv"),                  # ALWAYS set this
    cache_folder=None,                                  # default ~/.cache/compileiq
    disable_progress_bar=False,
    exit_on_failure=True,
    debug=False,
)
```

Always set `dump_results=Path(...)`. CSV is flushed every batch, so a crashed
or killed run leaves recoverable state.

## start(...) semantics

```python
results = tuner.start(num_workers=4, task_timeout=120)
```

- `num_workers`: ignored by workers where `respects_num_workers=False`
  (`RayWorker`, `AsyncWorker`); CompileIQ emits the warning
  `"num_workers is not supported by <WorkerName>"` (`compileiq/ciq.py:449-451`)
  so users recognize it.
- `task_timeout`: ignored where `supports_timeout=False` (`RayWorker`).
  Critical for `IsoMultiProcessWorker` — without it a hung config wedges that
  branch.
- Returns a `SearchResult`. Don't process inline; hand off to
  `compileiq-validate-result`.

## Tracker choice (one-line each)

```python
from compileiq.tracker import DisabledTrackerConfig, LoguruTrackerConfig, MLflowTrackerConfig
```

- `DisabledTrackerConfig()` — default, no overhead. Fine for one-off runs.
- `LoguruTrackerConfig(sink="optimization.log", level="INFO")` —
  recommended for serious campaigns. Negligible overhead.
- `MLflowTrackerConfig(experiment_name="...", tracking_uri="...", run_name="...")`
  — when integrating with ML Ops; creates a nested MLflow run per evaluation.

## Sample before you search

`Search.sample(n)` returns `n` randomly sampled parameter dicts from the
search space without running the search. Use it to:

1. Confirm the search space resolves at all (cheaper than the bootstrap
   round-trip; uses the in-memory state of `Search`).
2. Eyeball that the dicts have the keys your objective expects.
3. Feed a single sample into the objective by hand to verify it runs.

```python
sample = tuner.sample(1)[0]
print(sample)
print(objective(sample))   # should return a real float, not raise
```

## GPU clock locking (operator-level)

Stable measurements need locked clocks. Lock **before** `tuner.start()`,
unlock via `atexit`. Requires sudo.

```bash
sudo nvidia-smi -pm 1
MAX_GPU=$(nvidia-smi --query-gpu=clocks.max.graphics --format=csv,noheader,nounits | head -1)
MAX_MEM=$(nvidia-smi --query-gpu=clocks.max.memory --format=csv,noheader,nounits | head -1)
sudo nvidia-smi --lock-gpu-clocks=$MAX_GPU,$MAX_GPU --lock-memory-clocks=$MAX_MEM,$MAX_MEM
```

```python
import atexit, subprocess
def unlock():
    subprocess.run(["sudo", "nvidia-smi", "--reset-gpu-clocks", "--reset-memory-clocks"],
                   check=False)
atexit.register(unlock)
```

Inside a CI container or a shared cluster where sudo isn't available, skip
this; report higher CV% to the validation skill so it knows to compensate.

## Self-test

```bash
python scripts/smoke_search.py
```

Runs a 2-generation search on `x**2 + y` with `MultiProcessWorker` and
verifies `results.get_best_result()` returns a dict with `score_1` and `params`.

## Gotchas

- **Forgetting `task_timeout`** with `IsoMultiProcessWorker` is the most
  common reason a search hangs for hours. The worker will *kill* a stuck
  process but only after `task_timeout` elapses.
- **`forkserver` issues** on some hosts manifest as `EOFError` or "Broken pipe"
  on the first eval. Set `CIQ_PROCESS_MODE=spawn`.
- **`num_workers > num_gpus`** is fine for fast CPU-side objectives but
  oversubscribes GPUs for kernel objectives. For GPU kernels: pin
  `CUDA_VISIBLE_DEVICES` inside the objective and set
  `num_workers = num_gpus`.
- **Don't put GPU-clock lock calls inside the objective.** They require sudo
  and are per-host operator setup, not per-eval.

## Next

- After `.start()` returns: `compileiq-validate-result`.
- If something's wrong: `compileiq-debug`.
