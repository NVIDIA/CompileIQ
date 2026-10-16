# Search Strategy

CompileIQ searches a large configuration space by evaluating batches of candidate
parameter sets, keeping the most useful signal from each batch, and using that
signal to choose the next batch. You usually only need to provide an objective
function, a search space, and a `SearchConfiguration`.

## Core Concepts

- A `search space` describes the parameters CompileIQ may vary.
- A `candidate` is one sampled parameter set from that space.
- A `pool` is the batch of candidates evaluated in one iteration.
- A `generation` is one search iteration.
- The `engine` is the optimizer that chooses candidates. The default, `amber`, is a
  native NSGA-III implementation.

For example, two parameters with ranges from 1 to 100 and step size 1 create
10,000 possible candidates. CompileIQ samples and evaluates a manageable subset
instead of requiring you to search that space manually.

## How a Search Progresses

1. CompileIQ samples an initial pool of candidates.
2. Workers evaluate each candidate by calling your objective function.
3. Scores are submitted to the selected search engine with `tell()`.
4. The engine uses the scores to choose another pool of candidates.
5. The process repeats until the configured number of generations completes.

Larger `pool_size` values give CompileIQ more candidates to compare per
generation. Larger `generations` values give it more rounds to refine the
search. For expensive compiler workloads, start small and scale only after your
objective function, correctness checks, and timeout behavior are reliable.

## Exploration And Refinement

CompileIQ balances two needs:

- Explore enough of the search space to avoid getting stuck on an early result.
- Refine promising regions so the final candidates are useful.

`pool_size` and `generations` are common controls. Seed windows declared by
`seed_low` and `seed_high` bias candidates toward a declared range, and
`init_with_true_random_threshold` sets how often they apply unless the
`seeded_range_probability` option overrides it. The `amber` engine owns
its remaining optimizer settings through `EngineConfig.options`, for example
`mutation_probability`, `survivor_count`, `reference_direction_weights`, and
`seed`. The `SearchConfiguration` fields `cull_size`, `mutate_rate`,
`objective_weights`, and `enable_large_fail_pool` have no effect since CompileIQ
1.1. `Search` ignores a non-default value for them with a
`FutureWarning` that names the closest `amber` option; those options are not
direct equivalents, so set them deliberately.
The defaults are intended to be usable for common cases, but large or
failure-heavy search spaces may need larger pools, more generations, or
objective-side guardrails that mark bad candidates as `INVALID_SCORE`.

`amber` never issues the same candidate twice in one search. If a small discrete
search space, or a search that has converged, leaves too few unseen candidates for
the next batch, `start()` ends the search early: it returns the results so far and
emits a `SearchSpaceExhaustedWarning`. If not even the first batch can be filled, it
raises `SearchSpaceExhaustedError`. A manual `ask()` raises the engine's error
instead. Widen the search space, or lower `pool_size` or `generations`, to run every
generation.

## Practical Guidance

Use `Search.sample(...)` before a full run to inspect candidate shapes and catch
objective-function assumptions early. This is especially useful for nested search
spaces and searches that combine application parameters with compiler controls,
where the objective function needs to handle a specific input shape.

Think about `pool_size` as the amount of evidence CompileIQ gathers in each
generation. A larger pool gives the engine more candidates to compare before it
chooses the next direction. Small pools are useful for smoke tests and expensive
workloads, but they can miss useful regions of a large search space. When the
objective is stable and affordable, increasing `pool_size` is often the first
way to make a search more robust.

Think about `generations` as the number of opportunities CompileIQ has to react
to measurements. More generations let the search refine promising regions over
time, but they also multiply the total number of objective evaluations. Start
with a short run to validate correctness, then scale the run once the objective
function, cache behavior, and timeout handling are reliable.

For `amber`, the `survivor_count` option controls how many candidates survive
into the next generation's parent pool. Smaller values focus the search more
aggressively around the best observed candidates. Larger values preserve more
variety, which can help when the score is noisy or when many candidates fail for
reasons unrelated to performance.

The `mutation_probability` option controls how often variables of new candidates
are perturbed away from the regions that already look promising. Too little
mutation can leave the search stuck around an early result. Too much mutation can
make the search keep exploring after it has enough signal to refine useful
candidates.

If later candidates appear less varied than the initial pool, the search may be
spending more effort near regions that have already scored well. If the best
score stops improving, the failure rate rises, or every sampled candidate looks
similar too early, consider increasing `pool_size`, running more generations, or
adjusting `mutation_probability`.

When tuning compiler controls, always add correctness checks, compile/runtime
timeouts, and repeatable benchmarking before trusting a candidate. A fast but
incorrect candidate is still a failed candidate.

## Versioned Search Engines

`Search` uses the `amber` engine by default: the native NSGA-III runtime included
in every complete platform-specific CompileIQ wheel. It needs nothing beyond the
public installation:

```bash
python -m pip install compileiq
```

```python
tuner = Search(
    objective_function=objective,
    search_space=search_space,
    search_config=config,
    engine="amber",  # the default; shown for clarity
)
```

In CompileIQ 1.1, `amber` replaced the original search core.

An engine name selects a complete optimizer implementation. Opaque names such as
`amber` keep that public identity stable without binding it to an algorithm
label. A provider that later offers multiple independently selectable
optimizers can register each one under a distinct engine name.

Use `EngineConfig` when an engine also accepts engine-owned settings:

```python
from compileiq.backends import EngineConfig

tuner = Search(
    objective_function=objective,
    search_space=search_space,
    search_config=config,
    engine=EngineConfig(
        name="amber",
        options={"seed": 17},
    ),
)
```

Every registered backend uses the same stateful `ask()` and `tell()` lifecycle
and accepts a non-empty subset of outstanding candidate IDs in any order; a
backend may consume each subset immediately or buffer it. Backends separately declare whether they can issue another
batch while earlier IDs remain outstanding. CompileIQ rejects empty, duplicate,
unknown, already-reported, and colliding candidate identifiers, mismatched
candidate parameters, and mismatched objective counts before backend state
changes. `sample()` uses a
separate ephemeral backend, so it cannot advance the optimizer used by a later
search.

For `amber`, `pool_size` is the initial generation target and the normal batch
size of a bare `ask()` while earlier candidates remain pending. It is not a hard
upper bound on one generation. An explicit `ask(num_samples=n)` returns exactly
`n` candidates: it fills currently unfilled target positions first, then expands
the generation target by any excess. A bare `ask()` while candidates are pending
issues another `pool_size` batch under the same fill-then-expand rule. When no
candidates are pending but target positions remain unfilled, a bare `ask()`
returns exactly the unfilled count.

For example, with `pool_size=8`, `ask(num_samples=4)`, followed by `ask()`, then
`ask(num_samples=1)` returns 4, 8, and 1 candidates in the same enlarged
13-candidate generation. Pending candidates are never used as parents, and the
generation advances only after every target position is filled and every issued
candidate resolves. Failed or explicitly discarded candidates vacate their
positions and are replaced one-for-one without creating checkpoint-specific ask
behavior.

For manual orchestration, `tuner.pending_candidates` exposes a read-only snapshot
of outstanding work and `tuner.result` exposes accepted history. This includes
state restored from a checkpoint. Scores are associated by candidate ID, not by
list position. Parameters need not be echoed in a score; when supplied they are
validated for compatibility, while the parameters stored with the issued
candidate remain authoritative.

Select `WorkerTypes.MANUAL` when candidate evaluation happens outside CompileIQ.
In this mode, `evaluate()` and `start()` fail before running the objective or
requesting a managed batch; `ask()`, `tell()`, result inspection, and checkpointing
remain available:

```python
from compileiq.types import WorkerTypes
from compileiq.utils.validation import Score


def managed_execution_disabled(_config):
    raise RuntimeError("This search is evaluated externally")


def run_externally(_config):
    return 0.0  # Replace with an externally measured objective value.


tuner = Search(
    objective_function=managed_execution_disabled,
    search_space=search_space_config,
    search_config=main_config,
    worker_type=WorkerTypes.MANUAL,
)

candidates = tuner.ask()
scores = [
    Score(
        score=run_externally(candidate.params),
        param_id=candidate.param_id,
        num_objectives=1,
    )
    for candidate in candidates
]
tuner.tell(scores)
best = tuner.best_result
```

With an executable worker, `evaluate()` can instead run any pending subset
without submitting it, advancing the engine, or changing results. Tell order is
irrelevant:

```python
executable_tuner = Search(
    objective_function=objective,
    search_space=search_space_config,
    search_config=main_config,
    worker_type=WorkerTypes.NATIVE,
)
candidates = executable_tuner.ask()
scores = executable_tuner.evaluate(candidates)
executable_tuner.tell(scores)
```

`best_result` is a convenience for `tuner.result.get_best_result()`: it returns
one result dictionary for a single-objective search and the Pareto-front list for
a multi-objective search. Other result queries remain on `SearchResult`.
Managed execution returns that object, so the corresponding one-call form is
`search.start().get_best_result()`.

`knockout_prob` is a probability from 0.0 to 1.0 that a parameter is absent in
the initially sampled population. Native evolutionary search may subsequently
evolve that binary on/off state; the value is not a fixed quota for every batch.

The `amber` engine supports sealed optimizer checkpoints. A checkpoint continues
the saved Search; it is not a warm start or a fresh search seeded from prior
information. Those are separate future capabilities and should use visibly
different API names.

Checkpoint creation is explicit. During manual `ask()` / `tell()` orchestration,
call `save_checkpoint()` whenever the current state should be captured:

```python
checkpoint = tuner.save_checkpoint("search.checkpoint")
```

The call captures the current state without first draining workers, waiting for
late results, cancelling remote jobs, or altering pending candidates.

For managed execution (where `start()` runs the ask, evaluate, and tell loop),
provide an optional path to atomically replace a checkpoint after every
successfully accepted batch:

```python
result = tuner.start(checkpoint_path="search.checkpoint")
```

With no `checkpoint_path`, `start()` creates no automatic checkpoints. An engine
that does not support checkpointing rejects the option before starting work.
CompileIQ also checks the configured result-history and population projection
against checkpoint-v1 limits before evaluating candidates. If that projection
is too large, reduce `generations` or `pool_size`, or run without managed
checkpointing. The projection is deliberately conservative for a search space
that may exhaust early; refills or manually expanded generations can also grow
history beyond the nominal projection. The save-time limit therefore remains
authoritative. Manual `save_checkpoint()` cannot predict an unbounded
`ask()`/`tell()` history, but reports that limit clearly before sealing or
creating an artifact.

CompileIQ does not impose a separate total checkpoint file-size limit for this
local API. The checkpoint schema, bounded fields and collections, duplicate-key
rejection, integrity checks, and backend-owned validation still apply. Services
or other untrusted ingestion boundaries should enforce their own resource limits
before passing a checkpoint to CompileIQ.

Managed checkpoints are quiescent: no candidates issued by Search remain
pending. A failed batch may leave ordinary target positions unfilled in the same
generation; that state is included before the next batch is asked. Manual
`save_checkpoint()` remains available when pending work must be captured.

The default restore policy resubmits candidates that were pending when a manual
checkpoint was created:

```python
resumed = Search.from_checkpoint(
    checkpoint,
    objective_function=objective,
    search_space=search_space,
)
result = resumed.start()
```

The saved search configuration and backend-owned optimizer options are restored
automatically. The objective function and search space remain caller-supplied
executable inputs. Checkpoints record only the canonical type names for built-in
workers and trackers, not their objects, credentials, clients, or configuration.
`from_checkpoint()` reconstructs those built-in types with default settings. A
custom or unknown worker requires an explicit `worker_type`; pass
`tracker_config` to retain non-default tracker settings such as a logging level
or tracking URI. Explicit worker and tracker arguments override the saved type
names.

The supplied search space must have the saved identity. Incompatible engine,
provider, configuration, and provider option overrides are rejected.
The saved `exit_on_failure` value is restored as the execution-policy default,
while an explicit `exit_on_failure=True` or `exit_on_failure=False` argument
overrides it. Direct `restore_checkpoint()` retains the fresh target Search's
configured policy.
`Search.from_checkpoint(...)` is the canonical restore API.

For development against an intentionally rebuilt provider, the canonical restore
API offers one narrow, unsafe provenance override:

```python
resumed = Search.from_checkpoint(
    checkpoint,
    objective_function=objective,
    search_space=search_space,
    allow_unsafe_provenance_mismatch=True,
)
```

This option is off by default and is intended only for debugging. It changes
`provider_version` and `artifact_sha256` mismatches from errors into one
`CheckpointCompatibilityWarning` that reports the saved and current values. The
equivalent process setting is
`CIQ_UNSAFE_ALLOW_CHECKPOINT_PROVENANCE_MISMATCH=1`; it is read when restore is
called.

The override does not bypass checkpoint-format parsing, ABI, engine, provider,
search-space identity, search configuration, sealed-state checksum
or authentication, or provider-owned state validation. A restore using it comes
from a non-matching build and should not be treated as an ordinary compatible
continuation. Intentional ABI or optimizer-state migrations require a
provider-specific migration mechanism or a future resealing tool, not this
facade-level provenance option.

With `RESUBMIT`, CompileIQ sends every restored pending candidate through the
newly configured worker using the same candidate identifier and parameters. It
does not ask the optimizer for replacements first. This is *at-least-once*
evaluation: CompileIQ does not reattach to an old worker or remote job, and it
does not promise the same job-level timing or execution trajectory. CompileIQ
does not discover results that arrive only on the original execution branch. A
manual caller may explicitly submit such a result while its candidate ID remains
pending in the restored Search.

To abandon the old pending work instead, restore with an explicit discard
policy:

```python
resumed = Search.from_checkpoint(
    checkpoint,
    objective_function=objective,
    search_space=search_space,
    worker_type=worker_type,
    pending_policy="discard",
)
```

`DISCARD` removes those candidates from both public pending state and native
optimizer state. Their identifiers are no longer accepted by `tell()`. The
generation target does not shrink, so each discarded candidate creates one
ordinary unfilled position. With no other candidates pending, the next bare
`ask()` fills every unfilled position, including any that already existed before
the checkpoint. There is no checkpoint-only replacement mode.

| Behavior | Resubmit, the default | Discard, explicit |
| --- | --- | --- |
| Saved pending IDs and parameters | Preserved | Removed |
| First work after restore | Same candidates through the new worker | Fill the generation's ordinary unfilled positions |
| Original remote jobs | Not reattached or cancelled | Not reattached or cancelled |
| Duplicate execution risk | Yes; evaluation is at least once | Original work may be wasted while replacements run |
| Result completed on the original branch | Not discovered automatically; a manual caller may submit it while the ID remains pending | Rejected because the ID is no longer pending |
| Optimizer trajectory | Preserves the saved outstanding set | May diverge and advance sampling state |

For example, if candidate ID `41` with parameters `{x: 0.2}` is pending,
resubmit evaluates that same ID and parameters through the new worker. Discard
removes ID `41`, rejects a later score for it, and leaves its target position
unfilled. If every other target position was already occupied, the next bare
ask returns one replacement; a partially issued generation may have additional
unfilled positions to return at the same time.

If a workflow must checkpoint only at a quiescent boundary, fail closed instead
of draining automatically:

```python
checkpoint = tuner.save_checkpoint(
    "search.checkpoint",
    require_no_pending=True,
)
```

When candidates are outstanding, this raises before backend sealing and does not
create or replace an artifact. CompileIQ does not cancel jobs, drain workers, or
reattach to worker state as part of checkpoint creation or restoration.

The same checkpoint artifact carries the accumulated user-visible
`SearchResult`. Completed accepted scores are restored automatically, and
`resumed.start()` appends later accepted scores to that history. Pending,
discarded, rejected, duplicate, and late scores do not create result rows.
Protected search results retain the same opaque formatted candidate payloads
shown through the ordinary `SearchResult`; raw protected knob names and values
are not added to the checkpoint.

Checkpoints include public compatibility metadata and an opaque backend-sealed
execution-state document. The native document follows the strict
`compileiq-checkpoint-v1` contract: unknown fields and generic metadata are
forbidden, sizes and nesting are bounded, protected genotype names are excluded,
and the native checkpoint-sealing component validates and canonicalizes the
complete document before sealing and after unsealing. The user-visible result ledger is separately schema- and
resource-validated and is never used to reconstruct optimizer state. Restore
fails closed when the engine, provider, runtime artifact, ABI, search
configuration, search-space identity, checksum, or sealed data does not match.

At a quiescent boundary with no pending candidates, restoring the sealed native
optimizer and RNG state provides exact optimizer continuation. A checkpoint with
pending work preserves candidate identity, but the at-least-once resubmission
policy means CompileIQ does not claim preservation of the exact execution
trajectory. Exact continuation assumes a compatible native runtime environment;
it does not promise identical trajectories across runtime, dependency, Python,
or platform upgrades. Direct `restore_checkpoint()` is limited to a fresh, compatible
`Search`; it does not merge unrelated active state or result history. An engine
that does not advertise checkpoint support rejects both calls before reading the
artifact.
