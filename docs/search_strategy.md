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
- `mutate_rate` controls how aggressively later candidates explore new values.

For example, two parameters with ranges from 1 to 100 and step size 1 create
10,000 possible candidates. CompileIQ samples and evaluates a manageable subset
instead of requiring you to search that space manually.

## How a Search Progresses

1. CompileIQ samples an initial pool of candidates.
2. Workers evaluate each candidate by calling your objective function.
3. Scores are returned to the core over local socket IPC.
4. The core uses the scores to choose another pool of candidates.
5. The process repeats until the configured number of generations completes.

Larger `pool_size` values give CompileIQ more candidates to compare per
generation. Larger `generations` values give it more rounds to refine the
search. For expensive compiler workloads, start small and scale only after your
objective function, correctness checks, and timeout behavior are reliable.

## Exploration And Refinement

CompileIQ balances two needs:

- Explore enough of the search space to avoid getting stuck on an early result.
- Refine promising regions so the final candidates are useful.

`mutate_rate`, `pool_size`, `cull_size`, and `init_with_true_random_threshold`
control this balance. The defaults are intended to be usable for common cases,
but large or failure-heavy search spaces may need larger pools, more generations,
or objective-side guardrails that mark bad candidates as `INVALID_SCORE`.

## Practical Guidance

Use `Search.sample(...)` before a full run to inspect candidate shapes and catch
objective-function assumptions early. This is especially useful for nested search
spaces and searches that combine application parameters with compiler controls,
where the objective function needs to handle a specific input shape.

Think about `pool_size` as the amount of evidence CompileIQ gathers in each
generation. A larger pool gives the core more candidates to compare before it
chooses the next direction. Small pools are useful for smoke tests and expensive
workloads, but they can miss useful regions of a large search space. When the
objective is stable and affordable, increasing `pool_size` is often the first
way to make a search more robust.

Think about `generations` as the number of opportunities CompileIQ has to react
to measurements. More generations let the search refine promising regions over
time, but they also multiply the total number of objective evaluations. Start
with a short run to validate correctness, then scale the run once the objective
function, cache behavior, and timeout handling are reliable.

`cull_size` controls how much of the current pool is used as signal for later
generations. Smaller values focus the search more aggressively around the best
observed candidates. Larger values preserve more variety, which can help when
the score is noisy or when many candidates fail for reasons unrelated to
performance.

`mutate_rate` controls how often later candidates are perturbed away from the
regions that already look promising. Too little mutation can leave the search
stuck around an early result. Too much mutation can make the search keep
exploring after it has enough signal to refine useful candidates.

If later candidates appear less varied than the initial pool, the search may be
spending more effort near regions that have already scored well. If the best
score stops improving, the failure rate rises, or every sampled candidate looks
similar too early, consider increasing `pool_size`, running more generations, or
adjusting `mutate_rate`.

When tuning compiler controls, always add correctness checks, compile/runtime
timeouts, and repeatable benchmarking before trusting a candidate. A fast but
incorrect candidate is still a failed candidate.
