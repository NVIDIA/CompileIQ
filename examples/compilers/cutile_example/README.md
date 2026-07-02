# cuTile Optimization Example

Autotune a [cuTile](https://github.com/NVIDIA/cutile-python) (CudaTile) kernel with
CompileIQ.

CompileIQ is normally used to tune NVIDIA internal compiler parameters, but it can
also tune user-level parameters. This example focuses on user-level cuTile parameters;
compiler-parameter tuning will be showcased in a future version.

## Example (`cutile_autotune.py`)

Searches over a tiled matmul `C = A @ B`:

- **Tile sizes** `(tm, tn, tk)` — compile-time constants; each combo re-JITs the kernel.
- **Function-level hints** — `num_ctas`, `occupancy`, applied via `kernel.replace_hints(...)`.
- **Per-load hints** — `latency`, `allow_tma`, threaded into `ct.load(...)` as constants.

Each candidate is launched in-process and timed with CUDA events, behind a
`torch.matmul` correctness gate; a failed compile/launch or a numerical mismatch
returns `INVALID_SCORE` so it never crashes the search.

```bash
python cutile_autotune.py --generations 5 --pool-size 16
```

## Requirements

- CUDA 13.3+ and a CUDA-capable GPU
- PyTorch with CUDA
- cuTile-Python (`cuda.tile`)
- `pip install compileiq`

## How it works

`SEARCH_SPACE` is a plain dict of `compileiq.search_spaces.base` primitives, so the
objective receives one config dict per candidate. The objective rebuilds the kernel
for that candidate (`replace_hints` for the function-level hints; tile sizes and
per-load hints ride in as `ct.Constant` kernel arguments), verifies it, and returns
the mean runtime in milliseconds. CompileIQ's evolutionary search minimizes that.

## Output

Prints the best tile config and hint assignment found, e.g.:

```
Best runtime: 0.0421 ms
Best tile config: {'tm': 128, 'tn': 128, 'tk': 64}
Best hints: num_ctas=2 occupancy=2 latency=4 allow_tma=True
```

## Notes

- `latency` and `allow_tma` apply to every load in the kernel.

## Files

- `cutile_autotune.py` — kernel, objective, and CompileIQ search.
