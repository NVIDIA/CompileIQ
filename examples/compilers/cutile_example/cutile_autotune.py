"""
CompileIQ cuTile Example: autotune a CudaTile matmul kernel.

CompileIQ searches over surfaces cuTile-Python exposes:

  * tile sizes ``(tm, tn, tk)``    -- compile-time constants; each combo re-JITs
  * ``num_ctas`` / ``occupancy``   -- function-level hints via ``replace_hints``
  * ``latency`` / ``allow_tma``    -- per-load hints, threaded in as constants

Each candidate is a cuTile kernel launched and timed in-process, behind a
``torch.matmul`` correctness gate. ``latency`` and ``allow_tma`` apply to every
load in the kernel.

Usage:
    python cutile_autotune.py [--generations N] [--pool-size M]
"""

import argparse

import torch

import cuda.tile as ct

from compileiq.ciq import Search
import compileiq.search_spaces.base as ss
from compileiq.types import INVALID_SCORE, SearchConfiguration
from compileiq.utils.gpu import gpu_benchmark_mode

ConstInt = ct.Constant[int]

# Problem shape is fixed for the demo; every tile size below divides it evenly.
M, N, K = 1024, 1024, 1024

# Curated, known-valid tile shapes. Indexing into a finite list (rather than
# sweeping tm/tn/tk independently) keeps the autotuner away from illegal combos.
TILE_CONFIGS = [
    {"tm": 64, "tn": 64, "tk": 32},
    {"tm": 128, "tn": 64, "tk": 32},
    {"tm": 128, "tn": 128, "tk": 32},
    {"tm": 128, "tn": 128, "tk": 64},
]

# Single search space -> the objective receives a plain dict.
SEARCH_SPACE = {
    # A list index is categorical -- use choice() (not range()) so every config,
    # including index 0 and the last, is reachable and the optimizer treats the
    # indices as unordered labels rather than a numeric axis to interpolate.
    "tile_idx": ss.choice(list(range(len(TILE_CONFIGS)))),
    "num_ctas": ss.choice([1, 2]),
    "occupancy": ss.choice([1, 2, 3, 4]),
    "latency": ss.choice([1, 4, 8]),  # per-load DRAM-traffic hint (1--10)
    "allow_tma": ss.choice([0, 1]),  # 1 -> allow TMA on the loads, 0 -> forbid
}


@ct.kernel
def matmul_kernel(
    A, B, C, tm: ConstInt, tn: ConstInt, tk: ConstInt, latency: ConstInt, allow_tma: ConstInt
):
    """Tiled matmul ``C = A @ B``; latency/allow_tma are baked per load."""
    bidx = ct.bid(0)
    bidy = ct.bid(1)

    num_tiles_k = ct.num_tiles(A, axis=1, shape=(tm, tk))
    accumulator = ct.full((tm, tn), 0, dtype=ct.float32)
    zero_pad = ct.PaddingMode.ZERO
    use_tma = allow_tma == 1
    dtype = ct.tfloat32 if A.dtype == ct.float32 else A.dtype

    for k in range(num_tiles_k):
        a = ct.load(
            A,
            index=(bidx, k),
            shape=(tm, tk),
            padding_mode=zero_pad,
            latency=latency,
            allow_tma=use_tma,
        ).astype(dtype)
        b = ct.load(
            B,
            index=(k, bidy),
            shape=(tk, tn),
            padding_mode=zero_pad,
            latency=latency,
            allow_tma=use_tma,
        ).astype(dtype)
        accumulator = ct.mma(a, b, accumulator)

    accumulator = ct.astype(accumulator, C.dtype)
    ct.store(C, index=(bidx, bidy), tile=accumulator)


def build_kernel(config):
    """Apply the function-level hints for this candidate via ``replace_hints``."""
    return matmul_kernel.replace_hints(
        num_ctas=config["num_ctas"],
        occupancy=config["occupancy"],
    )


def run_matmul(kernel, a, b, c, config):
    """Launch one candidate into ``c``; tile sizes + per-load hints ride in as constants.

    The output buffer is passed in (not allocated here) so the timed loop measures
    only the kernel launch, not per-iteration allocator overhead.
    """
    tile = TILE_CONFIGS[config["tile_idx"]]
    grid = (ct.cdiv(M, tile["tm"]), ct.cdiv(N, tile["tn"]), 1)
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        kernel,
        (a, b, c, tile["tm"], tile["tn"], tile["tk"], config["latency"], config["allow_tma"]),
    )
    return c


def bench_ms(fn, warmup=25, iters=100):
    """Mean wall-clock per launch in milliseconds, measured with CUDA events."""
    for _ in range(warmup):
        fn()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def objective(config) -> float:
    """Verify correctness against torch, then return mean runtime (ms)."""
    device = torch.cuda.current_device()
    torch.manual_seed(0)
    a = torch.rand((M, K), device=device, dtype=torch.float16) - 0.5
    b = torch.rand((K, N), device=device, dtype=torch.float16) - 0.5

    try:
        kernel = build_kernel(config)
        c = torch.empty((M, N), device=device, dtype=torch.float16)
        out = run_matmul(kernel, a, b, c, config)
        ref = torch.matmul(a.float(), b.float()).to(torch.float16)
        torch.cuda.synchronize()
        if not torch.allclose(out, ref, atol=1e-1, rtol=1e-1):
            return INVALID_SCORE
        return bench_ms(lambda: run_matmul(kernel, a, b, c, config))
    except Exception:
        # Any compile/launch failure (e.g. an illegal hint combo) is just an
        # invalid candidate -- never crash the search.
        return INVALID_SCORE


def main():
    parser = argparse.ArgumentParser(description="Autotune a cuTile matmul with CompileIQ")
    parser.add_argument("--generations", type=int, default=5)
    parser.add_argument("--pool-size", type=int, default=16)
    parser.add_argument(
        "--clock-mhz",
        type=int,
        default=0,
        help="Lock GPU clocks during the search (0 = leave unlocked).",
    )
    args = parser.parse_args()

    config = SearchConfiguration(
        problem_type="min",
        generations=args.generations,
        pool_size=args.pool_size,
    )
    tuner = Search(
        objective_function=objective,
        search_space=SEARCH_SPACE,
        search_config=config,
    )

    # Locking GPU clocks before any latency measurement keeps the search stable.
    if args.clock_mhz:
        with gpu_benchmark_mode(clock_mhz=args.clock_mhz, raise_on_failure=False):
            results = tuner.start(task_timeout=60)
    else:
        results = tuner.start(task_timeout=60)

    best = results.get_best_result()
    tile = TILE_CONFIGS[best["params"]["tile_idx"]]
    print(f"Best runtime: {best['score_1']:.4f} ms")
    print(f"Best tile config: {tile}")
    print(
        f"Best hints: num_ctas={best['params']['num_ctas']} "
        f"occupancy={best['params']['occupancy']} "
        f"latency={best['params']['latency']} "
        f"allow_tma={bool(best['params']['allow_tma'])}"
    )


if __name__ == "__main__":
    main()
