"""Shared ptxas compile + CUDA timing helpers."""

from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from xla_ciq.search.cuda_runtime import CudaContext, CudaError, encode_tensor_map
from xla_ciq.search.shape_launch import grid_for_numel, parse_shape_token
from xla_ciq.search.types import SearchError
from xla_ciq.xla.tensor_map import TensorMapSpec


def temp_artifact_path(suffix: str) -> str:
    """Absolute path under the system temp dir (never the process CWD)."""
    return str(Path(tempfile.gettempdir()) / f"ciq_{uuid4().hex}{suffix}")


def ptxas_available() -> bool:
    from shutil import which

    return which("ptxas") is not None


def compile_ptx_to_cubin(
    *,
    ptx_path: str,
    arch: str,
    cubin_path: str,
    acf_bytes: bytes | None = None,
    timeout: float = 60.0,
) -> None:
    """Compile PTX to cubin, optionally applying an ACF via --apply-controls."""
    acf_path = None
    cmd = ["ptxas", "-v", f"-arch={arch}"]
    try:
        if acf_bytes is not None:
            acf_path = temp_artifact_path(".acf")
            Path(acf_path).write_bytes(acf_bytes)
            cmd.extend(["--apply-controls", acf_path])
        cmd.extend(["-o", cubin_path, ptx_path])
        compiled = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if compiled.returncode != 0 or not Path(cubin_path).is_file():
            detail = (compiled.stderr or compiled.stdout or "").strip()
            raise SearchError(f"ptxas failed: {detail}")
    finally:
        if acf_path and os.path.exists(acf_path):
            os.remove(acf_path)


def time_cubin(
    *,
    cubin_path: str,
    kernel_name: str,
    shape: str,
    block_size: int,
    num_args: int,
    timing_trials: int,
    warmup: int = 2,
    grid: int | tuple[int, int, int] | None = None,
    block: int | tuple[int, int, int] | None = None,
    shared_memory_bytes: int = 0,
    arg_nbytes: int | None = None,
    tensor_maps: dict[int, TensorMapSpec] | None = None,
) -> float:
    parsed = parse_shape_token(shape)
    nbytes = arg_nbytes if arg_nbytes is not None else parsed.nbytes
    tensor_maps = tensor_maps or {}
    if grid is None:
        grid = grid_for_numel(parsed.numel, block_size)
    if block is None:
        block = block_size
    with CudaContext() as ctx:
        module = ctx.load_cubin(cubin_path)
        ptrs: list = []
        try:
            func = ctx.get_function(module, kernel_name)
            ctx.allow_dynamic_shared_memory(func, shared_memory_bytes)
            arg_values: list = []
            for index in range(num_args):
                spec = tensor_maps.get(index)
                # A descriptor argument still needs backing global memory; the
                # buffer address lives inside the descriptor rather than in a
                # pointer slot, and its size comes from the tensor layout.
                size = spec.nbytes if spec is not None else nbytes
                dptr = ctx.alloc(size)
                ptrs.append(dptr)
                ctx.memset(dptr, 0, size)
                if spec is None:
                    arg_values.append(dptr)
                else:
                    arg_values.append(
                        encode_tensor_map(
                            spec,
                            device_ptr=int(dptr),
                            swizzle_bytes=spec.swizzle_candidates()[0],
                        )
                    )
            timed = ctx.time_launch(
                func,
                grid=grid,
                block=block,
                arg_values=arg_values,
                trials=timing_trials,
                warmup=warmup,
                shared_memory_bytes=shared_memory_bytes,
            )
            return timed.median_ms
        finally:
            for dptr in ptrs:
                ctx.free(dptr)
            ctx.unload(module)


def measure_kernel_ms(
    *,
    ptx_path: str,
    kernel_name: str,
    arch: str,
    shape: str,
    block_size: int,
    num_args: int,
    timing_trials: int,
    acf_bytes: bytes | None = None,
    warmup: int = 2,
    grid: int | tuple[int, int, int] | None = None,
    block: int | tuple[int, int, int] | None = None,
    shared_memory_bytes: int = 0,
    arg_nbytes: int | None = None,
    tensor_maps: dict[int, TensorMapSpec] | None = None,
) -> float:
    """Compile (optionally with ACF) and return median launch time in ms."""
    cubin_path = temp_artifact_path(".cubin")
    try:
        compile_ptx_to_cubin(
            ptx_path=ptx_path,
            arch=arch,
            cubin_path=cubin_path,
            acf_bytes=acf_bytes,
        )
        return time_cubin(
            cubin_path=cubin_path,
            kernel_name=kernel_name,
            shape=shape,
            block_size=block_size,
            num_args=num_args,
            timing_trials=timing_trials,
            warmup=warmup,
            grid=grid,
            block=block,
            shared_memory_bytes=shared_memory_bytes,
            arg_nbytes=arg_nbytes,
            tensor_maps=tensor_maps,
        )
    except (CudaError, SearchError):
        raise
    except Exception as exc:
        raise SearchError(f"CUDA launch failed: {exc}") from exc
    finally:
        if os.path.exists(cubin_path):
            os.remove(cubin_path)


@dataclass(frozen=True)
class TimingStats:
    median_ms: float
    mean_ms: float
    stdev_ms: float
    samples_ms: tuple[float, ...]


def measure_kernel_stats(
    *,
    ptx_path: str,
    kernel_name: str,
    arch: str,
    shape: str,
    block_size: int,
    num_args: int,
    timing_trials: int,
    acf_bytes: bytes | None = None,
    warmup: int = 2,
    rounds: int = 5,
    grid: int | tuple[int, int, int] | None = None,
    block: int | tuple[int, int, int] | None = None,
    shared_memory_bytes: int = 0,
    arg_nbytes: int | None = None,
    tensor_maps: dict[int, TensorMapSpec] | None = None,
) -> TimingStats:
    """Repeat measure_kernel_ms and return median/mean/stdev across rounds."""
    import statistics

    if rounds < 1:
        raise SearchError("rounds must be >= 1")
    samples = [
        measure_kernel_ms(
            ptx_path=ptx_path,
            kernel_name=kernel_name,
            arch=arch,
            shape=shape,
            block_size=block_size,
            num_args=num_args,
            timing_trials=timing_trials,
            acf_bytes=acf_bytes,
            warmup=warmup,
            grid=grid,
            block=block,
            shared_memory_bytes=shared_memory_bytes,
            arg_nbytes=arg_nbytes,
            tensor_maps=tensor_maps,
        )
        for _ in range(rounds)
    ]
    return TimingStats(
        median_ms=float(statistics.median(samples)),
        mean_ms=float(statistics.mean(samples)),
        stdev_ms=float(statistics.pstdev(samples)) if len(samples) > 1 else 0.0,
        samples_ms=tuple(samples),
    )
