"""Validate a CUDA launch in a disposable subprocess.

An out-of-bounds launch raises a *sticky* CUDA error that cannot be cleared:
after it fires, every later launch in the same process fails, and even a
freshly created context inherits the error. Probing in a throwaway process
keeps the caller healthy so one unlaunchable shape cannot poison a search.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass, field

from xla_ciq.xla.tensor_map import TensorMapError, TensorMapSpec

# A dumped kernel indexes the shape it was compiled for, not the `--shapes`
# token being fingerprinted, and its read extent follows from the launch grid.
# Every candidate therefore carries headroom: a buffer sized to exactly the
# requested shape can survive one probe launch and then fault mid-search, when
# a different allocation no longer pads the overrun.
_FOOTPRINT_MULTIPLIERS = (2, 8)
_PROBE_TIMEOUT_S = 120.0


@dataclass(frozen=True)
class ProbeResult:
    arg_nbytes: int
    median_ms: float
    tensor_maps: dict[int, TensorMapSpec] = field(default_factory=dict)


def _with_swizzle(spec: TensorMapSpec, swizzle_bytes: int) -> TensorMapSpec:
    return TensorMapSpec(
        dtype=spec.dtype,
        dims=spec.dims,
        box=spec.box,
        element_strides=spec.element_strides,
        swizzle_bytes=swizzle_bytes,
        source=spec.source,
    )


def resolve_swizzle_options(specs: dict[int, TensorMapSpec]) -> list[int]:
    """Swizzle modes worth trying for a whole kernel, most likely first.

    PTX does not record the swizzle the emitter chose, so it has to be probed.
    Descriptors in one kernel share a shared-memory layout, so only modes legal
    for every unresolved descriptor are worth a launch.
    """
    unresolved = [s for s in specs.values() if s.swizzle_bytes is None]
    if not unresolved:
        return [0]
    options = list(unresolved[0].swizzle_candidates())
    for spec in unresolved[1:]:
        allowed = set(spec.swizzle_candidates())
        options = [o for o in options if o in allowed]
    return options or [0]


def candidate_arg_nbytes(*, shape_nbytes: int, footprint_nbytes: int) -> list[int]:
    """Buffer sizes to try, smallest first."""
    base = max(shape_nbytes, footprint_nbytes)
    ordered: list[int] = []
    for multiplier in _FOOTPRINT_MULTIPLIERS:
        value = base * multiplier
        if value > 0 and value not in ordered:
            ordered.append(value)
    return ordered


def probe_launch(
    *,
    ptx_path: str,
    kernel_name: str,
    arch: str,
    shape: str,
    block_size: int,
    num_args: int,
    grid: tuple[int, int, int],
    block: tuple[int, int, int],
    shared_memory_bytes: int,
    shape_nbytes: int,
    footprint_nbytes: int,
    tensor_maps: dict[int, TensorMapSpec] | None = None,
) -> ProbeResult | None:
    """Return the smallest buffer size that launches cleanly, or None."""
    tensor_maps = tensor_maps or {}
    for arg_nbytes in candidate_arg_nbytes(
        shape_nbytes=shape_nbytes, footprint_nbytes=footprint_nbytes
    ):
        for swizzle in resolve_swizzle_options(tensor_maps):
            resolved = {
                index: (
                    spec
                    if spec.swizzle_bytes is not None
                    else _with_swizzle(spec, swizzle)
                )
                for index, spec in tensor_maps.items()
            }
            try:
                for spec in resolved.values():
                    spec.validate()
            except TensorMapError:
                continue
            median_ms = _measure_in_child(
                _payload(
                    ptx_path=ptx_path,
                    kernel_name=kernel_name,
                    arch=arch,
                    shape=shape,
                    block_size=block_size,
                    num_args=num_args,
                    grid=grid,
                    block=block,
                    shared_memory_bytes=shared_memory_bytes,
                    arg_nbytes=arg_nbytes,
                    tensor_maps=resolved,
                    timing_trials=3,
                    warmup=1,
                )
            )
            if median_ms is not None:
                return ProbeResult(
                    arg_nbytes=arg_nbytes,
                    median_ms=median_ms,
                    tensor_maps=resolved,
                )
    return None


def _payload(
    *,
    ptx_path: str,
    kernel_name: str,
    arch: str,
    shape: str,
    block_size: int,
    num_args: int,
    grid: tuple[int, int, int],
    block: tuple[int, int, int],
    shared_memory_bytes: int,
    arg_nbytes: int,
    tensor_maps: dict[int, TensorMapSpec],
    timing_trials: int,
    warmup: int,
    acf_path: str | None = None,
) -> dict:
    return {
        "ptx_path": ptx_path,
        "kernel_name": kernel_name,
        "arch": arch,
        "shape": shape,
        "block_size": block_size,
        "num_args": num_args,
        "grid": list(grid),
        "block": list(block),
        "shared_memory_bytes": shared_memory_bytes,
        "arg_nbytes": arg_nbytes,
        "tensor_maps": {str(i): spec.to_dict() for i, spec in tensor_maps.items()},
        "timing_trials": timing_trials,
        "warmup": warmup,
        "acf_path": acf_path,
    }


def _measure_in_child(payload: dict, timeout_s: float = _PROBE_TIMEOUT_S) -> float | None:
    """Run one timed launch in a throwaway process; None if it did not survive."""
    debug = bool(os.environ.get("CIQ_DEBUG_OBJECTIVE"))
    try:
        completed = subprocess.run(
            [sys.executable, "-m", "xla_ciq.search.launch_probe", json.dumps(payload)],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        if debug:
            print(
                f"PROBE_TIMEOUT after {timeout_s}s for {payload.get('kernel_name')}",
                file=sys.stderr,
            )
        return None
    if completed.returncode != 0:
        if debug:
            detail = (completed.stderr or completed.stdout or "").strip()
            print(f"PROBE_CHILD_FAIL rc={completed.returncode}: {detail}", file=sys.stderr)
        return None
    for line in reversed(completed.stdout.splitlines()):
        if line.startswith("PROBE_OK "):
            return float(line.split()[1])
    return None


def measure_isolated_ms(
    *,
    ptx_path: str,
    kernel_name: str,
    arch: str,
    shape: str,
    block_size: int,
    num_args: int,
    grid: tuple[int, int, int],
    block: tuple[int, int, int],
    shared_memory_bytes: int,
    arg_nbytes: int,
    timing_trials: int,
    warmup: int,
    tensor_maps: dict[int, TensorMapSpec] | None = None,
    acf_path: str | None = None,
    timeout_s: float = _PROBE_TIMEOUT_S,
) -> float | None:
    """Time a candidate out of process so a faulting ACF cannot poison the search.

    Some ptxas control settings turn a working kernel into one that faults. In
    process that error is sticky and every later candidate fails too, which
    reads as "all objective functions failed" rather than one bad candidate.
    """
    return _measure_in_child(
        _payload(
            ptx_path=ptx_path,
            kernel_name=kernel_name,
            arch=arch,
            shape=shape,
            block_size=block_size,
            num_args=num_args,
            grid=grid,
            block=block,
            shared_memory_bytes=shared_memory_bytes,
            arg_nbytes=arg_nbytes,
            tensor_maps=tensor_maps or {},
            timing_trials=timing_trials,
            warmup=warmup,
            acf_path=acf_path,
        ),
        timeout_s=timeout_s,
    )


def _main(argv: list[str]) -> int:
    from pathlib import Path

    from xla_ciq.search.ptxas_launch import measure_kernel_ms

    request = json.loads(argv[1])
    tensor_maps = {
        int(index): TensorMapSpec.from_dict(payload)
        for index, payload in (request.get("tensor_maps") or {}).items()
    }
    acf_path = request.get("acf_path")
    acf_bytes = Path(acf_path).read_bytes() if acf_path else None
    median_ms = measure_kernel_ms(
        ptx_path=request["ptx_path"],
        kernel_name=request["kernel_name"],
        arch=request["arch"],
        shape=request["shape"],
        block_size=request["block_size"],
        num_args=request["num_args"],
        timing_trials=int(request.get("timing_trials", 3)),
        acf_bytes=acf_bytes,
        warmup=int(request.get("warmup", 1)),
        grid=tuple(request["grid"]),
        block=tuple(request["block"]),
        shared_memory_bytes=request["shared_memory_bytes"],
        arg_nbytes=request["arg_nbytes"],
        tensor_maps=tensor_maps,
    )
    print(f"PROBE_OK {median_ms}")
    return 0


if __name__ == "__main__":  # pragma: no cover - subprocess entry point
    try:
        raise SystemExit(_main(sys.argv))
    except Exception as exc:  # noqa: BLE001 - probe reports failure via exit code
        print(f"PROBE_FAIL {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
