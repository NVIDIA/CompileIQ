"""Implement `xla-ciq bundle evaluate` for an experimental bundle."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from xla_ciq.bundle import BundleError, get_entry_metadata, load_bundle
from xla_ciq.fingerprint import fingerprint
from xla_ciq.search.ptx_meta import (
    inspect_ptx_entry,
    resolve_block_size,
    validate_synthetic_launch_params,
)
from xla_ciq.search.ptxas_launch import (
    compile_ptx_to_cubin,
    measure_kernel_stats,
    ptxas_available,
    temp_artifact_path,
)
from xla_ciq.search.launch_probe import probe_launch
from xla_ciq.search.shape_launch import parse_shape_token
from xla_ciq.search.tensor_map_plan import resolve_tensor_maps
from xla_ciq.search.types import SearchError, UnlaunchableShapeError
from xla_ciq.shapes import ShapeParseError, parse_shapes
from xla_ciq.xla.dump import max_shape_nbytes
from xla_ciq.xla.launch_config import fallback_launch_config, resolve_launch_config
from xla_ciq.xla.tensor_map import TensorMapError, load_tensor_map_file

# Relative speedup agreement between recorded populate metadata and evaluate.
_REPRO_TOLERANCE = 0.05


@dataclass(frozen=True)
class EvaluateEntryResult:
    fingerprint: str
    content_id: str
    kernel_name: str
    shape: str
    ok: bool
    baseline_ms: float | None = None
    baseline_stdev_ms: float | None = None
    tuned_ms: float | None = None
    tuned_stdev_ms: float | None = None
    speedup: float | None = None
    recorded_speedup: float | None = None
    speedup_reproduced: bool | None = None
    error: str | None = None


def run_bundle_evaluate(
    *,
    bundle: Path,
    input_path: Path,
    kernel_name: str,
    shapes: str | None = None,
    fingerprints: tuple[str, ...] = (),
    arch: str = "sm_90a",
    cuda_version: str = "13.3",
    block_size: int = 256,
    timing_trials: int = 10,
    rounds: int = 5,
    mock: bool = True,
    as_json: bool = False,
    dump_dir: Path | None = None,
    min_arg_bytes: int = 0,
    tensor_map_path: Path | None = None,
) -> int:
    if not bundle.is_file():
        print(f"error: bundle not found: {bundle}", file=sys.stderr)
        return 2
    if not input_path.is_file():
        print(f"error: input not found: {input_path}", file=sys.stderr)
        return 2
    if not mock and input_path.suffix.lower() != ".ptx":
        print(
            "error: real evaluate currently supports .ptx only; pass --mock",
            file=sys.stderr,
        )
        return 2
    if not mock and not ptxas_available():
        print(
            "error: ptxas not found on PATH; install CUDA toolkit or pass --mock",
            file=sys.stderr,
        )
        return 2
    if not mock:
        print(
            "warning: --real uses synthetic zero-filled launch inputs and does not "
            "check kernel outputs; results are illustrative, not correctness-validated",
            file=sys.stderr,
        )
    if rounds < 1:
        print("error: --rounds must be >= 1", file=sys.stderr)
        return 2

    try:
        message = load_bundle(bundle)
    except BundleError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    try:
        selected = _select_entries(
            message,
            input_path=input_path,
            kernel_name=kernel_name,
            shapes=shapes,
            fingerprints=fingerprints,
            arch=arch,
            cuda_version=cuda_version,
        )
    except (ShapeParseError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if not selected:
        print("error: no matching bundle entries to evaluate", file=sys.stderr)
        return 2

    if not min_arg_bytes and dump_dir is not None:
        min_arg_bytes = max_shape_nbytes(dump_dir, kernel_name)

    tensor_map_overrides: dict = {}
    if tensor_map_path is not None:
        try:
            tensor_map_overrides = load_tensor_map_file(tensor_map_path).get(
                kernel_name, {}
            )
        except TensorMapError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2

    results: list[EvaluateEntryResult] = []
    for fp, cid, shape, recorded_speedup in selected:
        acf_bytes = bytes(message.control_files[cid].content)
        if mock:
            results.append(
                EvaluateEntryResult(
                    fingerprint=fp,
                    content_id=cid,
                    kernel_name=kernel_name,
                    shape=shape,
                    ok=True,
                    recorded_speedup=recorded_speedup,
                    error=None,
                )
            )
            continue
        try:
            try:
                entry_meta = inspect_ptx_entry(input_path.read_text(), kernel_name)
                validate_synthetic_launch_params(entry_meta)
                launch_block = resolve_block_size(entry_meta, block_size)
                parsed_shape = parse_shape_token(shape)
            except ValueError as exc:
                raise SearchError(str(exc)) from exc
            launch = resolve_launch_config(
                kernel_name=kernel_name,
                ptx_path=input_path,
                dump_dir=dump_dir,
                reqntid=entry_meta.reqntid,
            )
            if launch is None:
                launch = fallback_launch_config(
                    numel=parsed_shape.numel,
                    block_size=launch_block,
                )
            else:
                launch_block = launch.block_size
            plan = resolve_tensor_maps(
                meta=entry_meta,
                ptx_path=input_path,
                dump_dir=dump_dir,
                overrides=tensor_map_overrides,
            )
            if not plan.is_complete:
                raise UnlaunchableShapeError(plan.describe_gap(entry_meta))
            probe = probe_launch(
                ptx_path=str(input_path.resolve()),
                kernel_name=kernel_name,
                arch=arch,
                shape=shape,
                block_size=launch_block,
                num_args=entry_meta.param_count,
                grid=launch.grid,
                block=launch.block,
                shared_memory_bytes=launch.shared_memory_bytes,
                shape_nbytes=parsed_shape.nbytes,
                footprint_nbytes=min_arg_bytes,
                tensor_maps=plan.specs,
            )
            if probe is None:
                raise UnlaunchableShapeError(
                    f"no safe CUDA launch found for shape={shape}"
                )
            launch_kwargs = {
                "grid": launch.grid,
                "block": launch.block,
                "shared_memory_bytes": launch.shared_memory_bytes,
                "arg_nbytes": probe.arg_nbytes,
                "tensor_maps": probe.tensor_maps,
            }
            baseline = measure_kernel_stats(
                ptx_path=str(input_path.resolve()),
                kernel_name=kernel_name,
                arch=arch,
                shape=shape,
                block_size=launch_block,
                num_args=entry_meta.param_count,
                timing_trials=timing_trials,
                acf_bytes=None,
                warmup=5,
                rounds=rounds,
                **launch_kwargs,
            )
            tuned = measure_kernel_stats(
                ptx_path=str(input_path.resolve()),
                kernel_name=kernel_name,
                arch=arch,
                shape=shape,
                block_size=launch_block,
                num_args=entry_meta.param_count,
                timing_trials=timing_trials,
                acf_bytes=acf_bytes,
                warmup=5,
                rounds=rounds,
                **launch_kwargs,
            )
            speedup = (
                baseline.median_ms / tuned.median_ms
                if tuned.median_ms and tuned.median_ms > 0
                else None
            )
            reproduced = None
            if speedup is not None and recorded_speedup is not None and recorded_speedup > 0:
                reproduced = (
                    abs(speedup - recorded_speedup) / recorded_speedup <= _REPRO_TOLERANCE
                )
            results.append(
                EvaluateEntryResult(
                    fingerprint=fp,
                    content_id=cid,
                    kernel_name=kernel_name,
                    shape=shape,
                    ok=True,
                    baseline_ms=baseline.median_ms,
                    baseline_stdev_ms=baseline.stdev_ms,
                    tuned_ms=tuned.median_ms,
                    tuned_stdev_ms=tuned.stdev_ms,
                    speedup=speedup,
                    recorded_speedup=recorded_speedup,
                    speedup_reproduced=reproduced,
                )
            )
        except SearchError as exc:
            apply_error = str(exc)
            try:
                cubin = temp_artifact_path(".cubin")
                compile_ptx_to_cubin(
                    ptx_path=str(input_path.resolve()),
                    arch=arch,
                    cubin_path=cubin,
                    acf_bytes=acf_bytes,
                )
                Path(cubin).unlink(missing_ok=True)
            except Exception as apply_exc:
                apply_error = str(apply_exc)
            results.append(
                EvaluateEntryResult(
                    fingerprint=fp,
                    content_id=cid,
                    kernel_name=kernel_name,
                    shape=shape,
                    ok=False,
                    recorded_speedup=recorded_speedup,
                    error=apply_error,
                )
            )

    payload = {
        "path": str(bundle),
        "input": str(input_path),
        "kernel_name": kernel_name,
        "mock": mock,
        "rounds": rounds,
        "timing_trials": timing_trials,
        "entries": [asdict(r) for r in results],
        "passed": sum(1 for r in results if r.ok),
        "failed": sum(1 for r in results if not r.ok),
    }

    if as_json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"bundle: {bundle}")
        print(f"kernel: {kernel_name}")
        print(
            f"evaluated: {len(results)} passed={payload['passed']} "
            f"failed={payload['failed']} rounds={rounds}"
        )
        for entry in results:
            status = "ok" if entry.ok else "FAIL"
            line = (
                f"  [{status}] shape={entry.shape} fingerprint={entry.fingerprint} "
                f"content_id={entry.content_id}"
            )
            if entry.ok and not mock:
                line += (
                    f" baseline_ms={entry.baseline_ms}±{entry.baseline_stdev_ms} "
                    f"tuned_ms={entry.tuned_ms}±{entry.tuned_stdev_ms} "
                    f"speedup={entry.speedup}"
                )
            if entry.recorded_speedup is not None:
                line += f" recorded_speedup={entry.recorded_speedup}"
            if entry.speedup_reproduced is not None:
                line += f" reproduced={entry.speedup_reproduced}"
            if entry.error:
                line += f" error={entry.error}"
            print(line)

    return 0 if payload["failed"] == 0 else 1


def _select_entries(
    message,
    *,
    input_path: Path,
    kernel_name: str,
    shapes: str | None,
    fingerprints: tuple[str, ...],
    arch: str,
    cuda_version: str,
) -> list[tuple[str, str, str, float | None]]:
    """Return (fingerprint, content_id, shape, recorded_speedup) rows."""
    fp_filter = set(fingerprints) if fingerprints else None
    source_bytes = input_path.read_bytes()
    selected: list[tuple[str, str, str, float | None]] = []

    if shapes:
        for shape in parse_shapes(shapes):
            fp = fingerprint(
                cuda_version=cuda_version,
                arch=arch,
                source_bytes=source_bytes,
                kernel_name=kernel_name,
                shape=shape,
            )
            if fp_filter is not None and fp not in fp_filter:
                continue
            cid = message.fingerprint_to_control_file_id.get(fp)
            if cid is None:
                raise ValueError(
                    f"no bundle entry for kernel={kernel_name!r} shape={shape!r} "
                    f"(fingerprint={fp})"
                )
            meta = get_entry_metadata(message, fp)
            recorded = meta.speedup if meta is not None else None
            selected.append((fp, cid, shape, recorded))
        return selected

    for fp, cid in message.fingerprint_to_control_file_id.items():
        if fp_filter is not None and fp not in fp_filter:
            continue
        meta = get_entry_metadata(message, fp)
        if meta is None:
            continue
        if meta.kernel_name != kernel_name:
            continue
        if not meta.shape:
            continue
        expected = fingerprint(
            cuda_version=cuda_version,
            arch=arch,
            source_bytes=source_bytes,
            kernel_name=kernel_name,
            shape=meta.shape,
        )
        if expected != fp:
            raise ValueError(
                f"fingerprint mismatch for shape={meta.shape}; bundle has {fp}, "
                f"recomputed {expected}; refusing to select the ACF"
            )
        selected.append((fp, cid, meta.shape, meta.speedup))

    if not selected and fp_filter:
        for fp in fp_filter:
            cid = message.fingerprint_to_control_file_id.get(fp)
            if cid is None:
                raise ValueError(f"fingerprint not in bundle: {fp}")
            meta = get_entry_metadata(message, fp)
            shape = meta.shape if meta is not None else ""
            if not shape:
                raise ValueError(
                    f"fingerprint {fp} has no recorded shape; pass --shapes"
                )
            selected.append(
                (fp, cid, shape, meta.speedup if meta is not None else None)
            )

    if not selected:
        raise ValueError(
            "no entries matched; pass --shapes <...> or ensure the bundle has "
            f"metadata for kernel={kernel_name!r}"
        )
    return selected
