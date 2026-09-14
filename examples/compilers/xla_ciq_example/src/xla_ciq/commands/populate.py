"""Populate an experimental ACF bundle through the CompileIQ adapter."""

from __future__ import annotations

import sys
from pathlib import Path

from xla_ciq.bundle import (
    BundleError,
    EntryMetadata,
    append_acf,
    load_bundle,
    new_bundle,
    remove_acf,
    save_bundle,
)
from xla_ciq.fingerprint import content_id, fingerprint
from xla_ciq.search.factory import get_search_engine
from xla_ciq.search.types import SearchError, SearchRequest, UnlaunchableShapeError
from xla_ciq.shapes import ShapeParseError, is_all_shapes, parse_shapes
from xla_ciq.xla.dump import DumpInspectError, inspect_dump

SUPPORTED_SUFFIXES = {".ptx", ".mlir"}


def _beats_baseline(baseline_ms: float | None, best_ms: float | None) -> bool:
    """True when re-measured tuned runtime is strictly faster than baseline."""
    if baseline_ms is None or best_ms is None:
        return True  # no timing → keep ACF (e.g. --mock)
    return float(best_ms) < float(baseline_ms)


def run_populate(
    *,
    bundle: Path,
    kernel_name: str,
    input_path: Path,
    shapes: str,
    dump_dir: Path | None = None,
    arch: str = "sm_90a",
    cuda_version: str = "13.3",
    replace: bool = False,
    mock: bool = True,
    generations: int = 5,
    pool_size: int = 32,
    cull_size: int | None = None,
    mutate_rate: float = 0.25,
    normalize: bool = False,
    init_with_true_random_threshold: float = 0.9,
    enable_large_fail_pool: bool = True,
    block_size: int = 256,
    timing_trials: int = 10,
    task_timeout: float = 30.0,
    num_workers: int = 1,
    search_space_tag: str = "search-spaces-2026.08.14",
    keep_slower_acf: bool = False,
    min_arg_bytes: int = 0,
    tensor_map_path: Path | None = None,
    sweep_tensor_map_tiles: bool = False,
) -> int:
    try:
        parsed_shapes = _resolve_shapes(
            shapes,
            dump_dir=dump_dir,
            kernel_name=kernel_name,
        )
    except (ShapeParseError, DumpInspectError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if not input_path.is_file():
        print(f"error: input not found: {input_path}", file=sys.stderr)
        return 2
    if input_path.suffix.lower() not in SUPPORTED_SUFFIXES:
        print(
            f"error: unsupported input format {input_path.suffix!r}; "
            f"expected one of {sorted(SUPPORTED_SUFFIXES)}",
            file=sys.stderr,
        )
        return 2
    if not mock and input_path.suffix.lower() != ".ptx":
        print(
            "error: real CompileIQ mode currently supports .ptx only; "
            "pass --mock or provide PTX",
            file=sys.stderr,
        )
        return 2
    if not mock:
        print(
            "warning: --real uses synthetic zero-filled launch inputs and does not "
            "check kernel outputs; results are illustrative, not correctness-validated",
            file=sys.stderr,
        )

    source_bytes = input_path.read_bytes()
    engine = get_search_engine(mock=mock)

    try:
        message = load_bundle(bundle) if bundle.exists() else new_bundle()
        results: list[tuple[str, str, str, str]] = []
        mutated = False

        for shape in parsed_shapes:
            fp = fingerprint(
                cuda_version=cuda_version,
                arch=arch,
                source_bytes=source_bytes,
                kernel_name=kernel_name,
                shape=shape,
            )
            existing_cid = message.fingerprint_to_control_file_id.get(fp)
            if existing_cid is not None and not replace:
                print(
                    f"ACF already exists for shape={shape}; search skipped. "
                    f"Re-run with --replace to overwrite.",
                    file=sys.stderr,
                )
                results.append(("preserved", shape, fp, existing_cid))
                continue

            request = SearchRequest(
                kernel_name=kernel_name,
                input_path=input_path,
                source_bytes=source_bytes,
                shape=shape,
                arch=arch,
                cuda_version=cuda_version,
                generations=generations,
                pool_size=pool_size,
                cull_size=cull_size,
                mutate_rate=mutate_rate,
                normalize=normalize,
                init_with_true_random_threshold=init_with_true_random_threshold,
                enable_large_fail_pool=enable_large_fail_pool,
                block_size=block_size,
                timing_trials=timing_trials,
                task_timeout=task_timeout,
                num_workers=num_workers,
                search_space_tag=search_space_tag,
                dump_dir=dump_dir,
                min_arg_bytes=min_arg_bytes,
                tensor_map_path=tensor_map_path,
                sweep_tensor_map_tiles=sweep_tensor_map_tiles,
            )
            try:
                result = engine.run(request)
            except UnlaunchableShapeError as exc:
                # Not a search failure: this shape cannot be timed safely, so
                # leave it unmapped. Consumer fallback behavior is not demonstrated.
                print(f"skipped_unlaunchable: shape={shape} {exc}", file=sys.stderr)
                results.append(("skipped_unlaunchable", shape, fp, "-"))
                continue
            except SearchError as exc:
                print(f"error: {exc}", file=sys.stderr)
                return 1
            for note in result.notes:
                print(note, file=sys.stderr)
            if result.baseline_ms is not None or result.score is not None:
                print(
                    f"timing: shape={shape} "
                    f"baseline_ms={result.baseline_ms} "
                    f"best_ms={result.score}",
                    file=sys.stderr,
                )

            cid = content_id(result.acf_bytes)
            speedup = None
            if (
                result.baseline_ms is not None
                and result.score is not None
                and float(result.score) > 0
            ):
                speedup = float(result.baseline_ms) / float(result.score)

            if not keep_slower_acf and not _beats_baseline(
                result.baseline_ms, result.score
            ):
                # Leave the producer mapping absent for this fingerprint. Any
                # consumer fallback behavior is outside this example.
                removed = remove_acf(message, fp)
                print(
                    f"no_gain: shape={shape} baseline_ms={result.baseline_ms} "
                    f"best_ms={result.score} speedup={speedup}; "
                    "ACF not stored (mapping omitted). "
                    "Pass --keep-slower-acf to force-store.",
                    file=sys.stderr,
                )
                action = "removed_no_gain" if removed else "skipped_no_gain"
                results.append((action, shape, fp, "-"))
                if removed:
                    mutated = True
                continue

            action = append_acf(
                message,
                fingerprint=fp,
                acf_bytes=result.acf_bytes,
                content_id=cid,
                replace=replace,
                metadata=EntryMetadata(
                    kernel_name=kernel_name,
                    shape=shape,
                    arch=arch,
                    cuda_version=cuda_version,
                    baseline_ms=result.baseline_ms,
                    best_ms=result.score,
                    speedup=speedup,
                ),
            )
            results.append((action, shape, fp, cid))
            mutated = True

        if mutated:
            save_bundle(bundle, message)
        elif not bundle.exists():
            save_bundle(bundle, message)
    except BundleError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    for action, shape, fp, cid in results:
        print(
            f"{action}: shape={shape} fingerprint={fp} content_id={cid} "
            f"mappings={len(message.fingerprint_to_control_file_id)} "
            f"control_files={len(message.control_files)} path={bundle}"
        )
    return 0


def _resolve_shapes(
    shapes: str,
    *,
    dump_dir: Path | None,
    kernel_name: str,
) -> list[str]:
    if is_all_shapes(shapes):
        if dump_dir is None:
            raise ShapeParseError(
                "--dump-dir is required when --shapes all "
                "(shapes are taken from the XLA dump for this kernel)"
            )
        inspection = inspect_dump(dump_dir, kernel_name=kernel_name)
        if not inspection.kernels:
            raise ShapeParseError(
                f"no kernels found in XLA dump {dump_dir} for {kernel_name!r}"
            )
        resolved = list(inspection.kernels[0].shapes)
        if not resolved:
            raise ShapeParseError(
                f"no shapes found in XLA dump {dump_dir} for kernel {kernel_name!r}"
            )
        print(
            f"resolved --shapes all for kernel={kernel_name} from {dump_dir}: "
            f"{','.join(resolved)}",
            file=sys.stderr,
        )
        return resolved
    return parse_shapes(shapes)
