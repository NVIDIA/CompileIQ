"""Adapter from the experiment search interface to CompileIQ PTXAS search."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from xla_ciq.search.cuda_runtime import CudaError
from xla_ciq.search.ptx_meta import (
    inspect_ptx_entry,
    resolve_block_size,
    validate_synthetic_launch_params,
)
from xla_ciq.search.ptxas_launch import (
    measure_kernel_ms,
    ptxas_available,
    temp_artifact_path,
    time_cubin,
)
from xla_ciq.search.launch_probe import measure_isolated_ms, probe_launch
from xla_ciq.search.shape_launch import parse_shape_token
from xla_ciq.search.tensor_map_plan import resolve_tensor_maps
from xla_ciq.search.tensor_map_search import (
    apply_tile_config,
    build_tile_search_space,
    default_sweep_indices,
    describe_tile_config,
    unpack_search_config,
)
from xla_ciq.search.types import (
    SearchError,
    SearchRequest,
    SearchResult,
    UnlaunchableShapeError,
)
from xla_ciq.xla.dump import max_shape_nbytes
from xla_ciq.xla.launch_config import fallback_launch_config, resolve_launch_config
from xla_ciq.xla.tensor_map import (
    TensorMapError,
    TensorMapSpec,
    load_tensor_map_file,
)

# Avoid CompileIQ's tag="latest" path, which lists GitHub releases and is often
# rate-limited for unauthenticated clients. Override with --search-space-tag.
DEFAULT_SEARCH_SPACE_TAG = "search-spaces-2026.08.14"


@dataclass(frozen=True)
class RuntimeObjective:
    """Picklable CompileIQ objective (multiprocess workers cannot pickle nested defs)."""

    ptx_path: str
    kernel_name: str
    arch: str
    shape: str
    block_size: int
    num_args: int
    timing_trials: int
    grid: tuple[int, int, int]
    block: tuple[int, int, int]
    shared_memory_bytes: int
    arg_nbytes: int
    # Kept as pairs rather than a dict so the objective stays hashable and
    # picklable for CompileIQ's multiprocess workers.
    tensor_maps: tuple[tuple[int, TensorMapSpec], ...] = ()
    isolate: bool = False
    # Match CompileIQ's own per-task budget so a hung candidate is abandoned at
    # the same time the search gives up on it.
    isolate_timeout_s: float = 30.0

    @property
    def tensor_map_specs(self) -> dict[int, TensorMapSpec]:
        return dict(self.tensor_maps)

    def __call__(self, config) -> float:
        return _runtime_objective(
            config=config,
            ptx_path=self.ptx_path,
            kernel_name=self.kernel_name,
            arch=self.arch,
            shape=self.shape,
            block_size=self.block_size,
            num_args=self.num_args,
            timing_trials=self.timing_trials,
            grid=self.grid,
            block=self.block,
            shared_memory_bytes=self.shared_memory_bytes,
            arg_nbytes=self.arg_nbytes,
            tensor_maps=self.tensor_map_specs,
            isolate=self.isolate,
            isolate_timeout_s=self.isolate_timeout_s,
        )


class CompileIqRuntimeEngine:
    """Implement the experiment search interface through CompileIQ 1.0.3."""

    def run(self, request: SearchRequest) -> SearchResult:
        self._require_deps()
        if request.input_path.suffix.lower() != ".ptx":
            raise SearchError(
                "real CompileIQ runtime search currently supports PTX only; "
                "use --mock for other inputs or convert to PTX"
            )

        from compileiq.ciq import Search
        from compileiq.search_spaces.compilers import PtxasSearchSpace
        from compileiq.types import SearchConfiguration, WorkerTypes

        cuda_version = request.cuda_version or detect_ptxas_cuda_version()
        search_space_tag = request.search_space_tag or DEFAULT_SEARCH_SPACE_TAG
        ptx_path = str(request.input_path.resolve())
        try:
            meta = inspect_ptx_entry(request.input_path.read_text(), request.kernel_name)
        except ValueError as exc:
            raise SearchError(str(exc)) from exc
        if meta.param_count < 1:
            raise SearchError(
                f"kernel {request.kernel_name!r} has no .param entries; "
                "cannot build a CUDA launch"
            )
        try:
            validate_synthetic_launch_params(meta)
        except ValueError as exc:
            raise SearchError(str(exc)) from exc
        try:
            block_size = resolve_block_size(meta, request.block_size)
        except ValueError as exc:
            raise SearchError(str(exc)) from exc

        try:
            parsed_shape = parse_shape_token(request.shape)
        except ValueError as exc:
            raise SearchError(str(exc)) from exc

        dump_dir = Path(request.dump_dir) if request.dump_dir else None
        launch = resolve_launch_config(
            kernel_name=request.kernel_name,
            ptx_path=Path(ptx_path),
            dump_dir=dump_dir,
            reqntid=meta.reqntid,
        )
        if launch is None:
            launch = fallback_launch_config(
                numel=parsed_shape.numel,
                block_size=block_size,
            )
            launch_note = f"REAL: launch={launch.grid}/{launch.block} (heuristic fallback)"
        else:
            block_size = launch.block_size
            launch_note = (
                f"REAL: launch grid={launch.grid} block={launch.block} "
                f"shared_mem={launch.shared_memory_bytes} from {launch.source}"
            )

        footprint_nbytes = request.min_arg_bytes
        if not footprint_nbytes and dump_dir is not None:
            footprint_nbytes = max_shape_nbytes(dump_dir, request.kernel_name)

        plan = _resolve_tensor_map_plan(request=request, meta=meta, dump_dir=dump_dir)
        if not plan.is_complete:
            raise UnlaunchableShapeError(plan.describe_gap(meta))

        # Probe out of process: a faulting launch poisons this process for good,
        # so the buffer size is settled before any in-process timing happens.
        probe = probe_launch(
            ptx_path=ptx_path,
            kernel_name=request.kernel_name,
            arch=request.arch,
            shape=request.shape,
            block_size=block_size,
            num_args=meta.param_count,
            grid=launch.grid,
            block=launch.block,
            shared_memory_bytes=launch.shared_memory_bytes,
            shape_nbytes=parsed_shape.nbytes,
            footprint_nbytes=footprint_nbytes,
            tensor_maps=plan.specs,
        )
        if probe is None:
            raise UnlaunchableShapeError(
                f"no safe CUDA launch found for {request.kernel_name!r} "
                f"shape={request.shape} (grid={launch.grid} block={launch.block}); "
                "the kernel reads outside every buffer size tried"
            )
        arg_nbytes = probe.arg_nbytes
        # The probe settles the swizzle mode, which PTX does not record.
        tensor_maps = probe.tensor_maps

        notes = [
            "REAL: CompileIQ PTXAS runtime search",
            "REAL: illustrative synthetic timing only; kernel outputs are not checked",
            "REAL: CompileIQ isolated worker selected for every candidate",
            f"REAL: kernel={request.kernel_name} shape={request.shape}",
            f"REAL: arch={request.arch} cuda_version={cuda_version}",
            f"REAL: search_space_tag={search_space_tag}",
            f"REAL: num_args={meta.param_count} arg_nbytes={arg_nbytes} "
            f"(shape_nbytes={parsed_shape.nbytes} footprint={footprint_nbytes})",
            launch_note,
            f"REAL: generations={request.generations} pool_size={request.pool_size}",
            *_tensor_map_notes(meta, tensor_maps),
            f"REAL: cull_size={request.cull_size} mutate_rate={request.mutate_rate}",
            f"REAL: CompileIQ per-objective task_timeout_s={request.task_timeout}",
        ]

        objective = RuntimeObjective(
            ptx_path=ptx_path,
            kernel_name=request.kernel_name,
            arch=request.arch,
            shape=request.shape,
            block_size=block_size,
            num_args=meta.param_count,
            timing_trials=request.timing_trials,
            grid=launch.grid,
            block=launch.block,
            shared_memory_bytes=launch.shared_memory_bytes,
            arg_nbytes=arg_nbytes,
            tensor_maps=tuple(sorted(tensor_maps.items())),
            # Preflight and winner re-measurement happen outside CompileIQ's
            # worker too, so keep the launch itself isolated on every path.
            isolate=True,
            isolate_timeout_s=request.task_timeout,
        )
        if objective.isolate:
            notes.append(
                "REAL: launch timing isolated for preflight, candidates, and "
                "winner re-measurement"
            )
        baseline_ms = _preflight_launch(objective)
        notes.append(f"REAL: baseline_ms={baseline_ms}")

        search_kwargs: dict = {
            "problem_type": "min",
            "generations": request.generations,
            "pool_size": request.pool_size,
            "num_objectives": 1,
            "mutate_rate": request.mutate_rate,
            "normalize": request.normalize,
            "init_with_true_random_threshold": request.init_with_true_random_threshold,
            "enable_large_fail_pool": request.enable_large_fail_pool,
        }
        if request.cull_size is not None:
            search_kwargs["cull_size"] = request.cull_size
        search_config = SearchConfiguration(**search_kwargs)

        ptxas_space = PtxasSearchSpace(version=cuda_version, tag=search_space_tag)
        search_space: object = ptxas_space
        sweep_indices: tuple[int, ...] = ()
        if request.sweep_tensor_map_tiles:
            if not tensor_maps:
                raise SearchError(
                    "--sweep-tensor-map-tiles requires at least one tensor map "
                    "layout (from the dump or --tensor-maps)"
                )
            try:
                sweep_indices = default_sweep_indices(tensor_maps)
                tile_space = build_tile_search_space(
                    tensor_maps, sweep_indices=sweep_indices
                )
            except TensorMapError as exc:
                raise SearchError(str(exc)) from exc
            # Multi-config: core samples ACF + tiles together; the objective
            # receives [acf_hex, tile_dict] in this order.
            search_space = [ptxas_space, tile_space]
            notes.append(
                "REAL: sweeping tensor map tiles jointly with PTXAS controls "
                f"for param(s) {', '.join(map(str, sweep_indices))}"
            )
            notes.append(
                "REAL: tile init pinned to the seed layout "
                f"({request.init_with_true_random_threshold:.0%} of the first generation)"
            )

        search_kwargs_extra: dict = {}
        if request.sweep_tensor_map_tiles:
            # A rigid kernel rejects most non-seed tiles, so an unlucky first
            # generation must not abort a search that can still fall back to the
            # seed layout.
            search_kwargs_extra["exit_on_failure"] = False

        try:
            tuner = Search(
                objective_function=objective,
                search_space=search_space,
                search_config=search_config,
                worker_type=WorkerTypes.ISOLATED,
                **search_kwargs_extra,
            )
        except Exception as exc:
            raise SearchError(_format_search_space_error(exc)) from exc

        try:
            results = _start_tuner(tuner, request)
        except Exception as exc:
            raise SearchError(
                _format_search_failure(exc, tensor_maps=tensor_maps)
            ) from exc

        best = results.get_best_result()
        if best is None:
            raise SearchError("CompileIQ search returned no best result")
        params = best.get("params") if isinstance(best, dict) else best["params"]
        search_score = None
        if isinstance(best, dict):
            search_score = best.get("score_1")
        acf_payload, tile_payload = unpack_search_config(params)
        acf_bytes = _params_to_acf_bytes(acf_payload)
        notes.append(f"REAL: search_best_score_ms={search_score}")
        winning_maps = tensor_maps
        if tile_payload:
            try:
                winning_maps = apply_tile_config(tensor_maps, tile_payload)
            except TensorMapError as exc:
                raise SearchError(
                    f"winning tile config is invalid: {exc}"
                ) from exc
            notes.append(
                f"REAL: winning_tiles {describe_tile_config(tensor_maps, tile_payload)}"
            )
            for index in sorted(winning_maps):
                spec = winning_maps[index]
                seed = tensor_maps.get(index)
                # Only the layout matters here; the spec's source string always
                # changes once a tile config has been applied.
                if seed is not None and (
                    spec.box == seed.box and spec.swizzle_bytes == seed.swizzle_bytes
                ):
                    continue
                notes.append(
                    f"REAL: swept_tensor_map[{index}] {spec.dtype} "
                    f"dims={spec.dims} box={spec.box} swizzle={spec.swizzle_bytes}B"
                )

        if request.sweep_tensor_map_tiles:
            _require_seed_equivalent_winner(
                seed_maps=tensor_maps,
                winning_maps=winning_maps,
                sweep_indices=sweep_indices,
            )

        # Re-measure the winner alone. Search scores are noisy under multi-worker
        # GPU contention and short kernels, so do not trust score_1 as the reported
        # best/speedup without a clean timing pass. This does not check outputs.
        try:
            remeasured_ms = _remeasure_best(
                objective=objective,
                acf_bytes=acf_bytes,
                timing_trials=max(10, request.timing_trials),
                tensor_maps=winning_maps,
            )
        except SearchError as exc:
            raise SearchError(
                f"winning ACF failed re-measurement timing: {exc}"
            ) from exc
        notes.append(f"REAL: remeasured_best_ms={remeasured_ms}")
        if search_score is not None and float(search_score) > 0:
            drift = abs(remeasured_ms - float(search_score)) / float(search_score)
            notes.append(f"REAL: search_vs_remeasured_rel_diff={drift:.4f}")
            if drift > 0.05:
                notes.append(
                    "REAL: warning: search score did not reproduce under clean "
                    "re-measurement (likely timing noise / GPU contention); "
                    "bundle metadata uses remeasured_best_ms"
                )
        if baseline_ms > 0:
            notes.append(f"REAL: speedup={baseline_ms / remeasured_ms:.4f}x")
        return SearchResult(
            acf_bytes=acf_bytes,
            notes=notes,
            score=remeasured_ms,
            baseline_ms=baseline_ms,
        )

    @staticmethod
    def _require_deps() -> None:
        try:
            import compileiq  # noqa: F401
        except ImportError as exc:
            raise SearchError(
                "compileiq is not installed; pip install compileiq "
                "or pass --mock"
            ) from exc
        try:
            from cuda.bindings import driver as _cuda  # noqa: F401
        except ImportError as exc:
            raise SearchError(
                "cuda-python is not installed; pip install cuda-python "
                "or pass --mock"
            ) from exc
        if not ptxas_available():
            raise SearchError("ptxas not found on PATH; install CUDA toolkit or pass --mock")


def _require_seed_equivalent_winner(
    *,
    seed_maps: dict[int, TensorMapSpec],
    winning_maps: dict[int, TensorMapSpec],
    sweep_indices: tuple[int, ...],
) -> None:
    """Reject a winner whose swept TMA layout cannot be represented in the bundle."""
    differences: list[str] = []
    indices = sorted(set(seed_maps) | set(winning_maps) | set(sweep_indices))
    for index in indices:
        seed = seed_maps.get(index)
        winner = winning_maps.get(index)
        if seed is None or winner is None:
            differences.append(
                f"param[{index}] seed={_tile_settings(seed)} "
                f"winner={_tile_settings(winner)}"
            )
            continue
        if winner.box != seed.box or winner.swizzle_bytes != seed.swizzle_bytes:
            differences.append(
                f"param[{index}] seed={_tile_settings(seed)} "
                f"winner={_tile_settings(winner)}"
            )

    if differences:
        raise SearchError(
            "winning TMA tile settings differ from the seed and this experiment "
            "cannot serialize those settings: "
            + "; ".join(differences)
            + ". Bundle not modified."
        )


def _start_tuner(tuner, request: SearchRequest):
    """Start CompileIQ directly with its per-objective timeout."""
    return tuner.start(
        num_workers=request.num_workers,
        task_timeout=request.task_timeout,
    )


def _tile_settings(spec: TensorMapSpec | None) -> str:
    if spec is None:
        return "<missing>"
    return f"box={spec.box},swizzle={spec.swizzle_bytes}B"


def _resolve_tensor_map_plan(*, request: SearchRequest, meta, dump_dir: Path | None):
    """Descriptor layouts for this kernel, from an explicit file or the dump."""
    overrides: dict[int, TensorMapSpec] = {}
    if request.tensor_map_path is not None:
        try:
            by_kernel = load_tensor_map_file(Path(request.tensor_map_path))
        except TensorMapError as exc:
            raise SearchError(str(exc)) from exc
        overrides = by_kernel.get(request.kernel_name, {})
    try:
        return resolve_tensor_maps(
            meta=meta,
            ptx_path=Path(request.input_path),
            dump_dir=dump_dir,
            overrides=overrides,
        )
    except TensorMapError as exc:
        raise SearchError(str(exc)) from exc


def _tensor_map_notes(meta, tensor_maps: dict[int, TensorMapSpec]) -> list[str]:
    if not tensor_maps:
        return []
    notes = [f"REAL: tensor_maps={len(tensor_maps)} (TMA descriptor arguments)"]
    for index in sorted(tensor_maps):
        spec = tensor_maps[index]
        name = meta.params[index].name if index < len(meta.params) else f"param_{index}"
        notes.append(
            f"REAL: tensor_map[{index}] {name} {spec.dtype} dims={spec.dims} "
            f"box={spec.box} swizzle={spec.swizzle_bytes}B from {spec.source}"
        )
    return notes


def _remeasure_best(
    *,
    objective: RuntimeObjective,
    acf_bytes: bytes,
    timing_trials: int,
    tensor_maps: dict[int, TensorMapSpec] | None = None,
) -> float:
    """Re-time the winning ACF on its own, away from search-time GPU contention."""
    maps = tensor_maps if tensor_maps is not None else objective.tensor_map_specs
    common = dict(
        ptx_path=objective.ptx_path,
        kernel_name=objective.kernel_name,
        arch=objective.arch,
        shape=objective.shape,
        block_size=objective.block_size,
        num_args=objective.num_args,
        grid=objective.grid,
        block=objective.block,
        shared_memory_bytes=objective.shared_memory_bytes,
        arg_nbytes=objective.arg_nbytes,
        tensor_maps=maps,
        timing_trials=timing_trials,
    )
    if not objective.isolate:
        return measure_kernel_ms(acf_bytes=acf_bytes, warmup=5, **common)

    acf_path = temp_artifact_path(".acf")
    try:
        Path(acf_path).write_bytes(acf_bytes)
        median_ms = measure_isolated_ms(warmup=5, acf_path=acf_path, **common)
    finally:
        if os.path.exists(acf_path):
            os.remove(acf_path)
    if median_ms is None:
        raise SearchError("isolated re-measurement launch did not survive")
    return median_ms


def _preflight_launch(objective: RuntimeObjective) -> float:
    """Compile PTX without ACF and time one launch; return baseline median ms."""
    if objective.isolate:
        median_ms = measure_isolated_ms(
            ptx_path=objective.ptx_path,
            kernel_name=objective.kernel_name,
            arch=objective.arch,
            shape=objective.shape,
            block_size=objective.block_size,
            num_args=objective.num_args,
            grid=objective.grid,
            block=objective.block,
            shared_memory_bytes=objective.shared_memory_bytes,
            arg_nbytes=objective.arg_nbytes,
            timing_trials=max(3, objective.timing_trials),
            warmup=2,
            tensor_maps=objective.tensor_map_specs,
        )
        if median_ms is None:
            raise SearchError(
                f"preflight CUDA launch failed for {objective.kernel_name!r}"
            )
        return median_ms
    try:
        return measure_kernel_ms(
            ptx_path=objective.ptx_path,
            kernel_name=objective.kernel_name,
            arch=objective.arch,
            shape=objective.shape,
            block_size=objective.block_size,
            num_args=objective.num_args,
            timing_trials=max(3, objective.timing_trials),
            acf_bytes=None,
            warmup=2,
            grid=objective.grid,
            block=objective.block,
            shared_memory_bytes=objective.shared_memory_bytes,
            arg_nbytes=objective.arg_nbytes,
            tensor_maps=objective.tensor_map_specs,
        )
    except SearchError:
        raise
    except Exception as exc:
        raise SearchError(
            f"preflight CUDA launch failed for {objective.kernel_name!r}: {exc}"
        ) from exc


def _format_search_failure(
    exc: BaseException, *, tensor_maps: dict[int, TensorMapSpec]
) -> str:
    """Explain a search failure, naming descriptor layouts as the likely cause.

    The baseline launch already succeeded by this point, so every candidate
    failing means the ACFs are producing code this launch cannot survive. For a
    descriptor kernel the usual reason is a tile that does not match what the
    kernel expects in shared memory.
    """
    text = f"CompileIQ search failed: {exc}"
    if tensor_maps and "objective functions failed" in str(exc).lower():
        inferred = sorted(
            index
            for index, spec in tensor_maps.items()
            if not spec.source.startswith("file:") and spec.source != "explicit"
        )
        detail = (
            "every candidate failed to launch while the untuned baseline "
            "succeeded, which usually means a tensor map tile does not match the "
            "kernel's shared-memory layout. Re-run with CIQ_DEBUG_OBJECTIVE=1 to "
            "see the per-candidate CUDA error"
        )
        if inferred:
            detail += (
                f"; the layout for parameter(s) {', '.join(map(str, inferred))} was "
                "inferred from the dump, so pass an explicit layout with --tensor-maps "
                "or search legal boxes with --sweep-tensor-map-tiles"
            )
        return f"{text} ({detail})"
    return text


def _format_search_space_error(exc: BaseException) -> str:
    text = str(exc)
    lower = text.lower()
    if "rate limit" in lower or "403" in text:
        return (
            f"CompileIQ search-space resolve failed (GitHub rate limit): {exc}. "
            "Retry later, pin --search-space-tag to a known release "
            f"(default {DEFAULT_SEARCH_SPACE_TAG}), set CIQ_SEARCH_SPACES_DIR to a "
            "local mirror with manifest.json + .bin assets, or authenticate GitHub "
            "API access for the compileiq resolver."
        )
    return f"CompileIQ search-space resolve failed: {exc}"


def detect_ptxas_cuda_version() -> str:
    result = subprocess.run(
        ["ptxas", "--version"],
        capture_output=True,
        text=True,
        check=True,
    )
    match = re.search(r"release (\d+\.\d+)", result.stdout + result.stderr)
    if not match:
        raise SearchError("could not parse CUDA version from ptxas --version")
    return match.group(1)


def _params_to_acf_bytes(params) -> bytes:
    if isinstance(params, list):
        if not params:
            raise SearchError("empty multi-config search params")
        params = params[0]
    if isinstance(params, bytes):
        return params
    if isinstance(params, str):
        # CompileIQ stores ACF payloads as hex strings.
        try:
            return bytes.fromhex(params)
        except ValueError:
            return params.encode("utf-8")
    raise SearchError(f"unexpected ACF params type: {type(params)!r}")


def _runtime_objective(
    *,
    config,
    ptx_path: str,
    kernel_name: str,
    arch: str,
    shape: str,
    block_size: int,
    num_args: int,
    timing_trials: int,
    grid: tuple[int, int, int],
    block: tuple[int, int, int],
    shared_memory_bytes: int,
    arg_nbytes: int,
    tensor_maps: dict[int, TensorMapSpec] | None = None,
    isolate: bool = False,
    isolate_timeout_s: float = 30.0,
) -> float:
    from compileiq.types import INVALID_SCORE
    from compileiq.utils.helpers import save_compiler_config

    try:
        parse_shape_token(shape)
    except ValueError:
        return INVALID_SCORE

    try:
        acf_payload, tile_payload = unpack_search_config(config)
        active_maps = apply_tile_config(tensor_maps or {}, tile_payload)
    except TensorMapError as exc:
        if os.environ.get("CIQ_DEBUG_OBJECTIVE"):
            print(f"OBJECTIVE_FAIL TensorMapError: {exc}", file=sys.stderr)
        return INVALID_SCORE

    if isinstance(acf_payload, str):
        acf_hex = acf_payload
    elif isinstance(acf_payload, bytes):
        acf_hex = acf_payload.hex()
    else:
        if os.environ.get("CIQ_DEBUG_OBJECTIVE"):
            print(
                f"OBJECTIVE_FAIL unexpected ACF type: {type(acf_payload)!r}",
                file=sys.stderr,
            )
        return INVALID_SCORE

    acf_path = None
    cubin_path = None
    try:
        acf_path = temp_artifact_path(".acf")
        cubin_path = temp_artifact_path(".cubin")
        save_compiler_config(acf_path, acf_hex)
        acf_bytes = Path(acf_path).read_bytes()
        from xla_ciq.search.ptxas_launch import compile_ptx_to_cubin

        if isolate:
            median_ms = measure_isolated_ms(
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
                timing_trials=timing_trials,
                warmup=2,
                tensor_maps=active_maps,
                acf_path=acf_path,
                timeout_s=isolate_timeout_s,
            )
            return INVALID_SCORE if median_ms is None else median_ms

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
            grid=grid,
            block=block,
            shared_memory_bytes=shared_memory_bytes,
            arg_nbytes=arg_nbytes,
            tensor_maps=active_maps,
        )
    except (CudaError, SearchError, Exception) as exc:
        # A candidate that cannot be compiled or launched is just a bad score to
        # the search, which makes a systematically broken launch look like an
        # unlucky generation. Set CIQ_DEBUG_OBJECTIVE=1 to see why.
        if os.environ.get("CIQ_DEBUG_OBJECTIVE"):
            import traceback

            print(f"OBJECTIVE_FAIL {type(exc).__name__}: {exc}", file=sys.stderr)
            traceback.print_exc()
        return INVALID_SCORE
    finally:
        for path in (acf_path, cubin_path):
            if path and os.path.exists(path):
                os.remove(path)
