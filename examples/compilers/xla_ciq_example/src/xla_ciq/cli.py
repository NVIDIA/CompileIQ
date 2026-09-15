"""Command-line entry point for the XLA-CIQ experiment."""

from __future__ import annotations

import sys
from pathlib import Path

import click

from xla_ciq import __version__
from xla_ciq.commands.bundle_evaluate import run_bundle_evaluate
from xla_ciq.commands.bundle_inspect import run_bundle_inspect
from xla_ciq.commands.populate import run_populate
from xla_ciq.commands.xla_inspect import run_xla_inspect


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(__version__, prog_name="xla-ciq")
def cli() -> None:
    """Experimental producer-side XLA/CompileIQ reference example."""


@cli.group("bundle")
def bundle_group() -> None:
    """Experimental ACF bundle operations."""


@bundle_group.command("populate")
@click.argument("bundle", type=click.Path(path_type=Path))
@click.option(
    "--kernel-name",
    required=True,
    help=(
        "Target kernel name within the input source "
        "(single kernel per invoke; multi-kernel populate is not supported yet)"
    ),
)
@click.option(
    "--input",
    "input_path",
    required=True,
    type=click.Path(path_type=Path),
    help="Path to PTX source (required for real search; .mlir only with --mock)",
)
@click.option(
    "--shapes",
    required=True,
    help=(
        "Comma-separated HLO-style shapes/sizes to tune, e.g. "
        "f32[1024],f32[2048], or 'all' to use every shape found for "
        "--kernel-name in --dump-dir. Each shape runs a separate search."
    ),
)
@click.option(
    "--dump-dir",
    type=click.Path(path_type=Path, exists=True, file_okay=False),
    default=None,
    help="XLA dump directory (required when --shapes all)",
)
@click.option("--arch", default="sm_90a", show_default=True, help="Target GPU architecture")
@click.option(
    "--cuda-version",
    default="13.3",
    show_default=True,
    help="CUDA version for fingerprint / PtxasSearchSpace",
)
@click.option(
    "--replace",
    is_flag=True,
    help="Replace an existing fingerprint mapping",
)
@click.option(
    "--keep-slower-acf",
    is_flag=True,
    help=(
        "Store ACF even when re-measured tuned timing is not faster than baseline; "
        "default is to leave the producer mapping absent"
    ),
)
@click.option(
    "--mock/--real",
    default=True,
    show_default=True,
    help="Use deterministic mock mode or explicitly opt into synthetic GPU timing",
)
@click.option("--generations", default=5, show_default=True, type=int, help="Search generations")
@click.option(
    "--pool-size",
    default=32,
    show_default=True,
    type=int,
    help="Candidates evaluated per generation",
)
@click.option(
    "--cull-size",
    default=None,
    type=int,
    help="Parents kept per generation (even integer; default: CompileIQ derives from pool-size)",
)
@click.option(
    "--mutate-rate",
    default=0.25,
    show_default=True,
    type=float,
    help="Chance a sampled candidate is perturbed between generations",
)
@click.option(
    "--normalize",
    is_flag=True,
    help="Normalize scores inside CompileIQ (multi-GPU / multi-machine)",
)
@click.option(
    "--init-random-threshold",
    default=0.9,
    show_default=True,
    type=float,
    help="Fraction of seed-high/seed-low sampling at search init",
)
@click.option(
    "--large-fail-pool/--no-large-fail-pool",
    default=True,
    show_default=True,
    help="Resubmit a full pool when a generation has too many failures",
)
@click.option(
    "--block-size",
    default=256,
    show_default=True,
    type=int,
    help="CUDA launch block size when PTX has no .reqntid",
)
@click.option(
    "--timing-trials",
    default=10,
    show_default=True,
    type=int,
    help="Timed CUDA launches per candidate (after warmup)",
)
@click.option(
    "--task-timeout",
    default=30.0,
    show_default=True,
    type=float,
    help=(
        "CompileIQ per-objective timeout in seconds; this is not an overall "
        "search deadline"
    ),
)
@click.option(
    "--num-workers",
    default=1,
    show_default=True,
    type=int,
    help="CompileIQ worker processes (1 recommended for stable single-GPU timing)",
)
@click.option(
    "--search-space-tag",
    default="search-spaces-2026.08.14",
    show_default=True,
    help=(
        "CompileIQ search-space release tag "
        "(pinning avoids resolver variability and rate limits)"
    ),
)
@click.option(
    "--min-arg-bytes",
    default=0,
    show_default=True,
    type=int,
    help=(
        "Floor for each launch buffer in bytes. 0 derives the kernel's "
        "footprint from --dump-dir, which avoids out-of-bounds launches"
    ),
)
@click.option(
    "--tensor-maps",
    "tensor_map_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help=(
        "JSON of TMA descriptor layouts, keyed by kernel then parameter index. "
        "Needed for kernels whose tensor map tiles are not in the XLA dump"
    ),
)
@click.option(
    "--sweep-tensor-map-tiles/--no-sweep-tensor-map-tiles",
    default=False,
    show_default=True,
    help=(
        "Jointly search legal CUtensorMap box/swizzle values with the PTXAS ACF "
        "space (CompileIQ multi-config). Seed layouts still come from the dump "
        "or --tensor-maps"
    ),
)
@click.pass_context
def populate_command(
    ctx: click.Context,
    bundle: Path,
    kernel_name: str,
    input_path: Path,
    shapes: str,
    dump_dir: Path | None,
    arch: str,
    cuda_version: str,
    replace: bool,
    keep_slower_acf: bool,
    mock: bool,
    generations: int,
    pool_size: int,
    cull_size: int | None,
    mutate_rate: float,
    normalize: bool,
    init_random_threshold: float,
    large_fail_pool: bool,
    block_size: int,
    timing_trials: int,
    task_timeout: float,
    num_workers: int,
    search_space_tag: str,
    min_arg_bytes: int,
    tensor_map_path: Path | None,
    sweep_tensor_map_tiles: bool,
) -> None:
    """Illustrate CompileIQ search and append producer-side ACF entries."""
    code = run_populate(
        bundle=bundle,
        kernel_name=kernel_name,
        input_path=input_path,
        shapes=shapes,
        dump_dir=dump_dir,
        arch=arch,
        cuda_version=cuda_version,
        replace=replace,
        keep_slower_acf=keep_slower_acf,
        mock=mock,
        generations=generations,
        pool_size=pool_size,
        cull_size=cull_size,
        mutate_rate=mutate_rate,
        normalize=normalize,
        init_with_true_random_threshold=init_random_threshold,
        enable_large_fail_pool=large_fail_pool,
        block_size=block_size,
        timing_trials=timing_trials,
        task_timeout=task_timeout,
        num_workers=num_workers,
        search_space_tag=search_space_tag,
        min_arg_bytes=min_arg_bytes,
        tensor_map_path=tensor_map_path,
        sweep_tensor_map_tiles=sweep_tensor_map_tiles,
    )
    ctx.exit(code)


@bundle_group.command("inspect")
@click.argument("bundle", type=click.Path(path_type=Path))
@click.option(
    "--json",
    "as_json",
    is_flag=True,
    help="Emit machine-readable JSON",
)
@click.pass_context
def bundle_inspect_command(
    ctx: click.Context,
    bundle: Path,
    as_json: bool,
) -> None:
    """Summarize fingerprints and control files in an ACF Bundle."""
    code = run_bundle_inspect(bundle=bundle, as_json=as_json)
    ctx.exit(code)


@bundle_group.command("evaluate")
@click.argument("bundle", type=click.Path(path_type=Path))
@click.option(
    "--input",
    "input_path",
    required=True,
    type=click.Path(path_type=Path),
    help="PTX used to select and re-measure bundled ACFs (must match populate input)",
)
@click.option(
    "--kernel-name",
    required=True,
    help="Kernel to evaluate (filters metadata / fingerprint computation)",
)
@click.option(
    "--shapes",
    default=None,
    help=(
        "Comma-separated shapes to evaluate. If omitted, evaluate every "
        "bundle entry with matching kernel metadata."
    ),
)
@click.option(
    "--fingerprint",
    "fingerprints",
    multiple=True,
    help="Restrict evaluation to specific fingerprint(s)",
)
@click.option("--arch", default="sm_90a", show_default=True, help="Target GPU architecture")
@click.option(
    "--cuda-version",
    default="13.3",
    show_default=True,
    help="CUDA version used for fingerprint recomputation",
)
@click.option(
    "--block-size",
    default=256,
    show_default=True,
    type=int,
    help="CUDA launch block size when PTX has no .reqntid",
)
@click.option(
    "--timing-trials",
    default=10,
    show_default=True,
    type=int,
    help="Timed CUDA launches per measurement round (after warmup)",
)
@click.option(
    "--rounds",
    default=5,
    show_default=True,
    type=int,
    help="Independent measurement rounds (median/stdev across rounds)",
)
@click.option(
    "--mock/--real",
    default=True,
    show_default=True,
    help="Check selection in mock mode or explicitly opt into synthetic GPU timing",
)
@click.option(
    "--json",
    "as_json",
    is_flag=True,
    help="Emit machine-readable JSON",
)
@click.option(
    "--dump-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="XLA dump directory used to size launch buffers safely",
)
@click.option(
    "--min-arg-bytes",
    default=0,
    show_default=True,
    type=int,
    help="Floor for each launch buffer in bytes (0 derives it from --dump-dir)",
)
@click.option(
    "--tensor-maps",
    "tensor_map_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="JSON of TMA descriptor layouts, keyed by kernel then parameter index",
)
@click.pass_context
def bundle_evaluate_command(
    ctx: click.Context,
    bundle: Path,
    input_path: Path,
    kernel_name: str,
    shapes: str | None,
    fingerprints: tuple[str, ...],
    arch: str,
    cuda_version: str,
    block_size: int,
    timing_trials: int,
    rounds: int,
    mock: bool,
    as_json: bool,
    dump_dir: Path | None,
    min_arg_bytes: int,
    tensor_map_path: Path | None,
) -> None:
    """Select bundled ACFs and optionally compare synthetic launch timing."""
    code = run_bundle_evaluate(
        bundle=bundle,
        input_path=input_path,
        kernel_name=kernel_name,
        shapes=shapes,
        fingerprints=fingerprints,
        arch=arch,
        cuda_version=cuda_version,
        block_size=block_size,
        timing_trials=timing_trials,
        rounds=rounds,
        mock=mock,
        as_json=as_json,
        dump_dir=dump_dir,
        min_arg_bytes=min_arg_bytes,
        tensor_map_path=tensor_map_path,
    )
    ctx.exit(code)


@cli.group("xla")
def xla_group() -> None:
    """XLA dump helpers."""


@xla_group.command("inspect")
@click.argument("dump_dir", type=click.Path(path_type=Path))
@click.option(
    "--kernel-name",
    default=None,
    help="Restrict output to one kernel function name",
)
@click.option(
    "--json",
    "as_json",
    is_flag=True,
    help="Emit machine-readable JSON",
)
@click.pass_context
def xla_inspect_command(
    ctx: click.Context,
    dump_dir: Path,
    kernel_name: str | None,
    as_json: bool,
) -> None:
    """Inspect an XLA dump directory for kernels and shapes."""
    code = run_xla_inspect(
        dump_dir=dump_dir,
        kernel_name=kernel_name,
        as_json=as_json,
    )
    ctx.exit(code)


def main(argv: list[str] | None = None) -> int:
    """Entry point used by the in-place module command and tests."""
    try:
        result = cli.main(args=argv, prog_name="xla-ciq", standalone_mode=False)
        return int(result or 0)
    except click.exceptions.Exit as exc:
        return int(exc.exit_code)
    except click.ClickException as exc:
        exc.show()
        return int(exc.exit_code)
    except SystemExit as exc:
        code = exc.code
        if code is None:
            return 0
        if isinstance(code, int):
            return code
        return 1


if __name__ == "__main__":
    sys.exit(main())
