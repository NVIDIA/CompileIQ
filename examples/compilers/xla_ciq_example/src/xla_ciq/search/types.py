"""Shared search request/result types."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SearchRequest:
    kernel_name: str
    input_path: Path
    source_bytes: bytes
    shape: str
    arch: str
    cuda_version: str
    generations: int = 5
    pool_size: int = 32
    cull_size: int | None = None
    mutate_rate: float = 0.25
    normalize: bool = False
    init_with_true_random_threshold: float = 0.9
    enable_large_fail_pool: bool = True
    block_size: int = 256
    timing_trials: int = 10
    task_timeout: float = 30.0
    num_workers: int = 1
    # Pin by default so CompileIQ does not list GitHub releases (rate-limited).
    search_space_tag: str = "search-spaces-2026.08.14"
    # Optional XLA dump root for thunk_sequence launch dims (sibling of .ptx also works).
    dump_dir: Path | None = None
    # Floor for per-argument launch buffers; 0 derives it from the XLA dump.
    min_arg_bytes: int = 0
    # Optional JSON of TMA descriptor layouts the dump cannot supply on its own.
    tensor_map_path: Path | None = None
    # Jointly search legal CUtensorMap box/swizzle values with the PTXAS ACF space.
    sweep_tensor_map_tiles: bool = False


@dataclass(frozen=True)
class SearchResult:
    acf_bytes: bytes
    notes: list[str]
    score: float | None = None
    baseline_ms: float | None = None


class SearchError(RuntimeError):
    """Raised when a real or mock search cannot produce an ACF."""


class UnlaunchableShapeError(SearchError):
    """Raised when no launch configuration for a shape runs without faulting.

    The shape cannot be timed on this GPU, so callers should skip it rather
    than treat it as a search failure.
    """
