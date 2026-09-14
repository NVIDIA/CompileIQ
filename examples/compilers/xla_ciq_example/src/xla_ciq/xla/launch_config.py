"""Resolve CUDA launch dimensions from XLA thunk_sequence dumps."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


_THUNK_LAUNCH_RE = re.compile(
    r"(?P<name>[A-Za-z_][A-Za-z0-9_$.]*)\s+"
    r"grid:\s*\[(?P<gx>\d+),\s*(?P<gy>\d+),\s*(?P<gz>\d+)\]"
    r"\s*threads:\s*\[(?P<tx>\d+),\s*(?P<ty>\d+),\s*(?P<tz>\d+)\]"
    r"(?:\s*shared_memory:\s*(?P<shmem>\d+)\s*bytes)?"
)


@dataclass(frozen=True)
class LaunchConfig:
    grid: tuple[int, int, int]
    block: tuple[int, int, int]
    shared_memory_bytes: int = 0
    source: str = ""

    @property
    def block_size(self) -> int:
        x, y, z = self.block
        return x * y * z


def thunk_sequence_for_ptx(ptx_path: Path) -> Path | None:
    """Map module_....211.ptx → module_....thunk_sequence.txt when present."""
    m = re.match(r"(.+)\.\d+\.ptx$", ptx_path.name)
    base = m.group(1) if m else ptx_path.stem
    candidate = ptx_path.parent / f"{base}.thunk_sequence.txt"
    return candidate if candidate.is_file() else None


def parse_thunk_launches(thunk_path: Path) -> list[tuple[str, LaunchConfig]]:
    """Return (kernel_name, launch) rows from a thunk_sequence.txt file."""
    rows: list[tuple[str, LaunchConfig]] = []
    text = thunk_path.read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines():
        match = _THUNK_LAUNCH_RE.search(line)
        if not match:
            continue
        cfg = LaunchConfig(
            grid=(
                int(match.group("gx")),
                int(match.group("gy")),
                int(match.group("gz")),
            ),
            block=(
                int(match.group("tx")),
                int(match.group("ty")),
                int(match.group("tz")),
            ),
            shared_memory_bytes=int(match.group("shmem") or 0),
            source=str(thunk_path),
        )
        rows.append((match.group("name"), cfg))
    return rows


def resolve_launch_config(
    *,
    kernel_name: str,
    ptx_path: Path,
    dump_dir: Path | None = None,
    reqntid: tuple[int, int, int] | None = None,
) -> LaunchConfig | None:
    """Find XLA launch dims for a PTX entry.

    Preference:
      1. Sibling thunk_sequence for this module PTX
      2. Any thunk_sequence under dump_dir (or ptx parent)
      3. Exact kernel name match; if several, prefer threads == reqntid
    """
    search_files: list[Path] = []
    sibling = thunk_sequence_for_ptx(ptx_path)
    if sibling is not None:
        search_files.append(sibling)

    root = dump_dir if dump_dir is not None else ptx_path.parent
    if root.is_dir():
        for path in sorted(root.rglob("*.thunk_sequence.txt")):
            if path not in search_files:
                search_files.append(path)

    exact: list[LaunchConfig] = []
    for path in search_files:
        for name, cfg in parse_thunk_launches(path):
            if name == kernel_name:
                exact.append(cfg)

    if not exact:
        return None
    if len(exact) == 1:
        return exact[0]
    if reqntid is not None:
        matched = [c for c in exact if c.block == reqntid]
        if matched:
            return matched[0]
    return exact[0]


def fallback_launch_config(*, numel: int, block_size: int) -> LaunchConfig:
    """Legacy heuristic when no thunk dims are available."""
    grid_x = max(1, (numel + block_size - 1) // block_size)
    return LaunchConfig(
        grid=(grid_x, 1, 1),
        block=(block_size, 1, 1),
        shared_memory_bytes=0,
        source="heuristic:numel/block",
    )
