"""Inspect XLA dump directories for PTX kernels and argument shapes."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from xla_ciq.search.shape_launch import is_supported_dtype, parse_shape_token


_ENTRY_RE = re.compile(
    r"^\.visible\s+\.entry\s+([A-Za-z0-9_$.]+)\s*\(",
    re.MULTILINE,
)
_PROFILE_RE = re.compile(
    r"profile_annotation\s*[:=]\s*[\"']?([^\"'\n]+)[\"']?",
    re.IGNORECASE,
)
_SHAPE_TOKEN_RE = re.compile(
    r"\b([a-z][a-z0-9]*)\[(\d+(?:,\d+)*)\]",
    re.IGNORECASE,
)


@dataclass
class KernelInfo:
    kernel_name: str
    ptx_files: list[str] = field(default_factory=list)
    thunk_sequence_files: list[str] = field(default_factory=list)
    thunk_metadata_files: list[str] = field(default_factory=list)
    profile_annotations: list[str] = field(default_factory=list)
    shapes: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass
class DumpInspection:
    dump_dir: str
    kernels: list[KernelInfo] = field(default_factory=list)


class DumpInspectError(ValueError):
    """Raised when a dump directory cannot be inspected."""


def inspect_dump(dump_dir: Path, kernel_name: str | None = None) -> DumpInspection:
    if not dump_dir.is_dir():
        raise DumpInspectError(f"dump directory not found: {dump_dir}")

    ptx_index = _index_ptx_kernels(dump_dir)
    if kernel_name is not None:
        if kernel_name not in ptx_index:
            raise DumpInspectError(
                f"kernel {kernel_name!r} not found in PTX under {dump_dir}"
            )
        names = [kernel_name]
    else:
        names = sorted(ptx_index)

    result = DumpInspection(dump_dir=str(dump_dir))
    for name in names:
        info = KernelInfo(kernel_name=name, ptx_files=sorted(ptx_index[name]))
        _enrich_from_thunks(dump_dir, info)
        _enrich_shapes_from_hlo(dump_dir, info)
        result.kernels.append(info)
    return result


def max_shape_nbytes(dump_dir: Path, kernel_name: str) -> int:
    """Largest recovered argument footprint for a kernel, in bytes.

    A dumped PTX kernel carries bounds for the shape it was compiled against,
    so launch buffers must cover that footprint regardless of which `--shapes`
    token is being fingerprinted. Returns 0 when nothing could be recovered.
    """
    try:
        inspection = inspect_dump(dump_dir, kernel_name=kernel_name)
    except DumpInspectError:
        return 0
    largest = 0
    for info in inspection.kernels:
        for token in info.shapes:
            try:
                largest = max(largest, parse_shape_token(token).nbytes)
            except ValueError:
                continue
    return largest


def _index_ptx_kernels(dump_dir: Path) -> dict[str, set[str]]:
    index: dict[str, set[str]] = {}
    for path in sorted(dump_dir.rglob("*.ptx")):
        text = path.read_text(encoding="utf-8", errors="replace")
        for match in _ENTRY_RE.finditer(text):
            name = match.group(1)
            index.setdefault(name, set()).add(str(path))
    return index


def _enrich_from_thunks(dump_dir: Path, info: KernelInfo) -> None:
    seq_hits: list[str] = []
    for path in sorted(dump_dir.rglob("*.thunk_sequence.txt")):
        text = path.read_text(encoding="utf-8", errors="replace")
        if info.kernel_name in text:
            seq_hits.append(str(path))
    info.thunk_sequence_files = seq_hits

    meta_hits: list[str] = []
    annotations: list[str] = []
    for path in sorted(dump_dir.rglob("*.thunk_metadata.txt")):
        text = path.read_text(encoding="utf-8", errors="replace")
        if info.kernel_name not in text:
            continue
        meta_hits.append(str(path))
        for block in _blocks_containing(text, info.kernel_name):
            for match in _PROFILE_RE.finditer(block):
                annotations.append(match.group(1).strip())
    info.thunk_metadata_files = meta_hits
    seen: set[str] = set()
    ordered: list[str] = []
    for ann in annotations:
        if ann not in seen:
            seen.add(ann)
            ordered.append(ann)
    info.profile_annotations = ordered
    if not seq_hits:
        info.notes.append("no thunk_sequence.txt mention found")
    if not meta_hits:
        info.notes.append("no thunk_metadata.txt mention found")


def _enrich_shapes_from_hlo(dump_dir: Path, info: KernelInfo) -> None:
    queries = list(info.profile_annotations) or [info.kernel_name]
    shapes: list[str] = []
    for path in sorted(dump_dir.rglob("*.txt")):
        name = path.name
        if name.endswith(".thunk_sequence.txt") or name.endswith(".thunk_metadata.txt"):
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for query in queries:
            if query not in text:
                continue
            for block in _blocks_containing(text, query):
                for match in _SHAPE_TOKEN_RE.finditer(block):
                    dtype = match.group(1).lower()
                    # HLO text also contains non-shape brackets such as
                    # `args[0]` / `data[1]`; those are not element types.
                    if not is_supported_dtype(dtype):
                        continue
                    token = f"{dtype}[{match.group(2)}]"
                    if token not in shapes:
                        shapes.append(token)
    info.shapes = shapes
    if not shapes:
        info.notes.append("no HLO argument shapes recovered")


def _blocks_containing(text: str, needle: str) -> list[str]:
    blocks: list[str] = []
    start = 0
    while True:
        idx = text.find(needle, start)
        if idx < 0:
            break
        lo = max(0, idx - 200)
        hi = min(len(text), idx + len(needle) + 400)
        blocks.append(text[lo:hi])
        start = idx + len(needle)
    return blocks
