"""Lightweight PTX entry metadata for CUDA launch configuration."""

from __future__ import annotations

import re
from dataclasses import dataclass

_ENTRY_RE = re.compile(
    r"\.visible\s+\.entry\s+([\w$.]+)\s*\((.*?)\)\s*(.*?)\s*\{",
    re.DOTALL,
)
_PARAM_SPLIT_RE = re.compile(r"\.param\s+")
_REQNTID_RE = re.compile(r"\.reqntid\s+(\d+)\s*,\s*(\d+)\s*,\s*(\d+)")
_ARRAY_RE = re.compile(r"\.b8\s+([\w$.]+)\s*\[\s*(\d+)\s*\]")
_SCALAR_RE = re.compile(r"\.(?:u|s|f|b)(8|16|32|64)\b\s+([\w$.]+)\s*$")
_ALIGN_RE = re.compile(r"\.align\s+(\d+)")

# `mov.b64 %rdN, <param>;` then `cvta.param.u64 %rdM, %rdN;` is how a by-value
# tensor map reaches the register a TMA instruction addresses it through.
_MOV_PARAM_RE = re.compile(r"mov\.b64\s+(%rd\d+)\s*,\s*([\w$.]+)\s*;")
_CVTA_PARAM_RE = re.compile(r"cvta\.param\.u64\s+(%rd\d+)\s*,\s*(%rd\d+)\s*;")
# A TMA store reads `[map, {coords}], [smem]` while a load reads
# `[smem], [map, {coords}]`, so the descriptor is the operand carrying the
# coordinate list rather than a fixed position.
_TMA_RE = re.compile(r"cp\.async\.bulk\.tensor\.(\d+)d\b")
_TMA_MAP_RE = re.compile(
    r"cp\.async\.bulk\.tensor\.(\d+)d\.(global|shared)[^\n;]*?\[\s*(%rd\d+)\s*,\s*\{"
)

# A CUtensorMap is an opaque 128-byte descriptor passed by value.
TENSOR_MAP_NBYTES = 128


@dataclass(frozen=True)
class PtxParam:
    """One declared kernel parameter."""

    name: str
    nbytes: int
    align: int
    is_pointer: bool
    tensor_rank: int | None = None
    tensor_direction: str | None = None

    @property
    def is_tensor_map(self) -> bool:
        """True for a by-value 128-byte blob, i.e. a CUtensorMap slot."""
        return not self.is_pointer and self.nbytes == TENSOR_MAP_NBYTES


@dataclass(frozen=True)
class PtxEntryMeta:
    kernel_name: str
    param_count: int
    reqntid: tuple[int, int, int] | None
    params: tuple[PtxParam, ...] = ()

    @property
    def block_size(self) -> int | None:
        if self.reqntid is None:
            return None
        x, y, z = self.reqntid
        return x * y * z

    @property
    def tensor_map_indices(self) -> tuple[int, ...]:
        return tuple(i for i, p in enumerate(self.params) if p.is_tensor_map)

    @property
    def uses_tensor_maps(self) -> bool:
        return bool(self.tensor_map_indices)


def resolve_block_size(meta: PtxEntryMeta, requested: int) -> int:
    """Prefer PTX .reqntid when present; otherwise use the CLI --block-size."""
    if meta.block_size is not None:
        return meta.block_size
    if requested <= 0:
        raise ValueError("--block-size must be positive")
    return requested


def validate_synthetic_launch_params(meta: PtxEntryMeta) -> None:
    """Reject parameters for which this example cannot reconstruct values."""
    unsupported = [
        f"{param.name} ({param.nbytes}-byte value)"
        for param in meta.params
        if not param.is_pointer and not param.is_tensor_map
    ]
    if unsupported:
        raise ValueError(
            "synthetic launch supports pointer and tensor-map parameters only; "
            "explicit values are unavailable for " + ", ".join(unsupported)
        )


def _parse_param(decl: str) -> PtxParam:
    text = decl.strip().rstrip(",").strip()
    align_match = _ALIGN_RE.search(text)
    align = int(align_match.group(1)) if align_match else 1

    array_match = _ARRAY_RE.search(text)
    if array_match:
        return PtxParam(
            name=array_match.group(1),
            nbytes=int(array_match.group(2)),
            align=align,
            is_pointer=False,
        )

    scalar_match = _SCALAR_RE.search(text)
    if scalar_match:
        bits = int(scalar_match.group(1))
        name = scalar_match.group(2)
    else:
        bits, name = 64, text.split()[-1]

    # `.ptr` is the explicit marker; a bare `.u64` in XLA output is also a
    # buffer address, and treating it as one is what makes the launch legal.
    is_pointer = ".ptr" in text or bits == 64
    return PtxParam(name=name, nbytes=bits // 8, align=align, is_pointer=is_pointer)


def _tensor_usage_by_param(body: str) -> dict[str, tuple[int, str]]:
    """Map descriptor param name -> (TMA rank, direction) via cvta.param chains."""
    reg_to_param: dict[str, str] = {}
    for match in _MOV_PARAM_RE.finditer(body):
        reg_to_param[match.group(1)] = match.group(2)

    cvta_to_param: dict[str, str] = {}
    for match in _CVTA_PARAM_RE.finditer(body):
        source = reg_to_param.get(match.group(2))
        if source is not None:
            cvta_to_param[match.group(1)] = source

    usage: dict[str, tuple[int, str]] = {}
    for match in _TMA_MAP_RE.finditer(body):
        param = cvta_to_param.get(match.group(3))
        if param is not None:
            direction = "store" if match.group(2) == "global" else "load"
            usage[param] = (int(match.group(1)), direction)
    return usage


def _entry_body(ptx_text: str, brace_index: int) -> str:
    depth = 0
    for i in range(brace_index, len(ptx_text)):
        char = ptx_text[i]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return ptx_text[brace_index : i + 1]
    return ptx_text[brace_index:]


def inspect_ptx_entry(ptx_text: str, kernel_name: str) -> PtxEntryMeta:
    """Parse parameters, optional .reqntid, and TMA ranks for a PTX entry."""
    for match in _ENTRY_RE.finditer(ptx_text):
        name = match.group(1)
        if name != kernel_name:
            continue
        decls = [d for d in _PARAM_SPLIT_RE.split(match.group(2)) if d.strip()]
        params = [_parse_param(d) for d in decls]

        req = _REQNTID_RE.search(match.group(3))
        reqntid = None
        if req is not None:
            reqntid = (int(req.group(1)), int(req.group(2)), int(req.group(3)))

        body = _entry_body(ptx_text, match.end() - 1)
        usage = _tensor_usage_by_param(body)
        observed = {int(m.group(1)) for m in _TMA_RE.finditer(body)}
        sole_rank = observed.pop() if len(observed) == 1 else None
        params = [
            (
                param
                if not param.is_tensor_map
                else PtxParam(
                    name=param.name,
                    nbytes=param.nbytes,
                    align=param.align,
                    is_pointer=param.is_pointer,
                    tensor_rank=usage.get(param.name, (sole_rank, None))[0],
                    tensor_direction=usage.get(param.name, (None, None))[1],
                )
            )
            for param in params
        ]

        return PtxEntryMeta(
            kernel_name=name,
            param_count=len(params),
            reqntid=reqntid,
            params=tuple(params),
        )
    raise ValueError(f"PTX entry not found: {kernel_name!r}")
