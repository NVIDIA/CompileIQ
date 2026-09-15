"""HLO shape helpers for buffer sizing and launch config."""

from __future__ import annotations

import re
from dataclasses import dataclass

_SHAPE_RE = re.compile(
    r"^([a-z][a-z0-9]*)\[(\d+(?:,\d+)*)\]$",
    re.IGNORECASE,
)

_DTYPE_BYTES = {
    "pred": 1,
    "s8": 1,
    "u8": 1,
    "s16": 2,
    "u16": 2,
    "f16": 2,
    "bf16": 2,
    "s32": 4,
    "u32": 4,
    "f32": 4,
    "s64": 8,
    "u64": 8,
    "f64": 8,
    "c64": 8,
    "c128": 16,
}

SUPPORTED_DTYPES = frozenset(_DTYPE_BYTES)


def is_supported_dtype(dtype: str) -> bool:
    """True when dtype is an HLO element type we can size buffers for."""
    return dtype.lower() in _DTYPE_BYTES


@dataclass(frozen=True)
class ParsedShape:
    dtype: str
    dims: tuple[int, ...]

    @property
    def numel(self) -> int:
        n = 1
        for d in self.dims:
            n *= d
        return n

    @property
    def element_size(self) -> int:
        if self.dtype not in _DTYPE_BYTES:
            raise ValueError(f"unsupported dtype for launch sizing: {self.dtype}")
        return _DTYPE_BYTES[self.dtype]

    @property
    def nbytes(self) -> int:
        return self.numel * self.element_size


def parse_shape_token(token: str) -> ParsedShape:
    match = _SHAPE_RE.match(token.strip())
    if not match:
        raise ValueError(f"invalid shape token: {token!r}")
    dtype = match.group(1).lower()
    dims = tuple(int(x) for x in match.group(2).split(","))
    return ParsedShape(dtype=dtype, dims=dims)


def grid_for_numel(numel: int, block_size: int) -> int:
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    return max(1, (numel + block_size - 1) // block_size)
