"""Parse HLO-style shapes for the CLI."""

from __future__ import annotations

import re

from xla_ciq.search.shape_launch import SUPPORTED_DTYPES, is_supported_dtype

_SHAPE_RE = re.compile(
    r"^([a-z][a-z0-9]*)\[(\d+(?:,\d+)*)\]$",
    re.IGNORECASE,
)
_ALL_KEYWORD = "all"


class ShapeParseError(ValueError):
    """Raised when a shape token or list is invalid."""


def is_all_shapes(value: str) -> bool:
    """Return True if ``value`` is the standalone ``all`` keyword."""
    return value is not None and str(value).strip().lower() == _ALL_KEYWORD


def parse_shapes(value: str) -> list[str]:
    """Parse a comma-separated list of explicit HLO-style shapes.

    The keyword ``all`` is handled by populate via an XLA dump (see
    ``is_all_shapes``); do not pass ``all`` here.

    Examples:
        f32[1024,2048]
        f32[1024], f32[2048], f32[4096]
    """
    if value is None or not str(value).strip():
        raise ShapeParseError("shapes string must not be empty")

    stripped = value.strip()
    if is_all_shapes(stripped):
        raise ShapeParseError(
            "use --shapes all with --dump-dir to resolve shapes from an XLA dump"
        )

    tokens = _split_shape_list(stripped)
    if not tokens:
        raise ShapeParseError("shapes string must not be empty")

    if any(token.lower() == _ALL_KEYWORD for token in tokens):
        raise ShapeParseError(
            "'all' must be used alone (e.g. --shapes all), not mixed with "
            "explicit shapes"
        )

    normalized: list[str] = []
    for token in tokens:
        normalized.append(_normalize_shape(token))
    return normalized


def _split_shape_list(value: str) -> list[str]:
    """Split on commas that separate shapes, not dims inside brackets."""
    parts: list[str] = []
    depth = 0
    start = 0
    for i, ch in enumerate(value):
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth < 0:
                raise ShapeParseError(f"unbalanced brackets in shapes: {value!r}")
        elif ch == "," and depth == 0:
            parts.append(value[start:i].strip())
            start = i + 1
    if depth != 0:
        raise ShapeParseError(f"unbalanced brackets in shapes: {value!r}")
    parts.append(value[start:].strip())
    return [p for p in parts if p]


def _normalize_shape(token: str) -> str:
    match = _SHAPE_RE.match(token.strip())
    if not match:
        raise ShapeParseError(
            f"invalid shape {token!r}; expected forms like f32[1024,2048] "
            f"or the keyword 'all'"
        )
    dtype = match.group(1).lower()
    dims = match.group(2)
    if not is_supported_dtype(dtype):
        raise ShapeParseError(
            f"invalid shape {token!r}: {dtype!r} is not an HLO element type. "
            f"Expected one of {', '.join(sorted(SUPPORTED_DTYPES))}"
        )
    return f"{dtype}[{dims}]"
