"""Tensor map (TMA) descriptor layouts for Hopper+ kernels.

A kernel that moves tiles with `cp.async.bulk.tensor` takes its global buffer
through a 128-byte `CUtensorMap` passed by value, not through a pointer. The
descriptor is built on the host, so PTX alone does not say what tensor it
describes: we need the element type, the global dimensions, and the tile (box)
dimensions. This module holds that layout as plain data so it can be recovered
from an XLA dump, authored by hand, and unit tested without a GPU.

Dimension order follows the CUDA convention: index 0 is the innermost
(contiguous) dimension, which is the reverse of how HLO prints a shape.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from xla_ciq.search.shape_launch import parse_shape_token

MAX_TENSOR_RANK = 5
_TMA_ALIGN_BYTES = 16
_VALID_SWIZZLE_BYTES = (0, 32, 64, 128)

# e.g. `%input_transpose_fusion.1 = bf16[2,3,2,8,2048]{4,3,2,1,0} fusion(...`
_INSTRUCTION_RE = re.compile(
    r"^\s*%(?P<name>[\w$.]+)\s*=\s*"
    r"(?P<shape>[a-z]\w*\[\d+(?:,\d+)*\])"
    r"(?:\{(?P<layout>[\d,]*)\})?\s*"
    r"(?P<rest>fusion\(.*)$"
)
_TILE_RE = re.compile(r"\"sizes\"\s*:\s*\[(?P<sizes>[^\]]*)\]")
_QUOTED_INT_RE = re.compile(r"\d+")


class TensorMapError(ValueError):
    """Raised when a tensor map layout is missing or unusable."""


@dataclass(frozen=True)
class TensorMapSpec:
    """Layout of one CUtensorMap, in CUDA (innermost-first) dimension order."""

    dtype: str
    dims: tuple[int, ...]
    box: tuple[int, ...]
    element_strides: tuple[int, ...] = ()
    swizzle_bytes: int | None = None
    source: str = ""

    @property
    def rank(self) -> int:
        return len(self.dims)

    @property
    def element_size(self) -> int:
        return parse_shape_token(f"{self.dtype}[1]").element_size

    @property
    def strides(self) -> tuple[int, ...]:
        """Byte strides for dimensions 1..rank-1, as cuTensorMapEncodeTiled wants."""
        out: list[int] = []
        acc = self.element_size
        for dim in self.dims[:-1]:
            acc *= dim
            out.append(acc)
        return tuple(out)

    @property
    def numel(self) -> int:
        total = 1
        for dim in self.dims:
            total *= dim
        return total

    @property
    def nbytes(self) -> int:
        return self.numel * self.element_size

    @property
    def effective_element_strides(self) -> tuple[int, ...]:
        return self.element_strides or tuple(1 for _ in self.dims)

    def validate(self) -> None:
        """Reject layouts the driver would refuse, with a readable reason."""
        if not self.dims:
            raise TensorMapError("tensor map needs at least one dimension")
        if self.rank > MAX_TENSOR_RANK:
            raise TensorMapError(f"tensor map rank {self.rank} exceeds {MAX_TENSOR_RANK}")
        if len(self.box) != self.rank:
            raise TensorMapError(
                f"box rank {len(self.box)} does not match dims rank {self.rank}"
            )
        if len(self.effective_element_strides) != self.rank:
            raise TensorMapError("element_strides rank does not match dims rank")
        if any(d <= 0 for d in self.dims) or any(b <= 0 for b in self.box):
            raise TensorMapError("tensor map dims and box must be positive")
        for i, (box, dim) in enumerate(zip(self.box, self.dims)):
            if box > dim:
                raise TensorMapError(f"box[{i}]={box} exceeds dim[{i}]={dim}")
        inner_bytes = self.box[0] * self.element_size
        if inner_bytes % _TMA_ALIGN_BYTES:
            raise TensorMapError(
                f"innermost box must be a multiple of {_TMA_ALIGN_BYTES} bytes, "
                f"got {inner_bytes}"
            )
        if self.swizzle_bytes is not None and self.swizzle_bytes not in _VALID_SWIZZLE_BYTES:
            raise TensorMapError(f"invalid swizzle_bytes: {self.swizzle_bytes}")
        for stride in self.strides:
            if stride % _TMA_ALIGN_BYTES:
                raise TensorMapError(
                    f"global strides must be {_TMA_ALIGN_BYTES}-byte aligned, "
                    f"got {stride}"
                )

    def swizzle_candidates(self) -> tuple[int, ...]:
        """Swizzle modes to try, most likely first.

        The emitter's choice is not recorded in PTX. Only modes whose
        granularity fits the innermost tile are legal, so try the largest such
        mode down to none.
        """
        if self.swizzle_bytes is not None:
            return (self.swizzle_bytes,)
        inner_bytes = self.box[0] * self.element_size
        return tuple(s for s in (128, 64, 32) if s <= inner_bytes) + (0,)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "dtype": self.dtype,
            "dims": list(self.dims),
            "box": list(self.box),
            "element_strides": list(self.effective_element_strides),
        }
        if self.swizzle_bytes is not None:
            payload["swizzle_bytes"] = self.swizzle_bytes
        if self.source:
            payload["source"] = self.source
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TensorMapSpec:
        """Build a spec from CUDA-order `dims`/`box` or HLO-order `hlo_shape`/`tile`."""
        if "hlo_shape" in payload:
            tile = payload.get("tile")
            if tile is None:
                raise TensorMapError("hlo_shape requires a matching 'tile'")
            spec = from_hlo_shape(
                shape_token=str(payload["hlo_shape"]),
                tile=[int(x) for x in tile],
                minor_to_major=payload.get("minor_to_major"),
                source=str(payload.get("source", "explicit")),
            )
        else:
            missing = [key for key in ("dtype", "dims", "box") if key not in payload]
            if missing:
                raise TensorMapError(
                    f"tensor map spec missing keys: {', '.join(missing)}"
                )
            spec = cls(
                dtype=str(payload["dtype"]).lower(),
                dims=tuple(int(x) for x in payload["dims"]),
                box=tuple(int(x) for x in payload["box"]),
                element_strides=tuple(
                    int(x) for x in payload.get("element_strides", ()) or ()
                ),
                source=str(payload.get("source", "explicit")),
            )
        swizzle = payload.get("swizzle_bytes")
        if swizzle is not None:
            spec = TensorMapSpec(
                dtype=spec.dtype,
                dims=spec.dims,
                box=spec.box,
                element_strides=spec.element_strides,
                swizzle_bytes=int(swizzle),
                source=spec.source,
            )
        spec.validate()
        return spec


def from_hlo_shape(
    *,
    shape_token: str,
    tile: list[int],
    minor_to_major: list[int] | None = None,
    source: str = "hlo",
) -> TensorMapSpec:
    """Convert an HLO shape plus tile sizes into CUDA innermost-first order."""
    parsed = parse_shape_token(shape_token)
    rank = len(parsed.dims)
    if len(tile) != rank:
        raise TensorMapError(
            f"tile rank {len(tile)} does not match shape rank {rank} for {shape_token}"
        )
    order = list(minor_to_major) if minor_to_major else list(reversed(range(rank)))
    if sorted(order) != list(range(rank)):
        raise TensorMapError(f"invalid minor_to_major {order} for rank {rank}")
    return TensorMapSpec(
        dtype=parsed.dtype,
        dims=tuple(parsed.dims[i] for i in order),
        box=tuple(int(tile[i]) for i in order),
        source=source,
    )


def load_tensor_map_file(path: Path) -> dict[str, dict[int, TensorMapSpec]]:
    """Read hand-authored descriptor layouts.

    Expected shape, where the inner keys are kernel parameter indices:

        {"my_kernel": {"0": {"hlo_shape": "bf16[128,64]", "tile": [64, 32]}}}
    """
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TensorMapError(f"cannot read tensor map file {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise TensorMapError(f"tensor map file must be a JSON object: {path}")

    result: dict[str, dict[int, TensorMapSpec]] = {}
    for kernel, entry in raw.items():
        params = entry.get("params", entry) if isinstance(entry, dict) else None
        if not isinstance(params, dict):
            raise TensorMapError(f"tensor map entry for {kernel!r} must be an object")
        specs: dict[int, TensorMapSpec] = {}
        for index, payload in params.items():
            try:
                slot = int(index)
            except (TypeError, ValueError) as exc:
                raise TensorMapError(
                    f"tensor map param key must be an index, got {index!r}"
                ) from exc
            if not isinstance(payload, dict):
                raise TensorMapError(
                    f"tensor map spec for {kernel}[{slot}] must be an object"
                )
            payload.setdefault("source", f"file:{path.name}")
            specs[slot] = TensorMapSpec.from_dict(payload)
        result[kernel] = specs
    return result


@dataclass(frozen=True)
class HloFusionLayout:
    """Output shape and tile recovered for a fusion instruction."""

    shape_token: str
    minor_to_major: tuple[int, ...]
    tile: tuple[int, ...]
    source: str


def _kernel_name_pattern(kernel_name: str) -> re.Pattern[str]:
    # XLA mangles `input_transpose_fusion.1` to `input_transpose_fusion_1`, so
    # each underscore may stand for either character in the HLO name.
    return re.compile("^" + re.escape(kernel_name).replace("_", "[_.]") + "$")


def _hlo_candidates(dump_dir: Path, ptx_path: Path | None) -> list[Path]:
    """HLO dumps to scan, preferring the module the PTX came from.

    The same kernel name can appear in several dumped modules with different
    shapes, so the module that produced this PTX wins.
    """
    paths = sorted(dump_dir.rglob("*after_optimizations.txt"))
    if ptx_path is None:
        return paths
    match = re.match(r"(.+)\.\d+\.ptx$", ptx_path.name)
    base = match.group(1) if match else ptx_path.stem
    return sorted(paths, key=lambda p: (not p.name.startswith(base), p.name))


def find_fusion_layout(
    dump_dir: Path, kernel_name: str, *, ptx_path: Path | None = None
) -> HloFusionLayout | None:
    """Recover the output shape and tile of the fusion behind a kernel."""
    pattern = _kernel_name_pattern(kernel_name)
    for path in _hlo_candidates(dump_dir, ptx_path):
        text = path.read_text(encoding="utf-8", errors="replace")
        for line in text.splitlines():
            match = _INSTRUCTION_RE.match(line)
            if not match or not pattern.match(match.group("name")):
                continue
            tile_match = _TILE_RE.search(match.group("rest"))
            if not tile_match:
                continue
            tile = tuple(
                int(x) for x in _QUOTED_INT_RE.findall(tile_match.group("sizes"))
            )
            layout = match.group("layout")
            rank = len(parse_shape_token(match.group("shape")).dims)
            minor_to_major = (
                tuple(int(x) for x in layout.split(",") if x != "")
                if layout
                else tuple(reversed(range(rank)))
            )
            return HloFusionLayout(
                shape_token=match.group("shape"),
                minor_to_major=minor_to_major,
                tile=tile,
                source=f"{path.name}:{match.group('name')}",
            )
    return None
