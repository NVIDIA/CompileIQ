"""Decide which CUtensorMap layout backs each descriptor parameter.

A descriptor's layout is chosen on the host, so it is absent from PTX. Two
sources can supply it: an explicit layout file, or the XLA dump. The dump only
records the tile of a fusion's *output*, so a descriptor the kernel stores
through can be inferred while descriptors it loads operands through cannot.
Anything left unresolved is reported rather than guessed, because a wrong box
size makes a kernel move the wrong number of bytes into shared memory.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from xla_ciq.search.ptx_meta import PtxEntryMeta
from xla_ciq.xla.tensor_map import (
    TensorMapError,
    TensorMapSpec,
    find_fusion_layout,
    from_hlo_shape,
)


@dataclass(frozen=True)
class TensorMapPlan:
    specs: dict[int, TensorMapSpec]
    unresolved: tuple[int, ...] = ()
    notes: tuple[str, ...] = ()

    @property
    def is_complete(self) -> bool:
        return not self.unresolved

    @property
    def required(self) -> bool:
        return bool(self.specs) or bool(self.unresolved)

    def describe_gap(self, meta: PtxEntryMeta) -> str:
        names = [
            meta.params[i].name if i < len(meta.params) else f"param_{i}"
            for i in self.unresolved
        ]
        detail = "; ".join(self.notes)
        return (
            f"kernel {meta.kernel_name} needs tensor map layouts for "
            f"{', '.join(names)}; supply them with --tensor-maps"
            + (f" ({detail})" if detail else "")
        )


def resolve_tensor_maps(
    *,
    meta: PtxEntryMeta,
    ptx_path: Path | None = None,
    dump_dir: Path | None = None,
    overrides: dict[int, TensorMapSpec] | None = None,
) -> TensorMapPlan:
    """Build the descriptor layouts for a kernel, or say which are missing."""
    indices = meta.tensor_map_indices
    if not indices:
        return TensorMapPlan(specs={})

    overrides = overrides or {}
    specs: dict[int, TensorMapSpec] = {}
    unresolved: list[int] = []
    notes: list[str] = []

    layout = None
    if dump_dir is not None and dump_dir.is_dir():
        layout = find_fusion_layout(dump_dir, meta.kernel_name, ptx_path=ptx_path)
        if layout is None:
            notes.append("no fusion tile found in dump")

    for index in indices:
        if index in overrides:
            specs[index] = overrides[index]
            continue

        param = meta.params[index]
        if param.tensor_direction != "store":
            notes.append(f"{param.name} loads an operand, whose tile the dump omits")
            unresolved.append(index)
            continue
        if layout is None:
            unresolved.append(index)
            continue
        if param.tensor_rank is not None and param.tensor_rank != len(layout.tile):
            notes.append(
                f"{param.name} is rank {param.tensor_rank} but the fusion tile is "
                f"rank {len(layout.tile)}"
            )
            unresolved.append(index)
            continue
        try:
            specs[index] = from_hlo_shape(
                shape_token=layout.shape_token,
                tile=list(layout.tile),
                minor_to_major=list(layout.minor_to_major),
                source=layout.source,
            )
        except TensorMapError as exc:
            notes.append(f"{param.name}: {exc}")
            unresolved.append(index)

    return TensorMapPlan(
        specs=specs, unresolved=tuple(unresolved), notes=tuple(dict.fromkeys(notes))
    )
