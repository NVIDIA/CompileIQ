"""Sweep TMA descriptor tiles as CompileIQ search knobs.

The PTX for a TMA kernel is compiled against a fixed shared-memory layout, so
the host-side ``CUtensorMap`` box must match what the emitter assumed. When the
dump does not record that box (typical for load descriptors), CompileIQ can
search over legal box sizes and swizzle modes jointly with the PTXAS ACF space
via a multi-config search: ``[PtxasSearchSpace, tile_dict]``.
"""

from __future__ import annotations

import builtins
from typing import Any

from xla_ciq.xla.tensor_map import TensorMapError, TensorMapSpec

_POW2_BOXES = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192)
_SWIZZLES = (0, 32, 64, 128)
_TMA_ALIGN_BYTES = 16
# Nested search-space root; restored into the objective as config["tm"].
TILE_SPACE_ROOT = "tm"
# Joint ACF×tile search explodes quickly; keep only the seed box and one
# octave above/below so early generations still sample the probed layout.
_NEAR_SEED_FACTORS = (1, 2)


def legal_box_sizes(
    dim: int,
    *,
    element_size: int,
    innermost: bool,
    seed: int | None = None,
) -> list[int]:
    """Positive box sizes that fit `dim` and satisfy TMA alignment when innermost.

    When a seed box is known (dump or ``--tensor-maps``), only that value and
    one octave nearby are offered. Otherwise every legal power-of-two up to
    ``dim`` is allowed so a cold search can still discover a working tile.
    """
    if dim <= 0:
        return []

    def _ok(size: int) -> bool:
        if size <= 0 or size > dim:
            return False
        if innermost and (size * element_size) % _TMA_ALIGN_BYTES:
            return False
        return True

    sizes: list[int] = []
    if seed is not None and _ok(seed):
        sizes.append(seed)
        for factor in _NEAR_SEED_FACTORS[1:]:
            for candidate in (seed * factor, max(1, seed // factor)):
                if _ok(candidate) and candidate not in sizes:
                    sizes.append(candidate)
        return sizes

    for size in _POW2_BOXES:
        if size > dim:
            break
        if _ok(size):
            sizes.append(size)
    if _ok(dim) and dim not in sizes:
        sizes.append(dim)
    return sizes


def box_size_options(spec: TensorMapSpec, axis: int) -> list[int]:
    """Box sizes offered for one axis, seed first.

    Derived only from the seed spec so the search space and the objective that
    decodes a sample always agree on the option list.
    """
    return legal_box_sizes(
        spec.dims[axis],
        element_size=spec.element_size,
        innermost=(axis == 0),
        seed=spec.box[axis],
    )


def swizzle_options(spec: TensorMapSpec) -> list[int]:
    """Swizzle modes offered for a descriptor, seed first."""
    seed = spec.swizzle_bytes if spec.swizzle_bytes is not None else 0
    options = [seed]
    for value in spec.swizzle_candidates():
        if value not in options:
            options.append(value)
    for value in _SWIZZLES:
        if value not in options and (
            value == 0 or value <= spec.box[0] * spec.element_size
        ):
            options.append(value)
    return options[:4]


def _index_knob(count: int) -> Any:
    """Knob selecting an option index, with init pinned to the seed (index 0).

    ``choice`` samples uniformly and cannot be seeded, so a wide joint space
    would almost never reproduce the seed layout that is known to launch. A
    ``range`` with ``seed_low``/``seed_high`` on the seed index makes CompileIQ
    initialize ``init_with_true_random_threshold`` of the first generation on it.
    """
    from compileiq.search_spaces.base import literal
    from compileiq.search_spaces.base import range as ciq_range

    if count <= 1:
        return literal(0)
    return ciq_range(0, count - 1, 1, seed_low=0, seed_high=0)


def build_tile_search_space(
    specs: dict[int, TensorMapSpec],
    *,
    sweep_indices: tuple[int, ...] | None = None,
) -> dict[str, Any]:
    """CompileIQ dict search space over box dims + swizzle for each descriptor.

    Knob values are indices into the per-axis option lists, not sizes, so that
    the seed layout is index 0 everywhere and can be pinned at search init.
    Requires ``compileiq.search_spaces.base`` at call time so unit tests can
    exercise the pure helpers without importing CompileIQ.
    """
    indices = sweep_indices if sweep_indices is not None else tuple(sorted(specs))
    if not indices:
        raise TensorMapError("no tensor map parameters to sweep")

    per_param: dict[str, dict[str, Any]] = {}
    for index in indices:
        if index not in specs:
            raise TensorMapError(f"cannot sweep missing tensor map param {index}")
        spec = specs[index]
        spec.validate()
        entry: dict[str, Any] = {}
        for axis in builtins.range(spec.rank):
            sizes = box_size_options(spec, axis)
            if not sizes:
                raise TensorMapError(
                    f"no legal box sizes for tensor map[{index}] axis {axis} "
                    f"dim={spec.dims[axis]}"
                )
            entry[f"b{axis}"] = _index_knob(len(sizes))
        entry["swz"] = _index_knob(len(swizzle_options(spec)))
        per_param[str(index)] = entry
    return {TILE_SPACE_ROOT: per_param}


def default_sweep_indices(specs: dict[int, TensorMapSpec]) -> tuple[int, ...]:
    """Prefer sweeping guessed/file layouts; fall back to every descriptor.

    Dump-inferred *store* tiles are usually authoritative. Load layouts from
    ``--tensor-maps`` are the ones we do not trust, so they are swept first.
    """
    guessed = [
        index
        for index, spec in sorted(specs.items())
        if spec.source.startswith("file:")
        or spec.source.startswith("guess:")
        or spec.source == "explicit"
    ]
    if guessed:
        return tuple(guessed)
    return tuple(sorted(specs))


def _option_index(value: Any, count: int) -> int:
    """Clamp a sampled knob value to a valid option index."""
    try:
        index = int(value)
    except (TypeError, ValueError) as exc:
        raise TensorMapError(f"tile option index must be an int, got {value!r}") from exc
    return max(0, min(index, count - 1))


def apply_tile_config(
    base: dict[int, TensorMapSpec],
    tiles: dict[str, Any] | None,
) -> dict[int, TensorMapSpec]:
    """Overlay a CompileIQ tile sample onto the seed tensor-map specs."""
    if not tiles:
        return dict(base)
    root = tiles.get(TILE_SPACE_ROOT, tiles)
    if not isinstance(root, dict):
        raise TensorMapError(f"tile config must be a dict, got {type(root).__name__}")

    updated = dict(base)
    for key, payload in root.items():
        try:
            index = int(key)
        except (TypeError, ValueError) as exc:
            raise TensorMapError(f"tile config key must be a param index, got {key!r}") from exc
        if index not in base:
            raise TensorMapError(f"tile config references unknown param {index}")
        if not isinstance(payload, dict):
            raise TensorMapError(f"tile config for param {index} must be an object")
        seed = base[index]
        box = list(seed.box)
        for axis in range(seed.rank):
            field = f"b{axis}"
            if field in payload and payload[field] is not None:
                sizes = box_size_options(seed, axis)
                box[axis] = sizes[_option_index(payload[field], len(sizes))]
        swizzle = seed.swizzle_bytes
        if "swz" in payload and payload["swz"] is not None:
            options = swizzle_options(seed)
            swizzle = options[_option_index(payload["swz"], len(options))]
        spec = TensorMapSpec(
            dtype=seed.dtype,
            dims=seed.dims,
            box=tuple(box),
            element_strides=seed.element_strides,
            swizzle_bytes=swizzle,
            source=f"sweep:{seed.source}" if seed.source else "sweep",
        )
        spec.validate()
        updated[index] = spec
    return updated


def unpack_search_config(config: Any) -> tuple[Any, dict[str, Any]]:
    """Split a multi-config sample into ``(acf_payload, tile_dict)``.

    Single-config PTXAS searches still pass a bare ACF hex string. Multi-config
    searches pass ``[acf, tiles]`` in the order the search spaces were listed.
    """
    if isinstance(config, list):
        if not config:
            raise TensorMapError("empty multi-config search sample")
        acf = config[0]
        tiles = config[1] if len(config) > 1 else {}
        if tiles is None:
            tiles = {}
        if not isinstance(tiles, dict):
            raise TensorMapError(
                f"tile config must be a dict, got {type(tiles).__name__}"
            )
        return acf, tiles
    return config, {}


def describe_tile_config(
    base: dict[int, TensorMapSpec],
    tiles: dict[str, Any] | None,
) -> str:
    """One-line summary of a tile sample for search notes.

    Knob values are option indices, so resolve them against the seed specs to
    report the box and swizzle the descriptor was actually built with.
    """
    if not tiles:
        return ""
    try:
        resolved = apply_tile_config(base, tiles)
    except TensorMapError as exc:
        return f"<unresolvable tile config: {exc}>"
    root = tiles.get(TILE_SPACE_ROOT, tiles)
    swept = sorted(root) if isinstance(root, dict) else sorted(resolved)
    parts: list[str] = []
    for key in swept:
        try:
            index = int(key)
        except (TypeError, ValueError):
            continue
        spec = resolved.get(index)
        if spec is None:
            continue
        box = ",".join(str(value) for value in spec.box)
        parts.append(f"param[{index}] box=[{box}] swizzle={spec.swizzle_bytes}")
    return "; ".join(parts)
