from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest

from xla_ciq.search.tensor_map_search import (
    TILE_SPACE_ROOT,
    apply_tile_config,
    box_size_options,
    build_tile_search_space,
    default_sweep_indices,
    describe_tile_config,
    legal_box_sizes,
    swizzle_options,
    unpack_search_config,
)
from xla_ciq.xla.tensor_map import TensorMapError, TensorMapSpec


def _spec(**kwargs) -> TensorMapSpec:
    defaults = dict(
        dtype="bf16",
        dims=(2048, 8, 2),
        box=(32, 8, 1),
        swizzle_bytes=64,
        source="file:maps.json",
    )
    defaults.update(kwargs)
    return TensorMapSpec(**defaults)


@pytest.fixture
def fake_compileiq_search_space(monkeypatch):
    compileiq = ModuleType("compileiq")
    search_spaces = ModuleType("compileiq.search_spaces")
    base = ModuleType("compileiq.search_spaces.base")
    base.literal = lambda value: SimpleNamespace(value=value)
    base.range = lambda low, high, step, **kwargs: SimpleNamespace(
        low=low,
        high=high,
        step=step,
        **kwargs,
    )
    compileiq.search_spaces = search_spaces
    search_spaces.base = base
    monkeypatch.setitem(sys.modules, "compileiq", compileiq)
    monkeypatch.setitem(sys.modules, "compileiq.search_spaces", search_spaces)
    monkeypatch.setitem(sys.modules, "compileiq.search_spaces.base", base)


def test_legal_box_sizes_seed_first_and_align_innermost():
    # bf16 innermost must be multiple of 8 elements (16 bytes).
    sizes = legal_box_sizes(64, element_size=2, innermost=True, seed=32)
    assert sizes[0] == 32
    assert 1 not in sizes  # 2 bytes, not 16-aligned
    assert sizes == [32, 64, 16]  # seed ± one octave only


def test_legal_box_sizes_outer_axes_allow_any_positive_pow2():
    sizes = legal_box_sizes(8, element_size=2, innermost=False, seed=1)
    assert sizes[0] == 1
    assert sizes == [1, 2]  # seed ± one octave


def test_legal_box_sizes_without_seed_lists_pow2():
    sizes = legal_box_sizes(64, element_size=2, innermost=True, seed=None)
    assert 8 in sizes and 16 in sizes and 32 in sizes and 64 in sizes
    assert 1 not in sizes


def test_default_sweep_indices_prefer_guessed_layouts():
    specs = {
        1: _spec(source="module.txt:fusion.1"),
        4: _spec(source="guess:output_M=8"),
        5: _spec(source="file:maps.json"),
    }
    assert default_sweep_indices(specs) == (4, 5)


def test_default_sweep_indices_fall_back_to_all():
    specs = {1: _spec(source="module.txt:fusion.1")}
    assert default_sweep_indices(specs) == (1,)


def test_option_lists_put_the_seed_first():
    spec = _spec()
    assert box_size_options(spec, 0)[0] == 32
    assert box_size_options(spec, 1)[0] == 8
    assert box_size_options(spec, 2)[0] == 1
    assert swizzle_options(spec)[0] == 64


def test_build_tile_search_space_pins_init_to_the_seed_index(
    fake_compileiq_search_space,
):
    space = build_tile_search_space({4: _spec()}, sweep_indices=(4,))
    entry = space[TILE_SPACE_ROOT]["4"]
    assert set(entry) == {"b0", "b1", "b2", "swz"}
    # Seed layout is index 0, and init sampling is pinned there so a rigid
    # kernel still sees the layout that already launched.
    for knob in entry.values():
        assert knob.low == 0
        assert knob.seed_low == 0 and knob.seed_high == 0
    assert entry["b0"].high == len(box_size_options(_spec(), 0)) - 1


def test_build_tile_search_space_uses_literal_for_single_option(
    fake_compileiq_search_space,
):
    # A rank-1 descriptor whose only legal box is the seed has nothing to sweep.
    spec = _spec(dims=(8,), box=(8,), swizzle_bytes=0)
    entry = build_tile_search_space({4: spec}, sweep_indices=(4,))[TILE_SPACE_ROOT]["4"]
    assert entry["b0"].value == 0


def test_apply_tile_config_resolves_option_indices():
    base = {4: _spec()}
    updated = apply_tile_config(
        base,
        {TILE_SPACE_ROOT: {"4": {"b0": 1, "b1": 1, "b2": 0, "swz": 1}}},
    )
    assert updated[4].box == (64, 4, 1)
    assert updated[4].swizzle_bytes == 0
    assert updated[4].source.startswith("sweep:")


def test_apply_tile_config_index_zero_reproduces_the_seed():
    base = {4: _spec()}
    updated = apply_tile_config(
        base,
        {TILE_SPACE_ROOT: {"4": {"b0": 0, "b1": 0, "b2": 0, "swz": 0}}},
    )
    assert updated[4].box == base[4].box
    assert updated[4].swizzle_bytes == base[4].swizzle_bytes


def test_apply_tile_config_clamps_out_of_range_index():
    updated = apply_tile_config(
        {4: _spec()},
        {TILE_SPACE_ROOT: {"4": {"b0": 99}}},
    )
    assert updated[4].box[0] == box_size_options(_spec(), 0)[-1]


def test_apply_tile_config_rejects_unknown_param():
    with pytest.raises(TensorMapError, match="unknown param"):
        apply_tile_config({4: _spec()}, {TILE_SPACE_ROOT: {"7": {"b0": 0}}})


def test_unpack_search_config_single_and_multi():
    acf, tiles = unpack_search_config("deadbeef")
    assert acf == "deadbeef" and tiles == {}

    acf, tiles = unpack_search_config(["abcd", {TILE_SPACE_ROOT: {"4": {"b0": 1}}}])
    assert acf == "abcd"
    assert tiles[TILE_SPACE_ROOT]["4"]["b0"] == 1


def test_describe_tile_config_reports_resolved_layout():
    text = describe_tile_config(
        {4: _spec()},
        {TILE_SPACE_ROOT: {"4": {"b0": 1, "b1": 0, "b2": 0, "swz": 0}}},
    )
    assert "param[4]" in text
    assert "box=[64,8,1]" in text
    assert "swizzle=64" in text
