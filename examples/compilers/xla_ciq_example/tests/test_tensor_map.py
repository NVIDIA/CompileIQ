from __future__ import annotations

import json

import pytest

from xla_ciq.search.ptx_meta import inspect_ptx_entry
from xla_ciq.search.launch_probe import resolve_swizzle_options
from xla_ciq.search.tensor_map_plan import resolve_tensor_maps
from xla_ciq.xla.tensor_map import (
    TensorMapError,
    TensorMapSpec,
    find_fusion_layout,
    from_hlo_shape,
    load_tensor_map_file,
)

from tests.test_ptx_meta import TMA_LOAD_PTX, TMA_STORE_PTX


# Mirrors a real XLA/Triton transpose fusion: bf16[2,3,2,8,2048] tiled by
# [1,3,2,8,32], stored through a rank-5 tensor map.
HLO_LINE = (
    "  %input_transpose_fusion.1 = bf16[2,3,2,8,2048]{4,3,2,1,0} "
    "fusion(%gemm_fusion_dot.42), kind=kCustom, calls=%fused_transpose.1, "
    'backend_config={"fusion_backend_config":{"kind":"__triton",'
    '"block_level_fusion_config":{"num_warps":"4",'
    '"output_tiles":[{"sizes":["1","3","2","8","32"]}],"num_ctas":1,'
    '"is_tma_allowed":true}}}\n'
)


def test_hlo_order_is_reversed_into_cuda_order():
    spec = from_hlo_shape(
        shape_token="bf16[2,3,2,8,2048]",
        tile=[1, 3, 2, 8, 32],
        minor_to_major=[4, 3, 2, 1, 0],
    )
    assert spec.dims == (2048, 8, 2, 3, 2)
    assert spec.box == (32, 8, 2, 3, 1)
    assert spec.rank == 5


def test_strides_are_cumulative_bytes_excluding_innermost():
    spec = from_hlo_shape(shape_token="bf16[2,3,2,8,2048]", tile=[1, 3, 2, 8, 32])
    assert spec.strides == (4096, 32768, 65536, 196608)
    assert spec.nbytes == 393216


def test_non_default_layout_permutes_dims():
    spec = from_hlo_shape(
        shape_token="f32[4,8]", tile=[4, 4], minor_to_major=[0, 1]
    )
    assert spec.dims == (4, 8)
    assert spec.box == (4, 4)


def test_tile_rank_must_match_shape_rank():
    with pytest.raises(TensorMapError, match="tile rank"):
        from_hlo_shape(shape_token="f32[4,8]", tile=[4])


def test_box_may_not_exceed_dims():
    spec = TensorMapSpec(dtype="f32", dims=(8,), box=(16,))
    with pytest.raises(TensorMapError, match="exceeds dim"):
        spec.validate()


def test_innermost_box_must_be_16_byte_aligned():
    spec = TensorMapSpec(dtype="f32", dims=(64,), box=(2,))
    with pytest.raises(TensorMapError, match="multiple of 16 bytes"):
        spec.validate()


def test_rank_is_capped_at_five():
    spec = TensorMapSpec(dtype="f32", dims=(4,) * 6, box=(4,) * 6)
    with pytest.raises(TensorMapError, match="exceeds 5"):
        spec.validate()


def test_swizzle_candidates_fit_the_innermost_tile():
    # 32 bf16 elements is 64 bytes, so a 128B swizzle cannot apply.
    spec = from_hlo_shape(shape_token="bf16[2,3,2,8,2048]", tile=[1, 3, 2, 8, 32])
    assert spec.swizzle_candidates() == (64, 32, 0)


def test_explicit_swizzle_is_the_only_candidate():
    spec = TensorMapSpec(dtype="f32", dims=(64,), box=(64,), swizzle_bytes=32)
    assert spec.swizzle_candidates() == (32,)


def test_swizzle_options_intersect_across_descriptors():
    wide = TensorMapSpec(dtype="f32", dims=(256,), box=(64,))
    narrow = TensorMapSpec(dtype="bf16", dims=(256,), box=(32,))
    assert resolve_swizzle_options({0: wide}) == [128, 64, 32, 0]
    assert resolve_swizzle_options({0: wide, 1: narrow}) == [64, 32, 0]


def test_swizzle_options_for_fully_resolved_specs():
    spec = TensorMapSpec(dtype="f32", dims=(64,), box=(64,), swizzle_bytes=0)
    assert resolve_swizzle_options({0: spec}) == [0]


def test_load_spec_file_accepts_hlo_authoring_form(tmp_path):
    path = tmp_path / "maps.json"
    path.write_text(
        json.dumps(
            {
                "gemm_fusion_dot_83": {
                    "params": {
                        "0": {"hlo_shape": "f32[4,4,8,2048]", "tile": [1, 1, 8, 64]}
                    }
                }
            }
        )
    )
    specs = load_tensor_map_file(path)["gemm_fusion_dot_83"]
    assert specs[0].dims == (2048, 8, 4, 4)
    assert specs[0].box == (64, 8, 1, 1)


def test_load_spec_file_accepts_cuda_order_form(tmp_path):
    path = tmp_path / "maps.json"
    path.write_text(
        json.dumps(
            {
                "k": {
                    "params": {
                        "2": {
                            "dtype": "f32",
                            "dims": [2048, 8],
                            "box": [64, 8],
                            "swizzle_bytes": 128,
                        }
                    }
                }
            }
        )
    )
    spec = load_tensor_map_file(path)["k"][2]
    assert spec.dims == (2048, 8)
    assert spec.swizzle_bytes == 128


def test_load_spec_file_rejects_incomplete_spec(tmp_path):
    path = tmp_path / "maps.json"
    path.write_text(json.dumps({"k": {"params": {"0": {"dtype": "f32"}}}}))
    with pytest.raises(TensorMapError, match="missing keys"):
        load_tensor_map_file(path)


def test_find_fusion_layout_prefers_the_ptx_module(tmp_path):
    (tmp_path / "module_1.jit_x.after_optimizations.txt").write_text(
        HLO_LINE.replace("2,3,2,8,2048", "9,9,9,9,9")
    )
    (tmp_path / "module_2.jit_x.after_optimizations.txt").write_text(HLO_LINE)
    layout = find_fusion_layout(
        tmp_path,
        "input_transpose_fusion_1",
        ptx_path=tmp_path / "module_2.jit_x.35.ptx",
    )
    assert layout is not None
    assert layout.shape_token == "bf16[2,3,2,8,2048]"
    assert layout.tile == (1, 3, 2, 8, 32)
    assert layout.minor_to_major == (4, 3, 2, 1, 0)


def test_plan_is_empty_for_kernels_without_descriptors():
    from tests.test_ptx_meta import WRAPPED_XOR_PTX

    meta = inspect_ptx_entry(WRAPPED_XOR_PTX, "wrapped_xor")
    plan = resolve_tensor_maps(meta=meta)
    assert plan.is_complete
    assert not plan.required


def test_plan_infers_a_store_descriptor_from_the_dump(tmp_path):
    (tmp_path / "module_2.jit_x.after_optimizations.txt").write_text(
        HLO_LINE.replace("input_transpose_fusion.1", "tma_store")
    )
    meta = inspect_ptx_entry(TMA_STORE_PTX, "tma_store")
    plan = resolve_tensor_maps(meta=meta, dump_dir=tmp_path)
    assert plan.is_complete
    assert plan.specs[1].dims == (2048, 8, 2, 3, 2)
    assert plan.specs[1].box == (32, 8, 2, 3, 1)


def test_plan_reports_load_descriptors_as_unresolved(tmp_path):
    meta = inspect_ptx_entry(TMA_LOAD_PTX, "tma_load")
    plan = resolve_tensor_maps(meta=meta, dump_dir=tmp_path)
    assert not plan.is_complete
    assert plan.unresolved == (0,)
    gap = plan.describe_gap(meta)
    assert "tma_load_param_0" in gap
    assert "--tensor-maps" in gap


def test_plan_overrides_take_precedence_over_inference(tmp_path):
    (tmp_path / "module_2.jit_x.after_optimizations.txt").write_text(
        HLO_LINE.replace("input_transpose_fusion.1", "tma_store")
    )
    override = TensorMapSpec(dtype="f32", dims=(64, 4), box=(64, 2))
    meta = inspect_ptx_entry(TMA_STORE_PTX, "tma_store")
    plan = resolve_tensor_maps(meta=meta, dump_dir=tmp_path, overrides={1: override})
    assert plan.is_complete
    assert plan.specs[1] is override


def test_search_failure_points_at_inferred_layouts():
    from xla_ciq.search.compileiq_runtime import _format_search_failure

    inferred = TensorMapSpec(
        dtype="bf16", dims=(2048, 8), box=(32, 8), source="dump.txt:fusion.1"
    )
    message = _format_search_failure(
        RuntimeError("All objective functions failed in the first gen."),
        tensor_maps={1: inferred},
    )
    assert "--tensor-maps" in message
    assert "parameter(s) 1" in message
    assert "CIQ_DEBUG_OBJECTIVE" in message


def test_search_failure_stays_terse_without_descriptors():
    from xla_ciq.search.compileiq_runtime import _format_search_failure

    message = _format_search_failure(RuntimeError("boom"), tensor_maps={})
    assert message == "CompileIQ search failed: boom"


def test_search_failure_does_not_blame_explicit_layouts():
    from xla_ciq.search.compileiq_runtime import _format_search_failure

    explicit = TensorMapSpec(
        dtype="bf16", dims=(2048, 8), box=(32, 8), source="file:maps.json"
    )
    message = _format_search_failure(
        RuntimeError("All objective functions failed in the first gen."),
        tensor_maps={1: explicit},
    )
    assert "parameter(s)" not in message
    assert "shared-memory layout" in message


def test_plan_rejects_a_rank_mismatch(tmp_path):
    (tmp_path / "module_2.jit_x.after_optimizations.txt").write_text(
        HLO_LINE.replace("input_transpose_fusion.1", "tma_store")
        .replace("bf16[2,3,2,8,2048]{4,3,2,1,0}", "bf16[8,2048]{1,0}")
        .replace('["1","3","2","8","32"]', '["8","32"]')
    )
    meta = inspect_ptx_entry(TMA_STORE_PTX, "tma_store")
    plan = resolve_tensor_maps(meta=meta, dump_dir=tmp_path)
    assert not plan.is_complete
    assert any("rank" in note for note in plan.notes)
