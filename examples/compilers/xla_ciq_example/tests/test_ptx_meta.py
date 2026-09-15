from __future__ import annotations

import pytest

from xla_ciq.search.ptx_meta import inspect_ptx_entry, validate_synthetic_launch_params


WRAPPED_XOR_PTX = """
.visible .entry wrapped_xor(
	.param .u64 .ptr .align 256 wrapped_xor_param_0,
	.param .u64 .ptr .align 256 wrapped_xor_param_1,
	.param .u64 .ptr .align 256 wrapped_xor_param_2
)
.reqntid 4, 1, 1
{
	ret;
}
"""


def test_inspect_ptx_entry_reqntid_and_params():
    meta = inspect_ptx_entry(WRAPPED_XOR_PTX, "wrapped_xor")
    assert meta.param_count == 3
    assert meta.reqntid == (4, 1, 1)
    assert meta.block_size == 4


def test_inspect_ptx_entry_missing():
    with pytest.raises(ValueError, match="not found"):
        inspect_ptx_entry(WRAPPED_XOR_PTX, "missing_kernel")


def test_resolve_block_size_prefers_reqntid():
    from xla_ciq.search.ptx_meta import resolve_block_size

    meta = inspect_ptx_entry(WRAPPED_XOR_PTX, "wrapped_xor")
    assert resolve_block_size(meta, 256) == 4


def test_pointer_params_are_not_tensor_maps():
    meta = inspect_ptx_entry(WRAPPED_XOR_PTX, "wrapped_xor")
    assert meta.tensor_map_indices == ()
    assert not meta.uses_tensor_maps
    assert all(p.is_pointer for p in meta.params)
    validate_synthetic_launch_params(meta)


def test_synthetic_launch_rejects_scalar_value_params():
    ptx = """
.visible .entry scalar_kernel(
    .param .u64 .ptr scalar_kernel_param_0,
    .param .u32 scalar_kernel_param_1
)
{
    ret;
}
"""
    meta = inspect_ptx_entry(ptx, "scalar_kernel")

    with pytest.raises(ValueError, match="scalar_kernel_param_1"):
        validate_synthetic_launch_params(meta)


# A TMA store addresses the descriptor through the operand carrying the
# coordinate list, which for `.global.shared` is the first bracket.
TMA_STORE_PTX = """
.visible .entry tma_store(
	.param .u64 .ptr .align 256 tma_store_param_0,
	.param .align 64 .b8 tma_store_param_1[128]
)
.reqntid 128, 1, 1
{
	ld.param.b64 	%rd4, [tma_store_param_0];
	mov.b64 	%rd6, tma_store_param_1;
	cvta.param.u64 	%rd3, %rd6;
	cp.async.bulk.tensor.5d.global.shared::cta.bulk_group [%rd3, {%r10, %r5, %r5, %r5, %r11}], [%r12];
	ret;
}
"""

# A TMA load reverses the operands, so the descriptor is the second bracket.
TMA_LOAD_PTX = """
.visible .entry tma_load(
	.param .align 64 .b8 tma_load_param_0[128],
	.param .u64 .ptr .align 256 tma_load_param_1
)
.reqntid 32, 1, 1
{
	mov.b64 	%rd3, tma_load_param_0;
	ld.param.b64 	%rd4, [tma_load_param_1];
	cvta.param.u64 	%rd1, %rd3;
	cp.async.bulk.tensor.1d.shared::cta.global.mbarrier::complete_tx::bytes [%r2], [%rd1, {%r3}], [%r1];
	ret;
}
"""


def test_detects_tensor_map_store_param():
    meta = inspect_ptx_entry(TMA_STORE_PTX, "tma_store")
    assert meta.tensor_map_indices == (1,)
    descriptor = meta.params[1]
    assert descriptor.is_tensor_map
    assert descriptor.nbytes == 128
    assert descriptor.tensor_rank == 5
    assert descriptor.tensor_direction == "store"
    assert meta.params[0].is_pointer


def test_detects_tensor_map_load_param():
    meta = inspect_ptx_entry(TMA_LOAD_PTX, "tma_load")
    assert meta.tensor_map_indices == (0,)
    descriptor = meta.params[0]
    assert descriptor.tensor_rank == 1
    assert descriptor.tensor_direction == "load"
