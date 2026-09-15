"""Minimal CUDA driver helpers for loading cubin and timing a kernel."""

from __future__ import annotations

import ctypes
import statistics
from dataclasses import dataclass

try:
    from cuda.bindings import driver as cuda
except ImportError:  # Keep mock and inspection commands usable without CUDA.
    cuda = None

from xla_ciq.xla.tensor_map import TensorMapError, TensorMapSpec

# CUtensorMap is an opaque 128-byte struct that must be 64-byte aligned both
# when the driver fills it in and when it is handed to a kernel by value.
TENSOR_MAP_NBYTES = 128
_TENSOR_MAP_ALIGN = 64

_TENSOR_MAP_DTYPES = {
    "u8": "UINT8",
    "s8": "UINT8",
    "pred": "UINT8",
    "u16": "UINT16",
    "s16": "UINT16",
    "f16": "FLOAT16",
    "bf16": "BFLOAT16",
    "u32": "UINT32",
    "s32": "INT32",
    "f32": "FLOAT32",
    "u64": "UINT64",
    "s64": "INT64",
    "f64": "FLOAT64",
}

_SWIZZLE_NAMES = {0: "NONE", 32: "32B", 64: "64B", 128: "128B"}

# A kernel gets 48 KiB of dynamic shared memory unless it opts in, and Hopper
# tile kernels routinely ask for more, so the opt-in is part of launching them.
_DEFAULT_MAX_DYNAMIC_SHARED = 48 * 1024


class CudaError(RuntimeError):
    """CUDA driver API failure."""


def _require_cuda_bindings() -> None:
    if cuda is None:
        raise CudaError(
            "CUDA Python driver bindings are unavailable; install a compatible "
            "cuda-python package for real GPU execution"
        )


def _check(result):
    if isinstance(result, tuple):
        err, *rest = result
    else:
        err, rest = result, []
    if err != cuda.CUresult.CUDA_SUCCESS:
        raise CudaError(str(err))
    if not rest:
        return None
    if len(rest) == 1:
        return rest[0]
    return tuple(rest)


def _as_dim3(value: int | tuple[int, int, int]) -> tuple[int, int, int]:
    if isinstance(value, tuple):
        if len(value) != 3:
            raise ValueError(f"expected 3-tuple grid/block, got {value!r}")
        return int(value[0]), int(value[1]), int(value[2])
    return int(value), 1, 1


@dataclass
class TimedLaunchResult:
    median_ms: float
    samples_ms: list[float]


def _aligned_buffer(payload: bytes, align: int) -> tuple[ctypes.Array, int]:
    """Host buffer holding payload at an `align`-byte boundary."""
    backing = (ctypes.c_char * (len(payload) + align))()
    base = ctypes.addressof(backing)
    offset = (-base) % align
    ctypes.memmove(base + offset, payload, len(payload))
    return backing, base + offset


def encode_tensor_map(
    spec: TensorMapSpec, *, device_ptr: int, swizzle_bytes: int
) -> bytes:
    """Build the 128-byte CUtensorMap describing `device_ptr` with this layout."""
    _require_cuda_bindings()
    spec.validate()
    dtype_name = _TENSOR_MAP_DTYPES.get(spec.dtype)
    if dtype_name is None:
        raise TensorMapError(f"no CUtensorMap data type for dtype {spec.dtype!r}")
    swizzle_name = _SWIZZLE_NAMES.get(swizzle_bytes)
    if swizzle_name is None:
        raise TensorMapError(f"invalid swizzle: {swizzle_bytes}")

    encoded = _check(
        cuda.cuTensorMapEncodeTiled(
            getattr(cuda.CUtensorMapDataType, f"CU_TENSOR_MAP_DATA_TYPE_{dtype_name}"),
            spec.rank,
            device_ptr,
            [cuda.cuuint64_t(x) for x in spec.dims],
            [cuda.cuuint64_t(x) for x in spec.strides],
            [cuda.cuuint32_t(x) for x in spec.box],
            [cuda.cuuint32_t(x) for x in spec.effective_element_strides],
            cuda.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE,
            getattr(cuda.CUtensorMapSwizzle, f"CU_TENSOR_MAP_SWIZZLE_{swizzle_name}"),
            cuda.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
            cuda.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
        )
    )
    return bytes((ctypes.c_char * TENSOR_MAP_NBYTES).from_address(int(encoded.getPtr())))


class CudaContext:
    def __init__(self) -> None:
        _require_cuda_bindings()
        _check(cuda.cuInit(0))
        self.device = _check(cuda.cuDeviceGet(0))
        self._ctx = _check(cuda.cuDevicePrimaryCtxRetain(self.device))
        _check(cuda.cuCtxSetCurrent(self._ctx))

    def close(self) -> None:
        _check(cuda.cuDevicePrimaryCtxRelease(self.device))

    def __enter__(self) -> CudaContext:
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def load_cubin(self, cubin_path: str):
        # cuda-python driver bindings expect bytes paths.
        return _check(cuda.cuModuleLoad(cubin_path.encode("utf-8")))

    def get_function(self, module, name: str):
        return _check(cuda.cuModuleGetFunction(module, name.encode("utf-8")))

    def alloc(self, nbytes: int):
        return _check(cuda.cuMemAlloc(nbytes))

    def free(self, ptr) -> None:
        _check(cuda.cuMemFree(ptr))

    def memset(self, ptr, value: int, nbytes: int) -> None:
        _check(cuda.cuMemsetD8(ptr, value, nbytes))

    def unload(self, module) -> None:
        _check(cuda.cuModuleUnload(module))

    def allow_dynamic_shared_memory(self, func, nbytes: int) -> None:
        """Raise this kernel's dynamic shared memory cap above the 48 KiB default."""
        if nbytes <= _DEFAULT_MAX_DYNAMIC_SHARED:
            return
        _check(
            cuda.cuFuncSetAttribute(
                func,
                cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                int(nbytes),
            )
        )

    def time_launch(
        self,
        func,
        *,
        grid: int | tuple[int, int, int],
        block: int | tuple[int, int, int],
        arg_values: list,
        trials: int,
        warmup: int = 2,
        shared_memory_bytes: int = 0,
    ) -> TimedLaunchResult:
        gx, gy, gz = _as_dim3(grid)
        bx, by, bz = _as_dim3(block)
        start = _check(cuda.cuEventCreate(0))
        end = _check(cuda.cuEventCreate(0))
        samples: list[float] = []
        packed, _holders = _pack_kernel_args(arg_values)
        try:
            for i in range(warmup + trials):
                _check(cuda.cuEventRecord(start, 0))
                _check(
                    cuda.cuLaunchKernel(
                        func,
                        gx,
                        gy,
                        gz,
                        bx,
                        by,
                        bz,
                        int(shared_memory_bytes),
                        0,
                        packed,
                        0,
                    )
                )
                _check(cuda.cuEventRecord(end, 0))
                _check(cuda.cuEventSynchronize(end))
                ms = _check(cuda.cuEventElapsedTime(start, end))
                if i >= warmup:
                    samples.append(float(ms))
        finally:
            _check(cuda.cuEventDestroy(start))
            _check(cuda.cuEventDestroy(end))
        return TimedLaunchResult(
            median_ms=float(statistics.median(samples)),
            samples_ms=samples,
        )


def _pack_kernel_args(arg_values: list) -> tuple[ctypes.Array, list]:
    """Build the void** kernel parameter array.

    Each slot holds the address of the argument's value: a device pointer for
    buffer arguments, or the descriptor bytes themselves for by-value
    arguments such as a CUtensorMap.
    """
    holders: list = []
    slots: list[ctypes.c_void_p] = []
    for value in arg_values:
        if isinstance(value, (bytes, bytearray)):
            backing, address = _aligned_buffer(bytes(value), _TENSOR_MAP_ALIGN)
            holders.append(backing)
            slots.append(ctypes.c_void_p(address))
        else:
            holder = ctypes.c_uint64(int(value))
            holders.append(holder)
            slots.append(ctypes.cast(ctypes.addressof(holder), ctypes.c_void_p))
    packed = (ctypes.c_void_p * len(slots))(*slots)
    return packed, holders
