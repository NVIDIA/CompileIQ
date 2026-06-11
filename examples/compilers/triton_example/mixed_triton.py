"""
CompileIQ Triton Example: Mixed search space optimization.

This example searches BOTH:
  - User-defined space: Triton kernel configs (block sizes, warps, stages)
  - Compiler space: PTXAS advanced controls

Usage:
    python mixed_triton.py
"""

import os
import re
import shutil
import subprocess
import tempfile

import torch
import triton
import triton.language as tl

import compileiq.search_spaces.base as ss
from compileiq.ciq import Search
from compileiq.search_spaces.compilers import PtxasSearchSpace
from compileiq.types import INVALID_SCORE, SearchConfiguration, WorkerTypes
from compileiq.utils.gpu import gpu_benchmark_mode
from compileiq.utils.helpers import save_compiler_config

DEVICE = triton.runtime.driver.active.get_active_torch_device()

# Triton configs to search over (user-defined space)
TRITON_CONFIGS = [
    {"block_m": 32, "block_n": 64, "block_k": 32, "stages": 3, "warps": 4},
    {"block_m": 64, "block_n": 64, "block_k": 32, "stages": 3, "warps": 4},
    {"block_m": 64, "block_n": 128, "block_k": 32, "stages": 4, "warps": 4},
    {"block_m": 128, "block_n": 128, "block_k": 32, "stages": 4, "warps": 4},
    {"block_m": 128, "block_n": 256, "block_k": 64, "stages": 3, "warps": 8},
]


def is_blackwell_gpu() -> bool:
    """Return True for Blackwell-class CUDA devices."""
    major, _ = torch.cuda.get_device_capability(DEVICE)
    return major >= 10


@triton.jit
def matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Simple exact-tile matmul: C = A @ B."""
    pid = tl.program_id(0)
    num_pid_n = N // BLOCK_N
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        k_idxs = k + offs_k
        a_ptrs = a_ptr + offs_m[:, None] * stride_am + k_idxs[None, :] * stride_ak
        b_ptrs = b_ptr + k_idxs[:, None] * stride_bk + offs_n[None, :] * stride_bn
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        acc += tl.dot(a, b)

    c = acc.to(tl.float16)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, c)


def matmul(a, b, controls_path: str, cfg: dict):
    """Run the exact-tile matmul with custom Triton config and PTXAS controls."""
    assert a.shape[1] == b.shape[0] and a.is_contiguous()
    M, K = a.shape
    _, N = b.shape
    assert M % cfg["block_m"] == 0
    assert N % cfg["block_n"] == 0
    assert K % cfg["block_k"] == 0
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    grid = ((M // cfg["block_m"]) * (N // cfg["block_n"]),)
    ptxas_options = f"--apply-controls={controls_path}" if controls_path else None
    matmul_kernel[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        BLOCK_M=cfg["block_m"],
        BLOCK_N=cfg["block_n"],
        BLOCK_K=cfg["block_k"],
        num_stages=cfg["stages"],
        num_warps=cfg["warps"],
        ptx_options=ptxas_options,
    )
    return c


def objective(mixed_config: list) -> float:
    """Evaluate combined Triton + PTXAS config."""
    user_space, ptxas_config = mixed_config
    cfg = TRITON_CONFIGS[user_space["config_idx"]]

    ptxas_path = shutil.which("ptxas")
    os.environ["TRITON_PTXAS_PATH"] = ptxas_path
    os.environ["TRITON_PTXAS_BLACKWELL_PATH"] = ptxas_path
    os.environ["TRITON_ALWAYS_COMPILE"] = "1"

    a = torch.rand((4096, 4096), device=DEVICE, dtype=torch.float16) - 0.5
    b = torch.rand((4096, 4096), device=DEVICE, dtype=torch.float16) - 0.5

    with tempfile.NamedTemporaryFile(suffix=".acf", delete=True) as f:
        save_compiler_config(f.name, ptxas_config)
        triton_out = matmul(a, b, f.name, cfg)
        torch_out = torch.matmul(a, b)

        if not torch.allclose(triton_out, torch_out, atol=1e-2, rtol=0):
            return INVALID_SCORE

        return triton.testing.do_bench(
            lambda: matmul(a, b, f.name, cfg), warmup=100, rep=1000, return_mode="mean"
        )


def main():
    # Check CUDA version
    version_output = subprocess.run(
        ["ptxas", "--version"], capture_output=True, text=True, check=True
    ).stdout
    cuda_version = re.search(r"release (\d+\.\d+),", version_output).group(1)
    assert float(cuda_version) >= 13.3, "CompileIQ requires CUDA 13.3+"

    # Define mixed search space: [user_space, compiler_space]
    search_space = [
        {"config_idx": ss.range(0, len(TRITON_CONFIGS) - 1)},  # User-defined
        PtxasSearchSpace(version=cuda_version),
    ]

    # Configure and run search
    config = SearchConfiguration(problem_type="min", generations=5, pool_size=32)
    search_kwargs = {}
    if is_blackwell_gpu():
        # This specific example may cause Illegal Memory accesses
        # We leverage fully isolated processes to prevent leaking across runs
        os.environ["CIQ_PROCESS_MODE"] = "spawn"
        search_kwargs["worker_type"] = WorkerTypes.ISOLATED

    tuner = Search(
        objective_function=objective,
        search_space=search_space,
        search_config=config,
        **search_kwargs,
    )
    with gpu_benchmark_mode(clock_mhz=1965, raise_on_failure=False):
        results = tuner.start(task_timeout=20)

    # Save best result
    best = results.get_best_result()
    user_space, ptxas_config = best["params"]
    print(f"Best runtime: {best['score_1']:.4f} ms")
    print(f"Best Triton config index: {user_space['config_idx']}")
    save_compiler_config("best_matmul.acf", ptxas_config)


if __name__ == "__main__":
    main()
