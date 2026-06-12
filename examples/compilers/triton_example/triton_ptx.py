"""
CompileIQ Triton Example: Optimize a fixed-config matmul kernel with PTXAS controls.

This example uses an illustrative Triton matmul kernel and
applies CompileIQ to search for optimal PTXAS compiler configurations.

Usage:
    python triton_ptx.py
"""

import os
import re
import shutil
import subprocess
import tempfile

import torch
import triton
import triton.language as tl

from compileiq.ciq import Search
from compileiq.search_spaces.compilers import PtxasSearchSpace
from compileiq.types import INVALID_SCORE, SearchConfiguration
from compileiq.utils.helpers import save_compiler_config
from compileiq.utils.gpu import gpu_benchmark_mode

DEVICE = triton.runtime.driver.active.get_active_torch_device()

BLOCK_M = 32
BLOCK_N = 64
BLOCK_K = 32
NUM_WARPS = 4
NUM_STAGES = 3


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


def matmul(a, b, controls_path: str):
    """Run the fixed-config matmul with custom PTXAS controls."""
    assert a.shape[1] == b.shape[0] and a.is_contiguous()
    M, K = a.shape
    _, N = b.shape
    assert M % BLOCK_M == 0
    assert N % BLOCK_N == 0
    assert K % BLOCK_K == 0
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    grid = ((M // BLOCK_M) * (N // BLOCK_N),)
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
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=NUM_WARPS,
        num_stages=NUM_STAGES,
        ptx_options=ptxas_options,
    )
    return c


def objective(config) -> float:
    """Evaluate a PTXAS config: verify correctness, then benchmark."""
    ptxas_path = shutil.which("ptxas")
    os.environ["TRITON_PTXAS_PATH"] = ptxas_path
    os.environ["TRITON_PTXAS_BLACKWELL_PATH"] = ptxas_path
    os.environ["TRITON_ALWAYS_COMPILE"] = "1"

    a = torch.rand((4096, 4096), device=DEVICE, dtype=torch.float16) - 0.5
    b = torch.rand((4096, 4096), device=DEVICE, dtype=torch.float16) - 0.5

    with tempfile.NamedTemporaryFile(suffix=".acf", delete=True) as f:
        save_compiler_config(f.name, config)
        triton_out = matmul(a, b, f.name)
        torch_out = torch.matmul(a, b)

        if not torch.allclose(triton_out, torch_out, atol=1e-2, rtol=0):
            return INVALID_SCORE

        return triton.testing.do_bench(
            lambda: matmul(a, b, f.name), warmup=100, rep=1000, return_mode="mean"
        )


def main():
    # Check CUDA version
    version_output = subprocess.run(
        ["ptxas", "--version"], capture_output=True, text=True, check=True
    ).stdout
    cuda_version = re.search(r"release (\d+\.\d+),", version_output).group(1)
    assert float(cuda_version) >= 13.3, "CompileIQ requires CUDA 13.3+"

    # Configure and run search
    config = SearchConfiguration(problem_type="min", generations=5)
    tuner = Search(
        objective_function=objective,
        search_space=PtxasSearchSpace(version=cuda_version),
        search_config=config,
    )

    # Locking gpu clocks before any latency measurements is crucial to prevent
    # large variations during the search.
    # Please set the clock speeds according to your GPU capabilities.
    # If using multiple machines with Ray, use `gpu_benchmark_mode` inside the `objective` function.
    with gpu_benchmark_mode(clock_mhz=1965, raise_on_failure=False):
        results = tuner.start(task_timeout=20)

    # Save best result
    best = results.get_best_result()
    print(f"Best runtime: {best['score_1']:.4f} ms")
    save_compiler_config("best_matmul.acf", best["params"])


if __name__ == "__main__":
    main()
