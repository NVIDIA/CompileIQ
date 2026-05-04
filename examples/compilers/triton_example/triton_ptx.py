"""
CompileIQ Triton Example: Optimize matmul kernel with PTXAS controls.

This example uses Triton's matmul kernel from the official tutorials and
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


# Matmul kernel from Triton tutorials (simplified config list)
@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_M": 8}, num_stages=3, num_warps=8
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 32, "GROUP_M": 8}, num_stages=4, num_warps=4
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8}, num_stages=4, num_warps=4
        ),
    ],
    key=["M", "N", "K"],
)
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
    GROUP_M: tl.constexpr,
    ACTIVATION: tl.constexpr,
):
    """Standard tiled matmul: C = A @ B"""
    pid = tl.program_id(0)
    num_pid_m, num_pid_n = tl.cdiv(M, BLOCK_M), tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c = acc.to(tl.float16)
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    tl.store(c_ptrs, c, mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N))


def matmul(a, b, acf_path: str):
    """Run matmul with custom PTXAS controls."""
    assert a.shape[1] == b.shape[0] and a.is_contiguous()
    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),)  # noqa: E731
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
        ACTIVATION="",
        ptx_options=f"--apply-controls={acf_path}",
    )
    return c


def objective(config: str) -> float:
    """Evaluate a PTXAS config: verify correctness, then benchmark."""
    os.environ["TRITON_PTXAS_PATH"] = shutil.which("ptxas")
    os.environ["TRITON_ALWAYS_COMPILE"] = "1"

    a = torch.rand((512, 512), device=DEVICE, dtype=torch.float16) - 0.5
    b = torch.rand((512, 512), device=DEVICE, dtype=torch.float16) - 0.5

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
    config = SearchConfiguration(problem_type="min", generations=5, pool_size=32)
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
        results = tuner.start()

    # Save best result
    best = results.get_best_result()
    print(f"Best runtime: {best['score_1']:.4f} ms")
    save_compiler_config("best_matmul.acf", best["params"])


if __name__ == "__main__":
    main()
