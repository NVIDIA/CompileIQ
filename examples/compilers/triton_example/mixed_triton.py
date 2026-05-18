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
from compileiq.types import INVALID_SCORE, SearchConfiguration
from compileiq.utils.helpers import save_compiler_config

DEVICE = triton.runtime.driver.active.get_active_torch_device()

# Triton configs to search over (user-defined space)
TRITON_CONFIGS = [
    {"block_m": 128, "block_n": 256, "block_k": 64, "group_m": 8, "stages": 3, "warps": 8},
    {"block_m": 64, "block_n": 256, "block_k": 32, "group_m": 8, "stages": 4, "warps": 4},
    {"block_m": 128, "block_n": 128, "block_k": 32, "group_m": 8, "stages": 4, "warps": 4},
    {"block_m": 128, "block_n": 64, "block_k": 32, "group_m": 8, "stages": 4, "warps": 4},
    {"block_m": 64, "block_n": 128, "block_k": 32, "group_m": 8, "stages": 4, "warps": 4},
]


@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr, ACTIVATION: tl.constexpr,
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


def matmul(a, b, acf_path: str, cfg: dict):
    """Run matmul with custom Triton config and PTXAS controls."""
    assert a.shape[1] == b.shape[0] and a.is_contiguous()
    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    grid = (triton.cdiv(M, cfg["block_m"]) * triton.cdiv(N, cfg["block_n"]),)
    matmul_kernel[grid](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
        BLOCK_M=cfg["block_m"], BLOCK_N=cfg["block_n"], BLOCK_K=cfg["block_k"],
        GROUP_M=cfg["group_m"], ACTIVATION="",
        num_stages=cfg["stages"], num_warps=cfg["warps"],
        ptx_options=f"--apply-controls={acf_path}",
    )
    return c


def objective(mixed_config: list) -> float:
    """Evaluate combined Triton + PTXAS config."""
    user_space, ptxas_config = mixed_config
    cfg = TRITON_CONFIGS[user_space["config_idx"]]

    os.environ["TRITON_PTXAS_PATH"] = shutil.which("ptxas")
    os.environ["TRITON_ALWAYS_COMPILE"] = "1"

    a = torch.rand((512, 512), device=DEVICE, dtype=torch.float16) - 0.5
    b = torch.rand((512, 512), device=DEVICE, dtype=torch.float16) - 0.5

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
    tuner = Search(
        objective_function=objective,
        search_space=search_space,
        search_config=config,
    )
    results = tuner.start()

    # Save best result
    best = results.get_best_result()
    print(f"Best runtime: {best['score_1']:.4f} ms")
    print(f"Best Triton config index: {best['user_space']['config_idx']}")
    save_compiler_config("best_matmul.acf", best["params"])


if __name__ == "__main__":
    main()
