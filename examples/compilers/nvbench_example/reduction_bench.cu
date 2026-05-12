/*
 * NVBench-instrumented CUDA reduction benchmark for CompileIQ optimization.
 *
 * Reduction using cub::BlockReduce, measured by NVBench instead of manual
 * cudaEvent timing. NVBench provides:
 *   - Cold measurements (L2 cache flushed between samples)
 *   - Entropy-based convergence (adapts sample count to noise)
 *   - Statistical rigor (full timing sample distribution)
 *
 * Build (requires NVBench installation):
 *   nvcc -O3 -std=c++17 -arch=sm_100 reduction_bench.cu \
 *     -I $NVBENCH_PATH/include -L $NVBENCH_PATH/lib \
 *     -Xlinker=-rpath,$NVBENCH_PATH/lib \
 *     -lnvbench -lcudart_static -lcuda \
 *     $NVBENCH_PATH/lib/objects-Release/nvbench.main/main.cu.o \
 *     -o reduction_bench
 *
 * Usage:
 *   ./reduction_bench -d 0 -b reduction -a "Elements[pow2]=26" \
 *     --no-batch --stopping-criterion entropy --jsonbin result.json
 */

#include <nvbench/nvbench.cuh>

#include <cub/block/block_reduce.cuh>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif
static constexpr int MAX_BLOCKS = 64;

// Reduction kernel using cub::BlockReduce.
// Each thread loads multiple elements via grid-stride loop (Brent's theorem)
// to keep the grid small, then CUB handles the block-wide reduction.
__global__ void reduce_kernel(const int *__restrict__ g_idata, int *__restrict__ g_odata,
                              unsigned int n)
{
    using BlockReduceT = cub::BlockReduce<int, BLOCK_SIZE>;
    __shared__ typename BlockReduceT::TempStorage temp_storage;

    unsigned int i = blockIdx.x * BLOCK_SIZE * 2 + threadIdx.x;
    unsigned int gridSize = BLOCK_SIZE * 2 * gridDim.x;

    // Grid-stride loop: each thread accumulates multiple elements
    int mySum = 0;
    while (i < n) {
        mySum += g_idata[i];
        if (i + BLOCK_SIZE < n)
            mySum += g_idata[i + BLOCK_SIZE];
        i += gridSize;
    }

    // Block-wide reduction via CUB
    int blockSum = BlockReduceT(temp_storage).Sum(mySum);

    if (threadIdx.x == 0)
        g_odata[blockIdx.x] = blockSum;
}

// NVBench benchmark function
void reduction_bench(nvbench::state &state)
{
    const auto n = static_cast<unsigned int>(state.get_int64("Elements"));
    if (n == 0) return;

    int numBlocks = (n + (BLOCK_SIZE * 2 - 1)) / (BLOCK_SIZE * 2);
    if (numBlocks > MAX_BLOCKS) numBlocks = MAX_BLOCKS;

    // Initialize host data with deterministic pattern
    std::vector<int> h_data(n);
    for (unsigned int i = 0; i < n; i++)
        h_data[i] = i & 0xFF;

    // RAII device memory via thrust (no manual cudaMalloc/cudaFree)
    thrust::device_vector<int> d_idata(h_data.begin(), h_data.end());
    thrust::device_vector<int> d_odata(numBlocks);

    int *d_idata_ptr = thrust::raw_pointer_cast(d_idata.data());
    int *d_odata_ptr = thrust::raw_pointer_cast(d_odata.data());

    // NVBench timed region: only the kernel launch is measured.
    // NVBench handles warm-up, cold measurements, and statistical convergence.
    state.exec([&](nvbench::launch &launch) {
        reduce_kernel<<<numBlocks, BLOCK_SIZE, 0, launch.get_stream()>>>(
            d_idata_ptr, d_odata_ptr, n);
    });
}

NVBENCH_BENCH(reduction_bench)
    .set_name("reduction")
    .add_int64_power_of_two_axis("Elements", {20, 22, 24, 26});
