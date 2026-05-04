/*
 * Self-contained CUDA reduction benchmark for CompileIQ optimization.
 *
 * Compiles in a single step:
 *   nvcc -O3 -std=c++17 -arch=sm_100 reduction.cu -o reduction
 *
 * Usage:
 *   ./reduction [-n=67108864]
 */

#include <cooperative_groups.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

#define CUDA_CHECK(call)                                                                    \
    do {                                                                                    \
        cudaError_t err = (call);                                                           \
        if (err != cudaSuccess) {                                                           \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,                \
                    cudaGetErrorString(err));                                                \
            exit(1);                                                                        \
        }                                                                                   \
    } while (0)

static constexpr int BLOCK_SIZE = 256;
static constexpr int MAX_BLOCKS = 64;
static constexpr int TEST_ITERATIONS = 100;

// Warp-level reduction via shuffle
__device__ __forceinline__ int warpReduceSum(int val)
{
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// Shared-memory reduction with warp shuffle for the final warp.
// Each thread loads multiple elements (Brent's theorem) to keep the grid small.
template <int BlockSize>
__global__ void reduce_kernel(const int *__restrict__ g_idata, int *__restrict__ g_odata,
                              unsigned int n)
{
    extern __shared__ int sdata[];
    cg::thread_block cta = cg::this_thread_block();

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * BlockSize * 2 + threadIdx.x;
    unsigned int gridSize = BlockSize * 2 * gridDim.x;

    // Grid-stride loop: each thread accumulates multiple elements
    int mySum = 0;
    while (i < n) {
        mySum += g_idata[i];
        if (i + BlockSize < n)
            mySum += g_idata[i + BlockSize];
        i += gridSize;
    }
    sdata[tid] = mySum;
    cg::sync(cta);

    // Tree reduction in shared memory (compile-time unrolled)
    if (BlockSize >= 512 && tid < 256) sdata[tid] = mySum = mySum + sdata[tid + 256];
    cg::sync(cta);
    if (BlockSize >= 256 && tid < 128) sdata[tid] = mySum = mySum + sdata[tid + 128];
    cg::sync(cta);
    if (BlockSize >= 128 && tid < 64) sdata[tid] = mySum = mySum + sdata[tid + 64];
    cg::sync(cta);

    // Final warp: shuffle reduction
    cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);
    if (cta.thread_rank() < 32) {
        if (BlockSize >= 64)
            mySum += sdata[tid + 32];
        for (int offset = tile32.size() / 2; offset > 0; offset /= 2)
            mySum += tile32.shfl_down(mySum, offset);
    }

    if (cta.thread_rank() == 0)
        g_odata[blockIdx.x] = mySum;
}

// CPU reference using Kahan summation
static long long reduceCPU(const int *data, int n)
{
    long long sum = 0;
    for (int i = 0; i < n; i++)
        sum += data[i];
    return sum;
}

static int parseIntArg(int argc, char **argv, const char *name, int defaultVal)
{
    for (int i = 1; i < argc; i++) {
        if (strncmp(argv[i], name, strlen(name)) == 0 && argv[i][strlen(name)] == '=')
            return atoi(argv[i] + strlen(name) + 1);
    }
    return defaultVal;
}

int main(int argc, char **argv)
{
    int n = parseIntArg(argc, argv, "-n", 1 << 26);  // 67108864

    // Allocate and initialize host data
    int *h_idata = new int[n];
    for (int i = 0; i < n; i++)
        h_idata[i] = rand() & 0xFF;

    // Compute grid dimensions
    int numBlocks = (n + (BLOCK_SIZE * 2 - 1)) / (BLOCK_SIZE * 2);
    if (numBlocks > MAX_BLOCKS) numBlocks = MAX_BLOCKS;

    // Allocate device memory
    int *d_idata, *d_odata;
    CUDA_CHECK(cudaMalloc(&d_idata, (size_t)n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_odata, numBlocks * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_idata, h_idata, (size_t)n * sizeof(int), cudaMemcpyHostToDevice));

    int smem = BLOCK_SIZE * sizeof(int);

    // Warm-up
    reduce_kernel<BLOCK_SIZE><<<numBlocks, BLOCK_SIZE, smem>>>(d_idata, d_odata, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timed iterations using CUDA events
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < TEST_ITERATIONS; i++)
        reduce_kernel<BLOCK_SIZE><<<numBlocks, BLOCK_SIZE, smem>>>(d_idata, d_odata, n);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float totalMs = 0;
    CUDA_CHECK(cudaEventElapsedTime(&totalMs, start, stop));
    double avgTimeSec = (totalMs / TEST_ITERATIONS) / 1000.0;

    // Read back block partial sums and finalize on CPU
    int *h_odata = new int[numBlocks];
    CUDA_CHECK(cudaMemcpy(h_odata, d_odata, numBlocks * sizeof(int), cudaMemcpyDeviceToHost));
    long long gpuResult = 0;
    for (int i = 0; i < numBlocks; i++)
        gpuResult += h_odata[i];

    // CPU reference
    long long cpuResult = reduceCPU(h_idata, n);

    // Report
    double throughput = 1.0e-9 * ((double)n * sizeof(int)) / avgTimeSec;
    printf("Reduction, Throughput = %.4f GB/s, Time = %.5f s, Size = %d Elements\n",
           throughput, avgTimeSec, n);
    printf("GPU result = %lld\n", gpuResult);
    printf("CPU result = %lld\n", cpuResult);
    printf(gpuResult == cpuResult ? "Test passed\n" : "Test failed!\n");

    // Cleanup
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_idata));
    CUDA_CHECK(cudaFree(d_odata));
    delete[] h_idata;
    delete[] h_odata;

    return (gpuResult == cpuResult) ? 0 : 1;
}
