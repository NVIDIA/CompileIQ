/*
 * Self-contained CUDA reduction benchmark for CompileIQ optimization.
 *
 * Uses cub::BlockReduce for the block-wide reduction and thrust::device_vector
 * for RAII device memory management.
 *
 * Compiles in a single step:
 *   nvcc -O3 -std=c++17 -arch=sm_100 reduction.cu -o reduction
 *
 * Usage:
 *   ./reduction [-n=67108864]
 */

#include <cub/block/block_reduce.cuh>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

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

// Reduction kernel using cub::BlockReduce.
// Each thread loads multiple elements via grid-stride loop (Brent's theorem)
// to keep the grid small, then CUB handles the block-wide reduction.
template <int BlockSize>
__global__ void reduce_kernel(const int *__restrict__ g_idata, int *__restrict__ g_odata,
                              unsigned int n)
{
    using BlockReduceT = cub::BlockReduce<int, BlockSize>;
    __shared__ typename BlockReduceT::TempStorage temp_storage;

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

    // Block-wide reduction via CUB
    int blockSum = BlockReduceT(temp_storage).Sum(mySum);

    if (threadIdx.x == 0)
        g_odata[blockIdx.x] = blockSum;
}

// CPU reference using simple summation
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

    // Initialize host data
    std::vector<int> h_idata(n);
    for (int i = 0; i < n; i++)
        h_idata[i] = rand() & 0xFF;

    // Compute grid dimensions
    int numBlocks = (n + (BLOCK_SIZE * 2 - 1)) / (BLOCK_SIZE * 2);
    if (numBlocks > MAX_BLOCKS) numBlocks = MAX_BLOCKS;

    // RAII device memory via thrust
    thrust::device_vector<int> d_idata(h_idata.begin(), h_idata.end());
    thrust::device_vector<int> d_odata(numBlocks);

    int *d_idata_ptr = thrust::raw_pointer_cast(d_idata.data());
    int *d_odata_ptr = thrust::raw_pointer_cast(d_odata.data());

    // Warm-up
    reduce_kernel<BLOCK_SIZE><<<numBlocks, BLOCK_SIZE, 0>>>(d_idata_ptr, d_odata_ptr, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timed iterations using CUDA events
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < TEST_ITERATIONS; i++)
        reduce_kernel<BLOCK_SIZE><<<numBlocks, BLOCK_SIZE, 0>>>(d_idata_ptr, d_odata_ptr, n);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float totalMs = 0;
    CUDA_CHECK(cudaEventElapsedTime(&totalMs, start, stop));
    double avgTimeSec = (totalMs / TEST_ITERATIONS) / 1000.0;

    // Read back block partial sums and finalize on CPU
    std::vector<int> h_odata(numBlocks);
    CUDA_CHECK(cudaMemcpy(h_odata.data(), d_odata_ptr, numBlocks * sizeof(int),
                          cudaMemcpyDeviceToHost));
    long long gpuResult = 0;
    for (int i = 0; i < numBlocks; i++)
        gpuResult += h_odata[i];

    // CPU reference
    long long cpuResult = reduceCPU(h_idata.data(), n);

    // Report
    double throughput = 1.0e-9 * ((double)n * sizeof(int)) / avgTimeSec;
    printf("Reduction, Throughput = %.4f GB/s, Time = %.5f s, Size = %d Elements\n",
           throughput, avgTimeSec, n);
    printf("GPU result = %lld\n", gpuResult);
    printf("CPU result = %lld\n", cpuResult);
    printf(gpuResult == cpuResult ? "Test passed\n" : "Test failed!\n");

    // Cleanup (events only — thrust handles device memory)
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return (gpuResult == cpuResult) ? 0 : 1;
}
