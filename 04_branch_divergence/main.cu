#include <iostream>
#include <vector>
#include <numeric>
#include <cuda_runtime.h>
#include <fmt/core.h>
#include <fmt/color.h>

#define CHECK_CUDA(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fmt::print(stderr, "CUDA error at {}:{}: {}\n", __FILE__, __LINE__, \
                       cudaGetErrorString(error)); \
            exit(1); \
        } \
    } while(0)

// CPU reduction for verification
int recursiveReduce(int *data, int const size)
{
    if (size == 1)
        return data[0];
    
    int const stride = size / 2;
    if (size % 2 == 1)
    {
        for (int i = 0; i < stride; i++)
        {
            data[i] += data[i + stride];
        }
        data[0] += data[size - 1];
    }
    else
    {
        for (int i = 0; i < stride; i++)
        {
            data[i] += data[i + stride];
        }
    }
    return recursiveReduce(data, stride);
}

// 增加计算强度的版本 - 有分支发散
__global__ void reduceWithBranchIntensive(int *g_idata, int *g_odata, unsigned int n)
{
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= n) return;
    
    extern __shared__ int sdata[];
    sdata[tid] = g_idata[idx];
    __syncthreads();
    
    for (unsigned int s = 1; s < blockDim.x; s *= 2)
    {
        if (tid % (2 * s) == 0)
        {
            // 增加计算强度：做一些额外的计算
            int temp = sdata[tid];
            for (int i = 0; i < 100; i++)
            {
                temp = (temp + sdata[tid + s]) / 2;
            }
            sdata[tid] = temp;
        }
        // else
        // {
        //     // 空闲线程也做一些计算，但不影响结果
        //     int dummy = sdata[tid];
        //     for (int i = 0; i < 50; i++)
        //     {
        //         dummy = (dummy * 7 + 13) % 1000000;
        //     }
        //     sdata[tid] = dummy;  // 这会被覆盖，只是为了防止编译器优化掉
        // }
        __syncthreads();
    }
    
    if (tid == 0)
    {
        g_odata[blockIdx.x] = sdata[0];
    }
}

// 无分支发散版本
__global__ void reduceWithoutBranchIntensive(int *g_idata, int *g_odata, unsigned int n)
{
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= n) return;
    
    extern __shared__ int sdata[];
    sdata[tid] = g_idata[idx];
    __syncthreads();
    
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            // 相同的计算强度
            int temp = sdata[tid];
            for (int i = 0; i < 100; i++)
            {
                temp = (temp + sdata[tid + s]) / 2;
            }
            sdata[tid] = temp;
        }
        __syncthreads();
    }
    
    if (tid == 0)
    {
        g_odata[blockIdx.x] = sdata[0];
    }
}

// 原始简单版本 - 有分支
__global__ void reduceWithBranch(int *g_idata, int *g_odata, unsigned int n)
{
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= n) return;
    
    extern __shared__ int sdata[];
    sdata[tid] = g_idata[idx];
    __syncthreads();
    
    for (unsigned int s = 1; s < blockDim.x; s *= 2)
    {
        if (tid % (2 * s) == 0)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0)
    {
        g_odata[blockIdx.x] = sdata[0];
    }
}

// 原始简单版本 - 无分支
__global__ void reduceWithoutBranch(int *g_idata, int *g_odata, unsigned int n)
{
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= n) return;
    
    extern __shared__ int sdata[];
    sdata[tid] = g_idata[idx];
    __syncthreads();
    
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0)
    {
        g_odata[blockIdx.x] = sdata[0];
    }
}

// 优化版本 1: 循环展开（最后的 warp 不需要同步）
__global__ void reduceUnrolled(int *g_idata, int *g_odata, unsigned int n)
{
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= n) return;
    
    extern __shared__ int sdata[];
    sdata[tid] = g_idata[idx];
    __syncthreads();
    
    // 常规 reduction，直到 s = 32
    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1)
    {
        if (tid < s)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // 最后一个 warp 的展开（不需要 __syncthreads）
    // warp 内的线程是 SIMT 同步的
    if (tid < 32)
    {
        volatile int* smem = sdata;  // 使用 volatile 防止编译器优化
        if (blockDim.x >= 64) smem[tid] += smem[tid + 32];
        if (blockDim.x >= 32) smem[tid] += smem[tid + 16];
        if (blockDim.x >= 16) smem[tid] += smem[tid + 8];
        if (blockDim.x >= 8)  smem[tid] += smem[tid + 4];
        if (blockDim.x >= 4)  smem[tid] += smem[tid + 2];
        if (blockDim.x >= 2)  smem[tid] += smem[tid + 1];
    }
    
    if (tid == 0)
    {
        g_odata[blockIdx.x] = sdata[0];
    }
}

// 优化版本 2: 每个 Block 处理多个数据块（增加每个线程的工作量）
__global__ void reduceMultiBlock(int *g_idata, int *g_odata, unsigned int n)
{
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    
    extern __shared__ int sdata[];
    
    // 每个线程加载并累加多个元素
    int sum = 0;
    if (idx < n) sum += g_idata[idx];
    if (idx + blockDim.x < n) sum += g_idata[idx + blockDim.x];
    
    sdata[tid] = sum;
    __syncthreads();
    
    // 标准 reduction
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0)
    {
        g_odata[blockIdx.x] = sdata[0];
    }
}

// 优化版本 3: 完全优化（循环展开 + 多块处理）
__global__ void reduceFullyOptimized(int *g_idata, int *g_odata, unsigned int n)
{
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    
    extern __shared__ int sdata[];
    
    // 每个线程加载并累加多个元素
    int sum = 0;
    if (idx < n) sum += g_idata[idx];
    if (idx + blockDim.x < n) sum += g_idata[idx + blockDim.x];
    
    sdata[tid] = sum;
    __syncthreads();
    
    // 常规 reduction，直到 s = 32
    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1)
    {
        if (tid < s)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // 最后一个 warp 的展开
    if (tid < 32)
    {
        volatile int* smem = sdata;
        if (blockDim.x >= 64) smem[tid] += smem[tid + 32];
        if (blockDim.x >= 32) smem[tid] += smem[tid + 16];
        if (blockDim.x >= 16) smem[tid] += smem[tid + 8];
        if (blockDim.x >= 8)  smem[tid] += smem[tid + 4];
        if (blockDim.x >= 4)  smem[tid] += smem[tid + 2];
        if (blockDim.x >= 2)  smem[tid] += smem[tid + 1];
    }
    
    if (tid == 0)
    {
        g_odata[blockIdx.x] = sdata[0];
    }
}

// 优化版本 4: 更激进的多块处理（每个线程处理 4 个元素）
__global__ void reduceMultiBlock4(int *g_idata, int *g_odata, unsigned int n)
{
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * (blockDim.x * 4) + threadIdx.x;
    
    extern __shared__ int sdata[];
    
    // 每个线程加载并累加 4 个元素
    int sum = 0;
    for (int i = 0; i < 4; i++)
    {
        unsigned int offset = idx + i * blockDim.x;
        if (offset < n)
            sum += g_idata[offset];
    }
    
    sdata[tid] = sum;
    __syncthreads();
    
    // 常规 reduction，直到 s = 32
    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1)
    {
        if (tid < s)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // 最后一个 warp 的展开
    if (tid < 32)
    {
        volatile int* smem = sdata;
        if (blockDim.x >= 64) smem[tid] += smem[tid + 32];
        if (blockDim.x >= 32) smem[tid] += smem[tid + 16];
        if (blockDim.x >= 16) smem[tid] += smem[tid + 8];
        if (blockDim.x >= 8)  smem[tid] += smem[tid + 4];
        if (blockDim.x >= 4)  smem[tid] += smem[tid + 2];
        if (blockDim.x >= 2)  smem[tid] += smem[tid + 1];
    }
    
    if (tid == 0)
    {
        g_odata[blockIdx.x] = sdata[0];
    }
}

template<typename KernelFunc>
float benchmarkKernel(int *d_idata, int *d_odata, unsigned int n, 
                      unsigned int gridSize, unsigned int blockSize,
                      KernelFunc kernel, int iterations = 1000)
{
    // Warm up
    kernel<<<gridSize, blockSize, blockSize * sizeof(int)>>>(d_idata, d_odata, n);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iterations; i++)
    {
        kernel<<<gridSize, blockSize, blockSize * sizeof(int)>>>(d_idata, d_odata, n);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    
    float milliseconds = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
    
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    
    return milliseconds / iterations;
}

int verifyResult(int *d_odata, unsigned int gridSize, int expected)
{
    std::vector<int> h_odata(gridSize);
    CHECK_CUDA(cudaMemcpy(h_odata.data(), d_odata, gridSize * sizeof(int), 
                          cudaMemcpyDeviceToHost));
    
    int result = 0;
    for (unsigned int i = 0; i < gridSize; i++)
    {
        result += h_odata[i];
    }
    return result;
}

int main()
{
    fmt::print(fmt::fg(fmt::color::cyan), "=== CUDA Branch Divergence Deep Analysis ===\n\n");
    
    // 打印 GPU 信息
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    fmt::print("GPU: {}\n", prop.name);
    fmt::print("Compute Capability: {}.{}\n", prop.major, prop.minor);
    fmt::print("Warp Size: {}\n\n", prop.warpSize);
    
    // 使用更大的数据量
    unsigned int n = 1 << 24;  // 16M elements
    fmt::print("Array size: {} elements ({:.2f} MB)\n\n", n, n * sizeof(int) / 1024.0 / 1024.0);
    
    std::vector<int> h_idata(n, 1);
    int expected_result = n;  // 所有元素都是 1，所以和就是 n
    
    int *d_idata, *d_odata;
    unsigned int blockSize = 256;
    unsigned int gridSize = (n + blockSize - 1) / blockSize;
    unsigned int gridSize2 = (n + blockSize * 2 - 1) / (blockSize * 2);
    unsigned int gridSize4 = (n + blockSize * 4 - 1) / (blockSize * 4);
    unsigned int maxGridSize = gridSize;  // 使用最大的那个
    if (gridSize2 > maxGridSize) maxGridSize = gridSize2;
    if (gridSize4 > maxGridSize) maxGridSize = gridSize4;
    
    CHECK_CUDA(cudaMalloc(&d_idata, n * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_odata, maxGridSize * sizeof(int)));
    CHECK_CUDA(cudaMemcpy(d_idata, h_idata.data(), n * sizeof(int), cudaMemcpyHostToDevice));
    
    fmt::print(fmt::fg(fmt::color::yellow), "Test 1: Simple reduction (original)\n");
    fmt::print("--------------------------------------\n");
    
    float time1 = benchmarkKernel(d_idata, d_odata, n, gridSize, blockSize, 
                                  reduceWithBranch, 1000);
    int result1 = verifyResult(d_odata, gridSize, expected_result);
    fmt::print("With branch divergence:    {:.4f} ms (Result: {}, {})\n", 
               time1, result1, result1 == expected_result ? "✓" : "✗");
    
    float time2 = benchmarkKernel(d_idata, d_odata, n, gridSize, blockSize, 
                                  reduceWithoutBranch, 1000);
    int result2 = verifyResult(d_odata, gridSize, expected_result);
    fmt::print("Without branch divergence: {:.4f} ms (Result: {}, {})\n", 
               time2, result2, result2 == expected_result ? "✓" : "✗");
    fmt::print("Speedup: {:.2f}x\n\n", time1 / time2);
    
    fmt::print(fmt::fg(fmt::color::green), "Test 2: Loop unrolling optimization\n");
    fmt::print("--------------------------------------\n");
    
    float time_unrolled = benchmarkKernel(d_idata, d_odata, n, gridSize, blockSize, 
                                          reduceUnrolled, 1000);
    int result_unrolled = verifyResult(d_odata, gridSize, expected_result);
    fmt::print("Unrolled kernel:           {:.4f} ms (Result: {}, {})\n", 
               time_unrolled, result_unrolled, result_unrolled == expected_result ? "✓" : "✗");
    fmt::print("Speedup vs baseline:       {:.2f}x\n\n", time2 / time_unrolled);
    
    fmt::print(fmt::fg(fmt::color::magenta), "Test 3: Multi-block processing\n");
    fmt::print("--------------------------------------\n");
    
    float time_multi2 = benchmarkKernel(d_idata, d_odata, n, gridSize2, blockSize, 
                                        reduceMultiBlock, 1000);
    int result_multi2 = verifyResult(d_odata, gridSize2, expected_result);
    fmt::print("Multi-block (2x):          {:.4f} ms (Grid: {}, Result: {}, {})\n", 
               time_multi2, gridSize2, result_multi2, result_multi2 == expected_result ? "✓" : "✗");
    fmt::print("Speedup vs baseline:       {:.2f}x\n\n", time2 / time_multi2);
    
    float time_multi4 = benchmarkKernel(d_idata, d_odata, n, gridSize4, blockSize, 
                                        reduceMultiBlock4, 1000);
    int result_multi4 = verifyResult(d_odata, gridSize4, expected_result);
    fmt::print("Multi-block (4x):          {:.4f} ms (Grid: {}, Result: {}, {})\n", 
               time_multi4, gridSize4, result_multi4, result_multi4 == expected_result ? "✓" : "✗");
    fmt::print("Speedup vs baseline:       {:.2f}x\n\n", time2 / time_multi4);
    
    fmt::print(fmt::fg(fmt::color::blue), "Test 4: Fully optimized kernel\n");
    fmt::print("--------------------------------------\n");
    
    float time_optimized = benchmarkKernel(d_idata, d_odata, n, gridSize2, blockSize, 
                                          reduceFullyOptimized, 1000);
    int result_optimized = verifyResult(d_odata, gridSize2, expected_result);
    fmt::print("Fully optimized:           {:.4f} ms (Result: {}, {})\n", 
               time_optimized, result_optimized, result_optimized == expected_result ? "✓" : "✗");
    fmt::print("Speedup vs baseline:       {:.2f}x\n", time2 / time_optimized);
    fmt::print("Speedup vs with-branch:    {:.2f}x\n\n", time1 / time_optimized);
    
    fmt::print(fmt::fg(fmt::color::green), "Test 5: Compute-intensive reduction\n");
    fmt::print("--------------------------------------\n");
    
    float time3 = benchmarkKernel(d_idata, d_odata, n, gridSize, blockSize, 
                                  reduceWithBranchIntensive, 100);
    fmt::print("With branch divergence:    {:.4f} ms\n", time3);
    
    float time4 = benchmarkKernel(d_idata, d_odata, n, gridSize, blockSize, 
                                  reduceWithoutBranchIntensive, 100);
    fmt::print("Without branch divergence: {:.4f} ms\n", time4);
    fmt::print("Speedup: {:.2f}x\n\n", time3 / time4);
    
    CHECK_CUDA(cudaFree(d_idata));
    CHECK_CUDA(cudaFree(d_odata));
    
    fmt::print(fmt::fg(fmt::color::cyan), "=== Key Optimization Techniques ===\n");
    fmt::print("1. Avoid branch divergence: Use tid < s instead of tid %% (2*s) == 0\n");
    fmt::print("2. Loop unrolling: Last warp doesn't need __syncthreads()\n");
    fmt::print("3. Multi-block processing: Each thread handles multiple elements\n");
    fmt::print("4. Reduce kernel launches: Fewer blocks = less overhead\n\n");
    
    fmt::print(fmt::fg(fmt::color::yellow), "Performance Factors:\n");
    fmt::print("- Branch divergence impact depends on compute intensity\n");
    fmt::print("- Modern GPUs handle simple branches better\n");
    fmt::print("- Multi-block processing reduces grid size and overhead\n");
    fmt::print("- Loop unrolling eliminates unnecessary synchronization\n");
    
    return 0;
}