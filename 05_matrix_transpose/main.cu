#include <iostream>
#include <vector>
#include <format>
#include <cuda_runtime.h>

// -------------------------------------------------------------------------
// 诊断宏定义
// -------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t const status = (call);                                  \
        if (status != cudaSuccess) {                                        \
            std::cerr << std::format("硬件状态异常诊断:\n  API: {}\n  错误: {}\n  位置: {}:{}\n", \
                                     #call, cudaGetErrorString(status), __FILE__, __LINE__); \
            std::exit(EXIT_FAILURE);                                        \
        }                                                                   \
    } while (0)

// -------------------------------------------------------------------------
// 设备端物理执行域：基础架构
// -------------------------------------------------------------------------

__global__ void transposeReadRowWriteCol(const float* __restrict__ A, float* __restrict__ B, int N) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < N && y < N) {
        B[x * N + y] = A[y * N + x];
    }
}

__global__ void transposeReadColWriteRow(const float* __restrict__ A, float* __restrict__ B, int N) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < N && y < N) {
        B[y * N + x] = A[x * N + y];
    }
}

// -------------------------------------------------------------------------
// 设备端物理执行域：共享内存架构 (消除双向跨步访存)
// -------------------------------------------------------------------------
constexpr int TILE_DIM = 32;
constexpr int BLOCK_ROWS = 8;

__global__ void transposeShared(const float* __restrict__ A, float* __restrict__ B, int N) {
    // 物理分配 SRAM，+1 偏移消除存储体冲突 (Bank Conflict)
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    if (x < N) {
        #pragma unroll
        for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
            if (y + j < N) {
                tile[threadIdx.y + j][threadIdx.x] = A[(y + j) * N + x];
            }
        }
    }

    __syncthreads();

    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    if (x < N) {
        #pragma unroll
        for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
            if (y + j < N) {
                B[(y + j) * N + x] = tile[threadIdx.x][threadIdx.y + j];
            }
        }
    }
}

// -------------------------------------------------------------------------
// 设备端物理执行域：展开架构 (展开因子 = 4，含严格边界诊断)
// -------------------------------------------------------------------------
__global__ void transposeRowUnroll4(const float * __restrict__ MatA, float * __restrict__ MatB, int N) {
    int ix = threadIdx.x + blockDim.x * blockIdx.x * 4;
    int iy = threadIdx.y + blockDim.y * blockIdx.y;

    if (iy < N) {
        int idx_row = ix + iy * N;
        int idx_col = ix * N + iy;

        if (ix < N) MatB[idx_col] = MatA[idx_row];
        if (ix + blockDim.x < N) MatB[idx_col + N * blockDim.x] = MatA[idx_row + blockDim.x];
        if (ix + 2 * blockDim.x < N) MatB[idx_col + N * 2 * blockDim.x] = MatA[idx_row + 2 * blockDim.x];
        if (ix + 3 * blockDim.x < N) MatB[idx_col + N * 3 * blockDim.x] = MatA[idx_row + 3 * blockDim.x];
    }
}

__global__ void transposeColUnroll4(const float * __restrict__ MatA, float * __restrict__ MatB, int N) {
    int ix = threadIdx.x + blockDim.x * blockIdx.x * 4;
    int iy = threadIdx.y + blockDim.y * blockIdx.y;

    if (iy < N) {
        int idx_row = ix + iy * N;
        int idx_col = ix * N + iy;

        if (ix < N) MatB[idx_row] = MatA[idx_col];
        if (ix + blockDim.x < N) MatB[idx_row + blockDim.x] = MatA[idx_col + N * blockDim.x];
        if (ix + 2 * blockDim.x < N) MatB[idx_row + 2 * blockDim.x] = MatA[idx_col + N * 2 * blockDim.x];
        if (ix + 3 * blockDim.x < N) MatB[idx_row + 3 * blockDim.x] = MatA[idx_col + N * 3 * blockDim.x];
    }
}

// -------------------------------------------------------------------------
// 主机端控制域
// -------------------------------------------------------------------------
int main() {
    const int N = 1024;
    const size_t elementCount = N * N;
    const size_t byteSize = elementCount * sizeof(float);

    std::vector<float> h_A(elementCount, 1.0f);
    std::vector<float> h_B(elementCount, 0.0f);

    float *d_A, *d_B;
    CUDA_CHECK(cudaMalloc((void**)&d_A, byteSize));
    CUDA_CHECK(cudaMalloc((void**)&d_B, byteSize));
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), byteSize, cudaMemcpyHostToDevice));

    // 1. 基础架构执行配置
    dim3 baseBlockSize(32, 16);
    dim3 baseGridSize((N + baseBlockSize.x - 1) / baseBlockSize.x, 
                      (N + baseBlockSize.y - 1) / baseBlockSize.y);

    // 2. 共享内存架构执行配置
    dim3 sharedBlockSize(TILE_DIM, BLOCK_ROWS);
    dim3 sharedGridSize((N + TILE_DIM - 1) / TILE_DIM, 
                        (N + TILE_DIM - 1) / TILE_DIM);

    // 3. 展开架构执行配置 (X 轴调度数量强制缩减至 1/4)
    dim3 unrollBlockSize(32, 16);
    dim3 unrollGridSize((N + unrollBlockSize.x * 4 - 1) / (unrollBlockSize.x * 4), 
                        (N + unrollBlockSize.y - 1) / unrollBlockSize.y);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    std::cout << std::format("启动矩阵规模 {} x {} 的硬件性能评估:\n", N, N);
    std::cout << std::string(75, '-') << "\n";
    std::cout << std::format("{:<45} | {:<15}\n", "执行执行架构", "物理耗时 (ms)");
    std::cout << std::string(75, '-') << "\n";

    // 测试单元宏定义，确保环境预热与同步的一致性
    #define RUN_TEST(kernel_name, grid, block, desc) do { \
        kernel_name<<<grid, block>>>(d_A, d_B, N); \
        CUDA_CHECK(cudaDeviceSynchronize()); \
        CUDA_CHECK(cudaEventRecord(start)); \
        kernel_name<<<grid, block>>>(d_A, d_B, N); \
        CUDA_CHECK(cudaEventRecord(stop)); \
        CUDA_CHECK(cudaEventSynchronize(stop)); \
        float ms = 0; \
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop)); \
        std::cout << std::format("{:<45} | {:<15.4f}\n", desc, ms); \
    } while(0)

    RUN_TEST(transposeReadRowWriteCol, baseGridSize, baseBlockSize, "方案 A (基础: 按行读 / 按列写)");
    RUN_TEST(transposeReadColWriteRow, baseGridSize, baseBlockSize, "方案 B (基础: 按列读 / 按行写)");
    RUN_TEST(transposeShared, sharedGridSize, sharedBlockSize, "方案 C (共享内存: 双向合并对齐)");
    RUN_TEST(transposeRowUnroll4, unrollGridSize, unrollBlockSize, "方案 D (展开 x4: 按行读 / 按列写)");
    RUN_TEST(transposeColUnroll4, unrollGridSize, unrollBlockSize, "方案 E (展开 x4: 按列读 / 按行写)");

    std::cout << std::string(75, '-') << "\n";

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));

    return 0;
}