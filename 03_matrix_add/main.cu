#include <iostream>
#include <fmt/format.h>
#include <vector>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__ << ": " \
                      << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

__global__ void matrixAdd2D(const float *A, const float *B, float *C,
                            int height, int width)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < height && col < width) {
        int idx = row * width + col;
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    const int width = 1024, height = 1024;
    const int size = width * height * sizeof(float);

    std::vector<float> h_A(width * height, 1.0f);
    std::vector<float> h_B(width * height, 2.0f);
    std::vector<float> h_C(width * height, 0.0f);

    // 分配显存
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc((void **)&d_A, size));
    CUDA_CHECK(cudaMalloc((void **)&d_B, size));
    CUDA_CHECK(cudaMalloc((void **)&d_C, size));

    CUDA_CHECK( cudaMemcpy(d_A, h_A.data(), size, cudaMemcpyHostToDevice) );
    CUDA_CHECK( cudaMemcpy(d_B, h_B.data(), size, cudaMemcpyHostToDevice) );

    // 定义一个配置集合 来分析不同配置下的性能表现
    std::vector<dim3> block_configs = {
        dim3(8, 8), // 64线程
        dim3(16, 16), // 256线程 对称
        dim3(32, 8), // 256线程 非对称
        dim3(32, 32), // 1024线程
    };

    // cuda event 用于测量时间
    cudaEvent_t start, stop;
    CUDA_CHECK( cudaEventCreate(&start) );
    CUDA_CHECK( cudaEventCreate(&stop) );

    fmt::println("启动矩阵{}x{}加法性能测试", height, width);
    fmt::println("{:<15} {:<15} {:<15}", "Gird Size", "Block Size", "Time (ms)");

    for (const auto &block_size : block_configs) {
        dim3 grid_size(
            (width + block_size.x - 1) / block_size.x,
            (height + block_size.y - 1) / block_size.y
        );

        matrixAdd2D<<<grid_size, block_size>>>(d_A, d_B, d_C, width, height);
        CUDA_CHECK( cudaDeviceSynchronize() ); // 减少第一次运行的加载开销对时间测量的影响
        
        // 记录开始时间
        CUDA_CHECK( cudaEventRecord(start) ); // 异步
        matrixAdd2D<<<grid_size, block_size>>>(d_A, d_B, d_C, height, width);
        CUDA_CHECK( cudaEventRecord(stop) ); // 异步
        CUDA_CHECK( cudaEventSynchronize(stop) ); // 等待事件完成

        float milliseconds = 0;
        CUDA_CHECK( cudaEventElapsedTime(&milliseconds, start, stop) );

        fmt::println("{:<15} {:<15} {:<15.3f}", 
            fmt::format("({}, {})", grid_size.x, grid_size.y), 
            fmt::format("({}, {})", block_size.x, block_size.y), 
            milliseconds);

    }
    
    CUDA_CHECK( cudaEventDestroy(start) );
    CUDA_CHECK( cudaEventDestroy(stop) );
    CUDA_CHECK( cudaFree(d_A) );
    CUDA_CHECK( cudaFree(d_B) );
    CUDA_CHECK( cudaFree(d_C) );

    return 0;
}