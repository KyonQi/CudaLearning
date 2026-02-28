#include <iostream>
#include <fmt/format.h>

__global__ void helloFromGPU() {
    int threadId = threadIdx.x;
    int blockId = blockIdx.x;

    printf("Hello from GPU, thread ID: %d, block ID: %d\n", threadId, blockId);
}

int main() {
    std::cout << fmt::format("主机启动：准备 GPU 调度, \n");

    helloFromGPU<<<2, 4>>>();

    auto err = cudaDeviceSynchronize(); // wait for GPU finish
    if (cudaError_t::cudaSuccess != err) {
        std::cerr << fmt::format("CUDA 错误: {}\n", cudaGetErrorString(err));
        return 1;
    }

    std::cout << fmt::format("主机完成：GPU 调度完成, \n");
    
    return 0;
}