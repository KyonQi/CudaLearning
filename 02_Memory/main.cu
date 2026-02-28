#include <fmt/format.h>
#include <iostream>

__global__ void kernel() {
    printf("Block ID: (%d, %d, %d), Thread ID: (%d, %d, %d)\n",
           blockIdx.x, blockIdx.y, blockIdx.z,
           threadIdx.x, threadIdx.y, threadIdx.z);
}

int main() {
    fmt::println("CUDA Block 和 Thread 数量说明");
    int n_elements = 32; // 总共的元素数量
    dim3 block_size(16); // 每个 Block 的线程数量
    dim3 grid_size((n_elements + block_size.x - 1) / block_size.x); // Grid 的 Block 数量，注意这里是取上确界
    fmt::println("总元素数量: {}", n_elements);
    fmt::println("每个 Block 的线程数量: {}", block_size.x);
    fmt::println("需要的 Block 数量: {}", grid_size.x);
    kernel<<<grid_size, block_size>>>();
    cudaDeviceSynchronize(); // 等待 GPU 完成
}