#include <iostream>
#include <vector>
#include <cmath>

__global__ void addVectors(const float *A, const float *B, float *C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    const int N = 1000000;
    const size_t size = N * sizeof(float);

    std::vector<float> h_A(N, 1.0f); // Initialize A with 1.0
    std::vector<float> h_B(N, 2.0f); // Initialize B
    std::vector<float> h_C(N, 0.0f); // Result vector

    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, h_A.data(), size, cudaMemcpyKind::cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), size, cudaMemcpyKind::cudaMemcpyHostToDevice);
    // cudaMemcpy(d_C, h_C.data(), size, cudaMemcpyKind::cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    addVectors<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);

    cudaMemcpy(h_C.data(), d_C, size, cudaMemcpyKind::cudaMemcpyDeviceToHost);

    bool hasError = false;
    for (int i = 0; i < N; ++i) {
        if (std::fabs(h_C[i] - 3.0f) > 1e-5) {
            std::cerr << "Error at index " << i << ": " << h_C[i] << " != 3.0" << std::endl;
            hasError = true;
            break;
        }
    }
    if (!hasError) {
        std::cout << "All results are correct!" << std::endl;
    }
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return 0;
}