cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

# set(CMAKE_C_COMPILER "/usr/bin/clang-20")
# set(CMAKE_CXX_COMPILER "/usr/bin/clang++-20")
# set(CMAKE_CUDA_COMPILER "/usr/bin/clang++-20")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES native)


project(CudaVectorAdd LANGUAGES CXX CUDA)



add_executable(vector_add main.cu)