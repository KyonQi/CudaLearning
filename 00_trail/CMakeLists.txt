cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

# set(CMAKE_C_COMPILER "/usr/bin/clang-20")
# set(CMAKE_CXX_COMPILER "/usr/bin/clang++-20")
# set(CMAKE_CUDA_COMPILER "/usr/bin/clang++-20")

project(CudaVectorAdd LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES native)

find_package(fmt)

add_executable(vector_add main.cu)

target_link_libraries(vector_add fmt::fmt)