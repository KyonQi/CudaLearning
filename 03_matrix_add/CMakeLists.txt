cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(Cuda03 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES native) # 注意这会导致在不支持的GPU上编译失败，如果需要兼容性，请指定具体架构

find_package(fmt REQUIRED)

add_executable(matrix_add main.cu)

target_link_libraries(matrix_add fmt::fmt)