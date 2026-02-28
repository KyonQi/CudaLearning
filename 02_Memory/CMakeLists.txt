cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(CudaHelloWorld LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES native)

find_package(fmt REQUIRED)

add_executable(memory main.cu)

target_link_libraries(memory fmt::fmt)