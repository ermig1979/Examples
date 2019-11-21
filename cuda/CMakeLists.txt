cmake_minimum_required(VERSION 2.8)
find_package(CUDA QUIET REQUIRED)

include_directories(kernels utility)

file(GLOB_RECURSE SRC_CU ./src/*.cu)
cuda_add_library(gemm ${SRC_CU})

file(GLOB_RECURSE SRC_CPP ./src/*.cpp)
set_source_files_properties(${SRC_CPP} PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -O3 -march=haswell")
add_executable(mmosbs ${SRC_CPP})
target_link_libraries(mmosbs gemm -lcuda)

# Specify include directories
#include_directories(kernels utility ../NVIDIA_CUDA-10.1_Samples/common/inc)
#cuda_add_executable(cuda_printf src/simplePrintf.cu)

