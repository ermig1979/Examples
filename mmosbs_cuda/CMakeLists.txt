cmake_minimum_required(VERSION 2.8)
project(mmosbs)

find_package(CUDA QUIET REQUIRED)

include_directories(./src)

file(GLOB_RECURSE SRC_CU ./src/*.cu)
cuda_add_library(gemm ${SRC_CU})

file(GLOB_RECURSE SRC_CPP ./src/*.cpp)
set_source_files_properties(${SRC_CPP} PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -O3 -march=native")
add_executable(mmosbs ${SRC_CPP})
target_link_libraries(mmosbs gemm -lcuda -lcublas)

