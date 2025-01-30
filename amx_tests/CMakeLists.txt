cmake_minimum_required(VERSION 3.7)
project(amx_tests)

file(GLOB_RECURSE SRC ./src/*.cpp)
set_source_files_properties(${SRC} PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -O3 -std=c++17 -fmax-errors=5 -mavx2 -mfma -mavx512f -mavx512cd -mavx512bw -mavx512vl -mavx512dq -mavx512vnni -mavx512vbmi -mavx512bf16 -mamx-tile -mamx-int8 -mamx-bf16")
add_executable(amx_tests ${SRC})
