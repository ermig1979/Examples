cmake_minimum_required(VERSION 2.8)

file(GLOB_RECURSE SRC ./src/*.cpp)
set_source_files_properties(${SRC} PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -O3 -march=haswell")
add_executable(mmosbs ${SRC})
