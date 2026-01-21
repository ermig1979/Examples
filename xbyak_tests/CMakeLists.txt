cmake_minimum_required(VERSION 3.10)
project(xbyak_tests)

include_directories(..)
file(GLOB_RECURSE SRC ./tests/*.cpp)
set_source_files_properties(${SRC} PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -O3 -march=native")
add_executable(xbyak_tests ${SRC})
