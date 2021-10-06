cmake_minimum_required(VERSION 3.10) 
project(deepstream_decode)

find_package(PkgConfig REQUIRED)

set(CUDA_VER "11.1")
set(NVDS_VER "5.1")

pkg_check_modules(GST REQUIRED gstreamer-1.0)
include_directories(${GST_INCLUDE_DIRS})
link_directories(${GST_LIBRARY_DIRS})

set(CUDA_DIR "/usr/local/cuda-${CUDA_VER}")
set(NVDS_DIR "/opt/nvidia/deepstream/deepstream-${NVDS_VER}")

set(APP_NAME "deepstream_decode")

set(LIB_INSTALL_DIR "${NVDS_DIR}/lib/")
set(APP_INSTALL_DIR "${NVDS_DIR}/bin/")
set(WORK_DIR "${NVDS_DIR}/sources/apps/sample_apps/${APP_NAME}")

file(GLOB APP_C_SRC ./*.c)
file(GLOB APP_CPP_SRC ./*.cpp)

include_directories(${NVDS_DIR}/sources/includes)
include_directories(${CUDA_DIR}/include)
include_directories(..)

add_executable(${APP_NAME} ${APP_C_SRC} ${APP_CPP_SRC})
target_link_libraries(${APP_NAME} ${GST_LIBRARIES} -L${CUDA_DIR}/lib64/ -lcudart -L${LIB_INSTALL_DIR} -lnvdsgst_meta -lnvds_meta -lcuda -Wl,-rpath,${LIB_INSTALL_DIR})

#add_custom_target(build-time-make-directory ALL COMMAND ${CMAKE_COMMAND} -E make_directory ${directory})
add_custom_command(TARGET ${APP_NAME} POST_BUILD COMMAND "${CMAKE_COMMAND}" -E make_directory "${WORK_DIR}" COMMENT "Create work directory")
add_custom_command(TARGET ${APP_NAME} POST_BUILD COMMAND "${CMAKE_COMMAND}" -E copy "${CMAKE_BINARY_DIR}/${APP_NAME}" "${WORK_DIR}" COMMENT "Copying binaries")




