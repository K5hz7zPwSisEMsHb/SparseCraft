cmake_minimum_required(VERSION 3.22)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_COMPILER "/usr/bin/g++")

set(CUDA_ARCHITECTURES 89)
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda")
set(CMAKE_CUDA_COMPILER "${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_${CUDA_ARCHITECTURES},code=sm_${CUDA_ARCHITECTURES} -ccbin /usr/bin -m64 -Xcompiler -fopenmp -Xcompiler -fopenmp-simd -g")

set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS})
set(CUDA_TOOLKIT_ROOT_DIR ${CUDA_TOOLKIT_ROOT_DIR})
set(CMAKE_CUDA_COMPILER ${CMAKE_CUDA_COMPILER})

message("CUDA_TOOLKIT_ROOT_DIR: ${CUDA_TOOLKIT_ROOT_DIR}")
message("CMAKE_CUDA_COMPILER: ${CMAKE_CUDA_COMPILER}")
message("CMAKE_CUDA_FLAGS: ${CMAKE_CUDA_FLAGS}")

add_definitions("${DEFINITIONS}")

project(BLAS VERSION 1.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast")

find_package(OpenMP REQUIRED)

find_library(CUSPARSE_LIBRARY cusparse
    HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib64
    NO_DEFAULT_PATH
)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

FILE(
    GLOB SOURCES 
    "${CMAKE_CURRENT_SOURCE_DIR}/src/tmatrix/*.cu" 
    "${CMAKE_CURRENT_SOURCE_DIR}/src/tmatrix/*.cpp" 
    "${CMAKE_CURRENT_SOURCE_DIR}/src/tmatrix/*/*.cu" 
    "${CMAKE_CURRENT_SOURCE_DIR}/src/tmatrix/*/*.cpp" 
)
add_executable(${PROJECT_NAME} main.cu ${SOURCES})


set(CMAKE_CUDA_NVCC_FLAGS ${CMAKE_CUDA_FLAGS})
target_link_libraries(${PROJECT_NAME}
    OpenMP::OpenMP_CXX
    ${CUSPARSE_LIBRARY}
)
