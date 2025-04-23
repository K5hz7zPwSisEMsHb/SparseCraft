cmake_minimum_required(VERSION 3.16)
project(spmv_perf LANGUAGES CXX)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置目标GPU架构（Ada Lovelace 8.9）
set(Kokkos_ARCH_ADA89 ON)


# 查找Kokkos和Kokkos Kernels安装路径
find_package(Kokkos REQUIRED)

find_package(KokkosKernels REQUIRED)

include_directories(include)

FILE (GLOB_RECURSE SRC_FILES src/*.cpp)

# 创建可执行文件
add_executable(kokkos_test main.cpp ${SRC_FILES})

# 链接库
target_link_libraries(kokkos_test
    PRIVATE
    Kokkos::kokkos
    KokkosKernels::kokkoskernels
)

# 生成compile_commands.json帮助IDE识别头文件
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
